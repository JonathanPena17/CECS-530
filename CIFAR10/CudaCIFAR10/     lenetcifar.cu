// lenet5_cifar10_train.cu
// LeNet-5 Training on CIFAR-10 Dataset - CUDA C++ Implementation
// Compile: nvcc -o lenet5_cifar10 lenet5_cifar10_train.cu -O3 -std=c++11

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Network architecture for CIFAR-10 (32x32x3 RGB input)
#define INPUT_SIZE 32
#define INPUT_CHANNELS 3  // RGB
#define C1_FILTERS 6
#define C1_SIZE 28
#define S2_SIZE 14
#define C3_FILTERS 16
#define C3_SIZE 10
#define S4_SIZE 5
#define C5_SIZE 120
#define F6_SIZE 84
#define OUTPUT_SIZE 10
#define CONV_KERNEL 5
#define POOL_SIZE 2

// Training hyperparameters
#define BATCH_SIZE 1
#define LEARNING_RATE 0.00001f
#define EPOCHS 40
#define TRAIN_SIZE 50000
#define TEST_SIZE 10000

// ==================== CIFAR-10 Data Loading ====================

typedef struct {
    float* images;      // [N, 3, 32, 32] flattened
    unsigned char* labels;
    int count;
} CIFAR10Data;

// Load single CIFAR-10 batch file (binary format)
void load_cifar10_batch(const char* filename, float* images, unsigned char* labels, int start_idx) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Cannot open file %s\n", filename);
        exit(1);
    }
    
    // CIFAR-10 binary format: [label (1 byte)][red (1024 bytes)][green (1024 bytes)][blue (1024 bytes)]
    // Each file contains 10,000 images
    unsigned char buffer[3073]; // 1 label + 3072 pixels (32x32x3)
    
    int idx = 0;
    while (fread(buffer, sizeof(unsigned char), 3073, file) == 3073) {
        int global_idx = start_idx + idx;
        
        // Store label
        labels[global_idx] = buffer[0];
        
        // Store image data: convert from [R...R G...G B...B] to [C, H, W] format
        // and normalize to [0, 1]
        for (int c = 0; c < 3; c++) {
            for (int h = 0; h < 32; h++) {
                for (int w = 0; w < 32; w++) {
                    int pixel_idx = 1 + c * 1024 + h * 32 + w;
                    int out_idx = global_idx * (3 * 32 * 32) + c * (32 * 32) + h * 32 + w;
                    images[out_idx] = buffer[pixel_idx] / 255.0f;
                }
            }
        }
        idx++;
    }
    
    fclose(file);
    printf("Loaded %d images from %s\n", idx, filename);
}

// Load all CIFAR-10 training data (5 batch files)
CIFAR10Data load_cifar10_train(const char* data_dir) {
    CIFAR10Data data;
    data.count = TRAIN_SIZE;
    data.images = (float*)malloc(TRAIN_SIZE * 3 * 32 * 32 * sizeof(float));
    data.labels = (unsigned char*)malloc(TRAIN_SIZE * sizeof(unsigned char));
    
    char filename[256];
    for (int i = 1; i <= 5; i++) {
        sprintf(filename, "%s/data_batch_%d.bin", data_dir, i);
        load_cifar10_batch(filename, data.images, data.labels, (i - 1) * 10000);
    }
    
    return data;
}

// Load CIFAR-10 test data
CIFAR10Data load_cifar10_test(const char* data_dir) {
    CIFAR10Data data;
    data.count = TEST_SIZE;
    data.images = (float*)malloc(TEST_SIZE * 3 * 32 * 32 * sizeof(float));
    data.labels = (unsigned char*)malloc(TEST_SIZE * sizeof(unsigned char));
    
    char filename[256];
    sprintf(filename, "%s/test_batch.bin", data_dir);
    load_cifar10_batch(filename, data.images, data.labels, 0);
    
    return data;
}

// ==================== CUDA Kernels ====================

// Tanh activation
__global__ void tanh_forward(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = tanhf(data[idx]);
    }
}

__global__ void tanh_backward(float* grad, const float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = output[idx];
        grad[idx] *= (1.0f - val * val);
    }
}

// ReLU activation (alternative to tanh)
__global__ void relu_forward(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

// Convolution forward pass
__global__ void conv2d_forward(
    const float* input, const float* weight, const float* bias,
    float* output, int in_channels, int out_channels,
    int input_size, int output_size, int kernel_size
) {
    int out_ch = blockIdx.x;
    int out_y = blockIdx.y;
    int out_x = threadIdx.x;
    
    if (out_ch >= out_channels || out_y >= output_size || out_x >= output_size) return;
    
    float sum = 0.0f;
    for (int in_ch = 0; in_ch < in_channels; in_ch++) {
        for (int ky = 0; ky < kernel_size; ky++) {
            for (int kx = 0; kx < kernel_size; kx++) {
                int in_y = out_y + ky;
                int in_x = out_x + kx;
                int in_idx = in_ch * input_size * input_size + in_y * input_size + in_x;
                int w_idx = ((out_ch * in_channels + in_ch) * kernel_size + ky) * kernel_size + kx;
                sum += input[in_idx] * weight[w_idx];
            }
        }
    }
    
    if (bias != NULL) sum += bias[out_ch];
    
    int out_idx = out_ch * output_size * output_size + out_y * output_size + out_x;
    output[out_idx] = sum;
}

// Max pooling forward
__global__ void maxpool2d_forward(
    const float* input, float* output, int* indices,
    int channels, int input_size, int output_size
) {
    int ch = blockIdx.x;
    int out_y = blockIdx.y;
    int out_x = threadIdx.x;
    
    if (ch >= channels || out_y >= output_size || out_x >= output_size) return;
    
    int in_y = out_y * 2;
    int in_x = out_x * 2;
    
    float max_val = -1e10f;
    int max_idx = 0;
    
    for (int dy = 0; dy < 2; dy++) {
        for (int dx = 0; dx < 2; dx++) {
            int in_idx = ch * input_size * input_size + (in_y + dy) * input_size + (in_x + dx);
            if (input[in_idx] > max_val) {
                max_val = input[in_idx];
                max_idx = dy * 2 + dx;
            }
        }
    }
    
    int out_idx = ch * output_size * output_size + out_y * output_size + out_x;
    output[out_idx] = max_val;
    if (indices) indices[out_idx] = max_idx;
}

// Fully connected forward
__global__ void fc_forward(
    const float* input, const float* weight, const float* bias,
    float* output, int input_size, int output_size
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= output_size) return;
    
    float sum = 0.0f;
    for (int i = 0; i < input_size; i++) {
        sum += input[i] * weight[out_idx * input_size + i];
    }
    if (bias != NULL) sum += bias[out_idx];
    output[out_idx] = sum;
}

// Softmax
__global__ void softmax_forward(float* data, int size) {
    __shared__ float max_val;
    __shared__ float sum_exp;
    
    if (threadIdx.x == 0) {
        max_val = data[0];
        for (int i = 1; i < size; i++) {
            max_val = fmaxf(max_val, data[i]);
        }
    }
    __syncthreads();
    
    int idx = threadIdx.x;
    if (idx < size) {
        data[idx] = expf(data[idx] - max_val);
    }
    __syncthreads();
    
    if (threadIdx.x == 0) {
        sum_exp = 0.0f;
        for (int i = 0; i < size; i++) {
            sum_exp += data[i];
        }
    }
    __syncthreads();
    
    if (idx < size) {
        data[idx] /= sum_exp;
    }
}

// Gradient kernels
__global__ void fc_backward_weight(
    const float* input, const float* grad_output,
    float* grad_weight, int input_size, int output_size, float lr
) {
    int out_idx = blockIdx.x;
    int in_idx = threadIdx.x;
    
    if (out_idx >= output_size || in_idx >= input_size) return;
    
    int w_idx = out_idx * input_size + in_idx;
    atomicAdd(&grad_weight[w_idx], -lr * grad_output[out_idx] * input[in_idx]);
}

__global__ void fc_backward_bias(
    const float* grad_output, float* bias, int output_size, float lr
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= output_size) return;
    atomicAdd(&bias[idx], -lr * grad_output[idx]);
}

__global__ void fc_backward_input(
    const float* grad_output, const float* weight,
    float* grad_input, int input_size, int output_size
) {
    int in_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (in_idx >= input_size) return;
    
    float sum = 0.0f;
    for (int out_idx = 0; out_idx < output_size; out_idx++) {
        sum += grad_output[out_idx] * weight[out_idx * input_size + in_idx];
    }
    grad_input[in_idx] = sum;
}

// Convolution backward - compute weight gradients
__global__ void conv2d_backward_weight(
    const float* input, const float* grad_output,
    float* grad_weight, int in_channels, int out_channels,
    int input_size, int output_size, int kernel_size, float lr
) {
    int out_ch = blockIdx.x;
    int in_ch = blockIdx.y;
    int ky = blockIdx.z;
    int kx = threadIdx.x;
    
    if (out_ch >= out_channels || in_ch >= in_channels || 
        ky >= kernel_size || kx >= kernel_size) return;
    
    float grad_sum = 0.0f;
    
    // Accumulate gradient over all output positions
    for (int out_y = 0; out_y < output_size; out_y++) {
        for (int out_x = 0; out_x < output_size; out_x++) {
            int in_y = out_y + ky;
            int in_x = out_x + kx;
            
            int in_idx = in_ch * input_size * input_size + in_y * input_size + in_x;
            int out_idx = out_ch * output_size * output_size + out_y * output_size + out_x;
            
            grad_sum += input[in_idx] * grad_output[out_idx];
        }
    }
    
    int w_idx = ((out_ch * in_channels + in_ch) * kernel_size + ky) * kernel_size + kx;
    atomicAdd(&grad_weight[w_idx], -lr * grad_sum);
}

// Convolution backward - compute bias gradients
__global__ void conv2d_backward_bias(
    const float* grad_output, float* bias,
    int out_channels, int output_size, float lr
) {
    int out_ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_ch >= out_channels) return;
    
    float grad_sum = 0.0f;
    for (int y = 0; y < output_size; y++) {
        for (int x = 0; x < output_size; x++) {
            int idx = out_ch * output_size * output_size + y * output_size + x;
            grad_sum += grad_output[idx];
        }
    }
    
    atomicAdd(&bias[out_ch], -lr * grad_sum);
}

// Convolution backward - propagate gradients to input
__global__ void conv2d_backward_input(
    const float* grad_output, const float* weight, float* grad_input,
    int in_channels, int out_channels, int input_size, int output_size, int kernel_size
) {
    int in_ch = blockIdx.x;
    int in_y = blockIdx.y;
    int in_x = threadIdx.x;
    
    if (in_ch >= in_channels || in_y >= input_size || in_x >= input_size) return;
    
    float grad_sum = 0.0f;
    
    // Sum contributions from all output positions that used this input
    for (int out_ch = 0; out_ch < out_channels; out_ch++) {
        for (int ky = 0; ky < kernel_size; ky++) {
            for (int kx = 0; kx < kernel_size; kx++) {
                int out_y = in_y - ky;
                int out_x = in_x - kx;
                
                // Check if this input contributes to a valid output position
                if (out_y >= 0 && out_y < output_size && out_x >= 0 && out_x < output_size) {
                    int out_idx = out_ch * output_size * output_size + out_y * output_size + out_x;
                    int w_idx = ((out_ch * in_channels + in_ch) * kernel_size + ky) * kernel_size + kx;
                    grad_sum += grad_output[out_idx] * weight[w_idx];
                }
            }
        }
    }
    
    int in_idx = in_ch * input_size * input_size + in_y * input_size + in_x;
    grad_input[in_idx] = grad_sum;
}

// Max pooling backward
__global__ void maxpool2d_backward(
    const float* grad_output, const int* indices, float* grad_input,
    int channels, int input_size, int output_size
) {
    int ch = blockIdx.x;
    int out_y = blockIdx.y;
    int out_x = threadIdx.x;
    
    if (ch >= channels || out_y >= output_size || out_x >= output_size) return;
    
    int out_idx = ch * output_size * output_size + out_y * output_size + out_x;
    int max_idx = indices[out_idx];
    
    int dy = max_idx / 2;
    int dx = max_idx % 2;
    int in_y = out_y * 2 + dy;
    int in_x = out_x * 2 + dx;
    
    int in_idx = ch * input_size * input_size + in_y * input_size + in_x;
    atomicAdd(&grad_input[in_idx], grad_output[out_idx]);
}

// ReLU backward (derivative)
__global__ void relu_backward(float* grad, const float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad[idx] *= (output[idx] > 0.0f) ? 1.0f : 0.0f;
    }
}


// ==================== LeNet-5 Model ====================

typedef struct {
    // Layer outputs (forward)
    float *d_input;  // [3, 32, 32]
    float *d_conv1_out, *d_pool1_out;
    float *d_conv2_out, *d_pool2_out;
    float *d_fc1_out, *d_fc2_out, *d_output;
    
    // Weights
    float *d_conv1_w, *d_conv1_b;  // [6, 3, 5, 5]
    float *d_conv2_w, *d_conv2_b;  // [16, 6, 5, 5]
    float *d_fc1_w, *d_fc1_b;      // [120, 400]
    float *d_fc2_w, *d_fc2_b;      // [84, 120]
    float *d_fc3_w, *d_fc3_b;      // [10, 84]
    
    // Gradients (for backprop)
    float *d_grad_output, *d_grad_fc2, *d_grad_fc1;
    float *d_grad_pool2, *d_grad_conv2;
    float *d_grad_pool1, *d_grad_conv1;
    
    // Max pooling indices
    int *d_pool1_indices, *d_pool2_indices;
} LeNet5;

void init_lenet5(LeNet5* model) {
    // Allocate all memory
    CHECK_CUDA(cudaMalloc(&model->d_input, INPUT_CHANNELS * INPUT_SIZE * INPUT_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->d_conv1_out, C1_FILTERS * C1_SIZE * C1_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->d_pool1_out, C1_FILTERS * S2_SIZE * S2_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->d_conv2_out, C3_FILTERS * C3_SIZE * C3_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->d_pool2_out, C3_FILTERS * S4_SIZE * S4_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->d_fc1_out, C5_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->d_fc2_out, F6_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->d_output, OUTPUT_SIZE * sizeof(float)));
    
    // Weights - Note: Conv1 now has 3 input channels for RGB
    CHECK_CUDA(cudaMalloc(&model->d_conv1_w, C1_FILTERS * INPUT_CHANNELS * CONV_KERNEL * CONV_KERNEL * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->d_conv1_b, C1_FILTERS * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->d_conv2_w, C3_FILTERS * C1_FILTERS * CONV_KERNEL * CONV_KERNEL * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->d_conv2_b, C3_FILTERS * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->d_fc1_w, C5_SIZE * (C3_FILTERS * S4_SIZE * S4_SIZE) * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->d_fc1_b, C5_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->d_fc2_w, F6_SIZE * C5_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->d_fc2_b, F6_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->d_fc3_w, OUTPUT_SIZE * F6_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->d_fc3_b, OUTPUT_SIZE * sizeof(float)));
    
    // Gradients
    CHECK_CUDA(cudaMalloc(&model->d_grad_output, OUTPUT_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->d_grad_fc2, F6_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->d_grad_fc1, C5_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->d_grad_pool2, C3_FILTERS * S4_SIZE * S4_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->d_grad_conv2, C3_FILTERS * C3_SIZE * C3_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->d_grad_pool1, C1_FILTERS * S2_SIZE * S2_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->d_grad_conv1, C1_FILTERS * C1_SIZE * C1_SIZE * sizeof(float)));
    
    // Pooling indices
    CHECK_CUDA(cudaMalloc(&model->d_pool1_indices, C1_FILTERS * S2_SIZE * S2_SIZE * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&model->d_pool2_indices, C3_FILTERS * S4_SIZE * S4_SIZE * sizeof(int)));
    
    // Initialize weights with Xavier initialization
    srand(time(NULL));
    auto init_weights = [](float* d_ptr, int size, float scale) {
        float* h_w = (float*)malloc(size * sizeof(float));
        for (int i = 0; i < size; i++) {
            h_w[i] = (((float)rand() / RAND_MAX) - 0.5f) * 2.0f * scale;
        }
        CHECK_CUDA(cudaMemcpy(d_ptr, h_w, size * sizeof(float), cudaMemcpyHostToDevice));
        free(h_w);
    };
    
    // Xavier init for RGB input
    init_weights(model->d_conv1_w, C1_FILTERS * INPUT_CHANNELS * CONV_KERNEL * CONV_KERNEL, 
                 sqrtf(2.0f / (INPUT_CHANNELS * CONV_KERNEL * CONV_KERNEL)));
    init_weights(model->d_conv2_w, C3_FILTERS * C1_FILTERS * CONV_KERNEL * CONV_KERNEL, 
                 sqrtf(2.0f / (C1_FILTERS * CONV_KERNEL * CONV_KERNEL)));
    init_weights(model->d_fc1_w, C5_SIZE * C3_FILTERS * S4_SIZE * S4_SIZE, 
                 sqrtf(2.0f / (C3_FILTERS * S4_SIZE * S4_SIZE)));
    init_weights(model->d_fc2_w, F6_SIZE * C5_SIZE, sqrtf(2.0f / C5_SIZE));
    init_weights(model->d_fc3_w, OUTPUT_SIZE * F6_SIZE, sqrtf(2.0f / F6_SIZE));
    
    CHECK_CUDA(cudaMemset(model->d_conv1_b, 0, C1_FILTERS * sizeof(float)));
    CHECK_CUDA(cudaMemset(model->d_conv2_b, 0, C3_FILTERS * sizeof(float)));
    CHECK_CUDA(cudaMemset(model->d_fc1_b, 0, C5_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMemset(model->d_fc2_b, 0, F6_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMemset(model->d_fc3_b, 0, OUTPUT_SIZE * sizeof(float)));
}

void forward_pass(LeNet5* model) {
    // Conv1: 32x32x3 -> 28x28x6
    dim3 grid1(C1_FILTERS, C1_SIZE, 1);
    conv2d_forward<<<grid1, C1_SIZE>>>(
        model->d_input, model->d_conv1_w, model->d_conv1_b,
        model->d_conv1_out, INPUT_CHANNELS, C1_FILTERS, INPUT_SIZE, C1_SIZE, CONV_KERNEL
    );
    relu_forward<<<(C1_FILTERS * C1_SIZE * C1_SIZE + 255) / 256, 256>>>(
        model->d_conv1_out, C1_FILTERS * C1_SIZE * C1_SIZE
    );
    
    // Pool1: 28x28x6 -> 14x14x6
    dim3 grid2(C1_FILTERS, S2_SIZE, 1);
    maxpool2d_forward<<<grid2, S2_SIZE>>>(
        model->d_conv1_out, model->d_pool1_out, model->d_pool1_indices,
        C1_FILTERS, C1_SIZE, S2_SIZE
    );
    
    // Conv2: 14x14x6 -> 10x10x16
    dim3 grid3(C3_FILTERS, C3_SIZE, 1);
    conv2d_forward<<<grid3, C3_SIZE>>>(
        model->d_pool1_out, model->d_conv2_w, model->d_conv2_b,
        model->d_conv2_out, C1_FILTERS, C3_FILTERS, S2_SIZE, C3_SIZE, CONV_KERNEL
    );
    relu_forward<<<(C3_FILTERS * C3_SIZE * C3_SIZE + 255) / 256, 256>>>(
        model->d_conv2_out, C3_FILTERS * C3_SIZE * C3_SIZE
    );
    
    // Pool2: 10x10x16 -> 5x5x16
    dim3 grid4(C3_FILTERS, S4_SIZE, 1);
    maxpool2d_forward<<<grid4, S4_SIZE>>>(
        model->d_conv2_out, model->d_pool2_out, model->d_pool2_indices,
        C3_FILTERS, C3_SIZE, S4_SIZE
    );
    
    // FC1: 400 -> 120
    fc_forward<<<(C5_SIZE + 255) / 256, 256>>>(
        model->d_pool2_out, model->d_fc1_w, model->d_fc1_b,
        model->d_fc1_out, C3_FILTERS * S4_SIZE * S4_SIZE, C5_SIZE
    );
    relu_forward<<<(C5_SIZE + 255) / 256, 256>>>(model->d_fc1_out, C5_SIZE);
    
    // FC2: 120 -> 84
    fc_forward<<<(F6_SIZE + 255) / 256, 256>>>(
        model->d_fc1_out, model->d_fc2_w, model->d_fc2_b,
        model->d_fc2_out, C5_SIZE, F6_SIZE
    );
    relu_forward<<<(F6_SIZE + 255) / 256, 256>>>(model->d_fc2_out, F6_SIZE);
    
    // FC3: 84 -> 10
    fc_forward<<<(OUTPUT_SIZE + 255) / 256, 256>>>(
        model->d_fc2_out, model->d_fc3_w, model->d_fc3_b,
        model->d_output, F6_SIZE, OUTPUT_SIZE
    );
    
    // Softmax
    softmax_forward<<<1, OUTPUT_SIZE>>>(model->d_output, OUTPUT_SIZE);
    
    CHECK_CUDA(cudaGetLastError());
}

void backward_pass(LeNet5* model, int label, float lr) {
    // Compute gradient at output (cross-entropy loss derivative)
    float h_output[OUTPUT_SIZE];
    CHECK_CUDA(cudaMemcpy(h_output, model->d_output, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        h_output[i] = (i == label) ? h_output[i] - 1.0f : h_output[i];
    }
    CHECK_CUDA(cudaMemcpy(model->d_grad_output, h_output, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    
    // ===== Backprop FC3 (84 -> 10) =====
    fc_backward_weight<<<OUTPUT_SIZE, F6_SIZE>>>(
        model->d_fc2_out, model->d_grad_output, model->d_fc3_w, F6_SIZE, OUTPUT_SIZE, lr
    );
    fc_backward_bias<<<1, OUTPUT_SIZE>>>(model->d_grad_output, model->d_fc3_b, OUTPUT_SIZE, lr);
    fc_backward_input<<<(F6_SIZE + 255) / 256, 256>>>(
        model->d_grad_output, model->d_fc3_w, model->d_grad_fc2, F6_SIZE, OUTPUT_SIZE
    );
    
    // Backprop through ReLU
    relu_backward<<<(F6_SIZE + 255) / 256, 256>>>(model->d_grad_fc2, model->d_fc2_out, F6_SIZE);
    
    // ===== Backprop FC2 (120 -> 84) =====
    fc_backward_weight<<<F6_SIZE, C5_SIZE>>>(
        model->d_fc1_out, model->d_grad_fc2, model->d_fc2_w, C5_SIZE, F6_SIZE, lr
    );
    fc_backward_bias<<<1, F6_SIZE>>>(model->d_grad_fc2, model->d_fc2_b, F6_SIZE, lr);
    fc_backward_input<<<(C5_SIZE + 255) / 256, 256>>>(
        model->d_grad_fc2, model->d_fc2_w, model->d_grad_fc1, C5_SIZE, F6_SIZE
    );
    
    // Backprop through ReLU
    relu_backward<<<(C5_SIZE + 255) / 256, 256>>>(model->d_grad_fc1, model->d_fc1_out, C5_SIZE);
    
    // ===== Backprop FC1 (400 -> 120) =====
    fc_backward_weight<<<C5_SIZE, C3_FILTERS * S4_SIZE * S4_SIZE>>>(
        model->d_pool2_out, model->d_grad_fc1, model->d_fc1_w,
        C3_FILTERS * S4_SIZE * S4_SIZE, C5_SIZE, lr
    );
    fc_backward_bias<<<1, C5_SIZE>>>(model->d_grad_fc1, model->d_fc1_b, C5_SIZE, lr);
    fc_backward_input<<<(C3_FILTERS * S4_SIZE * S4_SIZE + 255) / 256, 256>>>(
        model->d_grad_fc1, model->d_fc1_w, model->d_grad_pool2,
        C3_FILTERS * S4_SIZE * S4_SIZE, C5_SIZE
    );
    
    // ===== Backprop Pool2 (10x10x16 -> 5x5x16) =====
    CHECK_CUDA(cudaMemset(model->d_grad_conv2, 0, C3_FILTERS * C3_SIZE * C3_SIZE * sizeof(float)));
    dim3 pool2_grid(C3_FILTERS, S4_SIZE, 1);
    maxpool2d_backward<<<pool2_grid, S4_SIZE>>>(
        model->d_grad_pool2, model->d_pool2_indices, model->d_grad_conv2,
        C3_FILTERS, C3_SIZE, S4_SIZE
    );
    
    // Backprop through ReLU
    relu_backward<<<(C3_FILTERS * C3_SIZE * C3_SIZE + 255) / 256, 256>>>(
        model->d_grad_conv2, model->d_conv2_out, C3_FILTERS * C3_SIZE * C3_SIZE
    );
    
    // ===== Backprop Conv2 (14x14x6 -> 10x10x16) =====
    dim3 conv2_w_grid(C3_FILTERS, C1_FILTERS, CONV_KERNEL);
    conv2d_backward_weight<<<conv2_w_grid, CONV_KERNEL>>>(
        model->d_pool1_out, model->d_grad_conv2, model->d_conv2_w,
        C1_FILTERS, C3_FILTERS, S2_SIZE, C3_SIZE, CONV_KERNEL, lr
    );
    conv2d_backward_bias<<<1, C3_FILTERS>>>(
        model->d_grad_conv2, model->d_conv2_b, C3_FILTERS, C3_SIZE, lr
    );
    dim3 conv2_in_grid(C1_FILTERS, S2_SIZE, 1);
    conv2d_backward_input<<<conv2_in_grid, S2_SIZE>>>(
        model->d_grad_conv2, model->d_conv2_w, model->d_grad_pool1,
        C1_FILTERS, C3_FILTERS, S2_SIZE, C3_SIZE, CONV_KERNEL
    );
    
    // ===== Backprop Pool1 (28x28x6 -> 14x14x6) =====
    CHECK_CUDA(cudaMemset(model->d_grad_conv1, 0, C1_FILTERS * C1_SIZE * C1_SIZE * sizeof(float)));
    dim3 pool1_grid(C1_FILTERS, S2_SIZE, 1);
    maxpool2d_backward<<<pool1_grid, S2_SIZE>>>(
        model->d_grad_pool1, model->d_pool1_indices, model->d_grad_conv1,
        C1_FILTERS, C1_SIZE, S2_SIZE
    );
    
    // Backprop through ReLU
    relu_backward<<<(C1_FILTERS * C1_SIZE * C1_SIZE + 255) / 256, 256>>>(
        model->d_grad_conv1, model->d_conv1_out, C1_FILTERS * C1_SIZE * C1_SIZE
    );
    
    // ===== Backprop Conv1 (32x32x3 -> 28x28x6) =====
    dim3 conv1_w_grid(C1_FILTERS, INPUT_CHANNELS, CONV_KERNEL);
    conv2d_backward_weight<<<conv1_w_grid, CONV_KERNEL>>>(
        model->d_input, model->d_grad_conv1, model->d_conv1_w,
        INPUT_CHANNELS, C1_FILTERS, INPUT_SIZE, C1_SIZE, CONV_KERNEL, lr
    );
    conv2d_backward_bias<<<1, C1_FILTERS>>>(
        model->d_grad_conv1, model->d_conv1_b, C1_FILTERS, C1_SIZE, lr
    );
    
    CHECK_CUDA(cudaGetLastError());
}

void free_lenet5(LeNet5* model) {
    cudaFree(model->d_input);
    cudaFree(model->d_conv1_out); cudaFree(model->d_pool1_out);
    cudaFree(model->d_conv2_out); cudaFree(model->d_pool2_out);
    cudaFree(model->d_fc1_out); cudaFree(model->d_fc2_out); cudaFree(model->d_output);
    cudaFree(model->d_conv1_w); cudaFree(model->d_conv1_b);
    cudaFree(model->d_conv2_w); cudaFree(model->d_conv2_b);
    cudaFree(model->d_fc1_w); cudaFree(model->d_fc1_b);
    cudaFree(model->d_fc2_w); cudaFree(model->d_fc2_b);
    cudaFree(model->d_fc3_w); cudaFree(model->d_fc3_b);
    cudaFree(model->d_grad_output); cudaFree(model->d_grad_fc2); cudaFree(model->d_grad_fc1);
    cudaFree(model->d_grad_pool2); cudaFree(model->d_grad_conv2);
    cudaFree(model->d_grad_pool1); cudaFree(model->d_grad_conv1);
    cudaFree(model->d_pool1_indices); cudaFree(model->d_pool2_indices);
}

// ==================== Training and Testing ====================

void train_epoch(LeNet5* model, float* images, unsigned char* labels, int num_samples, float lr) {
    for (int i = 0; i < num_samples; i++) {
        // Copy image to device
        CHECK_CUDA(cudaMemcpy(model->d_input, &images[i * 3 * INPUT_SIZE * INPUT_SIZE],
                             3 * INPUT_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
        
        // Forward pass
        forward_pass(model);
        
        // Backward pass
        backward_pass(model, labels[i], lr);
        
        if ((i + 1) % 1000 == 0) {
            printf("  Processed %d/%d samples\r", i + 1, num_samples);
            fflush(stdout);
        }
    }
    printf("\n");
}

float test_accuracy(LeNet5* model, float* images, unsigned char* labels, int num_samples) {
    int correct = 0;
    float h_output[OUTPUT_SIZE];
    
    for (int i = 0; i < num_samples; i++) {
        CHECK_CUDA(cudaMemcpy(model->d_input, &images[i * 3 * INPUT_SIZE * INPUT_SIZE],
                             3 * INPUT_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
        
        forward_pass(model);
        
        CHECK_CUDA(cudaMemcpy(h_output, model->d_output, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
        
        int predicted = 0;
        float max_prob = h_output[0];
        for (int j = 1; j < OUTPUT_SIZE; j++) {
            if (h_output[j] > max_prob) {
                max_prob = h_output[j];
                predicted = j;
            }
        }
        
        if (predicted == labels[i]) correct++;
    }
    
    return 100.0f * correct / num_samples;
}

// ==================== Main ====================

int main(int argc, char** argv) {
    printf("LeNet-5 CIFAR-10 Training (CUDA Implementation)\n");
    printf("================================================\n\n");
    
    // Load CIFAR-10 data
    printf("Loading CIFAR-10 dataset...\n");
    const char* data_dir = (argc > 1) ? argv[1] : "./cifar-10-batches-bin";
    
    CIFAR10Data train_data = load_cifar10_train(data_dir);
    CIFAR10Data test_data = load_cifar10_test(data_dir);
    
    printf("Loaded %d training samples and %d test samples\n\n", train_data.count, test_data.count);
    
    // Initialize model
    printf("Initializing LeNet-5 model...\n");
    LeNet5 model;
    init_lenet5(&model);
    printf("Model initialized\n\n");
    
    // Training loop
    printf("Starting training...\n");
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        printf("Epoch %d/%d\n", epoch + 1, EPOCHS);
        
        // Train
        train_epoch(&model, train_data.images, train_data.labels, train_data.count, LEARNING_RATE);
        
        // Test
        float train_acc = test_accuracy(&model, train_data.images, train_data.labels, 1000); // Sample
        float test_acc = test_accuracy(&model, test_data.images, test_data.labels, test_data.count);
        
        printf("  Train Accuracy: %.2f%% | Test Accuracy: %.2f%%\n\n", train_acc, test_acc);
    }
    
    printf("Training complete!\n");
    printf("Final Test Accuracy: %.2f%%\n", test_accuracy(&model, test_data.images, test_data.labels, test_data.count));
    
    // Cleanup
    free(train_data.images);
    free(train_data.labels);
    free(test_data.images);
    free(test_data.labels);
    free_lenet5(&model);
    
    return 0;
}
