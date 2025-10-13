import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Create a timestamped log file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"training_log_{timestamp}.txt"
log_file = open(log_filename, "w")

def log(msg):
    print(msg)
    log_file.write(msg + "\n")

# Data transforms for MNIST
transform_mnist = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load MNIST dataset
train_dataset_mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
test_dataset_mnist = datasets.MNIST(root='./data', train=False, transform=transform_mnist)

train_loader_mnist = DataLoader(train_dataset_mnist, batch_size=64, shuffle=True)
test_loader_mnist = DataLoader(test_dataset_mnist, batch_size=1000, shuffle=False)

class LeNetMNIST(nn.Module):
    def __init__(self):
        super(LeNetMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Instantiate model, loss, optimizer
model_mnist = LeNetMNIST().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_mnist.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# Training parameters
num_epochs = 10
train_losses = []
val_losses = []
val_accuracies = []

for epoch in range(num_epochs):
    model_mnist.train()
    running_loss = 0.0
    for images, labels in train_loader_mnist:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model_mnist(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader_mnist)
    train_losses.append(avg_train_loss)

    # Validation
    model_mnist.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader_mnist:
            images, labels = images.to(device), labels.to(device)
            outputs = model_mnist(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_val_loss = val_loss / len(test_loader_mnist)
    val_accuracy = 100 * correct / total

    val_losses.append(avg_val_loss)
    val_accuracies.append(val_accuracy)

    # Log results
    log(f"Epoch [{epoch+1}/{num_epochs}]")
    log(f"Train Loss: {avg_train_loss:.4f}")
    log(f"Validation Loss: {avg_val_loss:.4f}")
    log(f"Validation Accuracy: {val_accuracy:.2f}%\n")

    scheduler.step()

# Close log file
log_file.close()

# Plot Loss Curve
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig(f"loss_curve_{timestamp}.png")
plt.show()

# Plot Accuracy Curve
plt.figure(figsize=(10, 5))
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Validation Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
plt.savefig(f"accuracy_curve_{timestamp}.png")
plt.show()

print(f"\nTraining complete. Log saved to: {log_filename}")
