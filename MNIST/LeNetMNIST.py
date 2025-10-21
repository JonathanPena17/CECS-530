"""
  Train for 20 epochs with data augmentation:
    python MNIST/LeNetMNIST.py --epochs 20 --augment

  Evaluate a saved checkpoint:
    python MNIST/LeNetMNIST.py --mode eval --ckpt MNIST/LeNetMNIST_best.pth
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from datetime import datetime
import os
import argparse

# ---------------------- Model ----------------------
class LeNetMNIST(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNetMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

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

# ---------------------- Training ----------------------
def train(model, loader, device, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

    avg_loss = total_loss / total
    acc = 100.0 * correct / total
    return avg_loss, acc

# ---------------------- Evaluation ----------------------
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    avg_loss = total_loss / total
    acc = 100.0 * correct / total
    return avg_loss, acc

# ---------------------- Main ----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"])
    parser.add_argument("--ckpt", type=str, default="MNIST/LeNetMNIST_best.pth", help="Checkpoint file path")
    args = parser.parse_args()

    # Data transforms for MNIST
    transform_mnist = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load MNIST dataset
    train_dataset_mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
    test_dataset_mnist = datasets.MNIST(root='./data', train=False, transform=transform_mnist)

    train_loader_mnist = DataLoader(train_dataset_mnist, batch_size=args.batch_size, shuffle=True)
    test_loader_mnist = DataLoader(test_dataset_mnist, batch_size=args.batch_size, shuffle=False)

    # Device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Model, loss, optimizer
    model_mnist = LeNetMNIST().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_mnist.parameters(), lr=args.lr)

    # Evaluation mode: load and test
    if args.mode == "eval":
        if not os.path.exists(args.ckpt):
            raise FileNotFoundError(f"Checkpoint '{args.ckpt}' not found.")
        checkpoint = torch.load(args.ckpt, map_location=device)
        model_mnist.load_state_dict(checkpoint['model'])
        test_loss, test_acc = evaluate(model_mnist, test_loader_mnist, device, criterion)
        print(f"[EVAL] Loaded '{args.ckpt}' | Test Accuracy: {test_acc:.2f}% | Loss: {test_loss:.4f}")
        exit()

    # Training mode
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train(model_mnist, train_loader_mnist, device, optimizer, criterion)
        test_loss, test_acc = evaluate(model_mnist, test_loader_mnist, device, criterion)

        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({'model': model_mnist.state_dict(),
                        'acc': best_acc,
                        'epoch': epoch}, args.ckpt)
            print(f"  -> Saved new best model ({best_acc:.2f}%) to {args.ckpt}")

    print(f"Best Test Accuracy: {best_acc:.2f}%")