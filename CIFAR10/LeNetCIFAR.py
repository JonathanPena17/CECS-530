"""
  Fault-Injected Evaluation Example for NVBitFI
  -------------------------------------------------
  Train for 30 epochs with data augmentation:
      python CIFAR10/LeNetCIFAR.py --epochs 30 --augment

  Evaluate a saved checkpoint (fault-free and fault-injected modes):
      python CIFAR10/LeNetCIFAR.py --mode eval --ckpt CIFAR10/lenet_cifar10_best.pth
"""

import argparse
import os
import random
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from dataclasses import dataclass


# ------------------- Model Definition -------------------
class LeNetCIFAR10(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ------------------- Dataset -------------------
def get_loaders(batch_size: int = 128, augment: bool = False, num_workers: int = 2):
    normalize = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                     std=(0.247, 0.243, 0.261))
    if augment:
        train_tfms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_tfms = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    test_tfms = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_tfms)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_tfms)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


# ------------------- Configs -------------------
@dataclass
class TrainConfig:
    epochs: int = 20
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 5e-4
    step_size: int = 10
    gamma: float = 0.1
    batch_size: int = 128
    augment: bool = False
    seed: int = 0
    ckpt_path: str = 'lenet_cifar10_best.pth'


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------------------ Evaluation Metrics ------------------
def evaluate_with_fault_logging(model, loader, device, log_dir="nvbitfi_logs"):
    model.eval()
    correct, total = 0, 0
    total_diff, total_mismatch = 0.0, 0.0
    os.makedirs(log_dir, exist_ok=True)

    # Load golden outputs if available
    golden_path = os.path.join(log_dir, "gold_outputs.pt")
    if os.path.exists(golden_path):
        golden_outputs = torch.load(golden_path, map_location=device)
        use_golden = True
    else:
        golden_outputs, use_golden = [], False

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

            # Compare with golden if available
            if use_golden:
                gold_logits = golden_outputs[batch_idx]
                diff = torch.abs(logits - gold_logits).mean().item()
                mismatch = (preds != gold_logits.argmax(dim=1)).float().mean().item()
                total_diff += diff
                total_mismatch += mismatch
            else:
                golden_outputs.append(logits.cpu())

    # Save golden reference if first run (fault-free)
    if not use_golden:
        torch.save(golden_outputs, golden_path)
        print(f"Saved baseline golden outputs at {golden_path}")

    avg_diff = total_diff / len(loader) if use_golden else 0
    avg_mismatch = total_mismatch / len(loader) if use_golden else 0
    acc = 100.0 * correct / total

    log_data = {
        "accuracy": acc,
        "avg_output_diff": avg_diff,
        "avg_label_mismatch": avg_mismatch,
        "fault_injection_run": bool(use_golden),
    }
    with open(os.path.join(log_dir, f"run_log_{os.getpid()}.json"), "w") as f:
        json.dump(log_data, f, indent=2)

    print(f"[EVAL] Accuracy: {acc:.2f}% | Mean Output Δ: {avg_diff:.6f} | Label Mismatch: {avg_mismatch:.4f}")
    return log_data


# ------------------- Training -------------------
def train_epoch(model, loader, device, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
    return running_loss / len(loader.dataset)


def accuracy(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ------------------- Main -------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'])
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--ckpt', type=str, default='CIFAR10/lenet_cifar10_best.pth')
    args = parser.parse_args()

    cfg = TrainConfig(epochs=args.epochs, lr=args.lr, momentum=args.momentum,
                      weight_decay=args.weight_decay, step_size=args.step_size,
                      gamma=args.gamma, batch_size=args.batch_size,
                      augment=args.augment, seed=args.seed, ckpt_path=args.ckpt)

    set_seed(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_loaders(cfg.batch_size, cfg.augment)

    model = LeNetCIFAR10(num_classes=10).to(device)
    print(f'Parameters: {count_params(model):,}')

    if args.mode == 'eval':
        assert os.path.isfile(cfg.ckpt_path), f'Checkpoint not found: {cfg.ckpt_path}'
        ckpt = torch.load(cfg.ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        evaluate_with_fault_logging(model, test_loader, device)
        return

    # Training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)
    best_acc = 0.0

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_epoch(model, train_loader, device, optimizer, criterion)
        scheduler.step()
        test_acc = accuracy(model, test_loader, device)
        print(f'Epoch {epoch:02d}/{cfg.epochs} | Loss: {train_loss:.4f} | Test Acc: {test_acc:.2f}%')

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({'model': model.state_dict(), 'acc': best_acc, 'epoch': epoch}, cfg.ckpt_path)
            print(f'  -> Saved new best model (acc={best_acc:.2f}%)')

    print(f'Best Accuracy: {best_acc:.2f}%')


if __name__ == "__main__":
    main()
