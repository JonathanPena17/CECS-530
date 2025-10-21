import argparse, os, random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights

# CIFAR-10 normalization (widely used values)
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2023, 0.1994, 0.2010)


def make_cifar10_loaders(batch_size=128, augment=False, num_workers=2, pin_memory=False):
    normalize = transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
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
    train = datasets.CIFAR10('./data', train=True, download=True, transform=train_tfms)
    test  = datasets.CIFAR10('./data', train=False, download=True, transform=test_tfms)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=pin_memory),
        DataLoader(test,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    )


def cifar_resnet18(num_classes=10):
    # Use the new weights API to avoid deprecation; "weights=None" replaces pretrained=False
    m = resnet18(weights=None)
    # CIFAR-friendly stem: 3x3, stride=1, no maxpool
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    use_amp = (device.type == 'cuda')
    with torch.cuda.amp.autocast(enabled=use_amp):
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    model.train()
    return 100.0 * correct / max(1, total)


def train_epoch(model, loader, device, optimizer, criterion, scaler):
    model.train()
    loss_sum = 0.0
    use_amp = (device.type == 'cuda')
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        loss_sum += loss.item() * x.size(0)
    return loss_sum / len(loader.dataset)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--ckpt', type=str, default='resnet18_cifar10_best.pth')
    parser.add_argument('--warmup_epochs', type=int, default=5)
    args = parser.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = (device.type == 'cuda')

    pin = (device.type == 'cuda')
    train_loader, test_loader = make_cifar10_loaders(args.batch_size, args.augment, pin_memory=pin)

    model = cifar_resnet18(num_classes=10).to(device)
    print(f'Parameters: {count_params(model):,}')

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay, nesterov=True)
    # Cosine schedule over full training; we apply manual warmup below
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    best = 0.0
    for epoch in range(1, args.epochs + 1):
        # Linear warmup for first few epochs
        if epoch <= args.warmup_epochs:
            warmup_factor = 1e-3
            warmup_lr = args.lr * (warmup_factor + (1 - warmup_factor) * epoch / args.warmup_epochs)
            for pg in optimizer.param_groups: pg['lr'] = warmup_lr
        else:
            scheduler.step()

        loss = train_epoch(model, train_loader, device, optimizer, criterion, scaler)
        acc = accuracy(model, test_loader, device)
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch:03d}/{args.epochs} | Loss: {loss:.4f} | Test Acc: {acc:.2f}% | LR: {current_lr:.5f}')
        if acc > best:
            best = acc
            torch.save({'model': model.state_dict(), 'acc': best, 'epoch': epoch}, args.ckpt)
            print(f'  -> Saved new best to {args.ckpt} (acc={best:.2f}%)')

    print(f'Best Test Accuracy: {best:.2f}%')


if __name__ == '__main__':
    main()
