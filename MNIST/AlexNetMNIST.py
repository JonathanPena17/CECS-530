import argparse, random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import alexnet

MNIST_MEAN, MNIST_STD = (0.1307,), (0.3081,)

def make_mnist_loaders(batch_size=128, num_workers=2):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD),
    ])
    train = datasets.MNIST('./data', train=True, download=True, transform=tfm)
    test  = datasets.MNIST('./data', train=False, download=True, transform=tfm)
    return (DataLoader(train, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True),
            DataLoader(test,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True))

def mnist_alexnet(num_classes=10):
    m = alexnet(pretrained=False)
    # 1-channel stem suitable for 28x28: 3x3, stride=1, padding=1
    m.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
    m.features[2] = nn.Identity()  # drop early maxpool
    # Keep AdaptiveAvgPool2d((6,6)) so classifier stays consistent
    m.classifier[6] = nn.Linear(m.classifier[6].in_features, num_classes)
    return m

def accuracy(model, loader, device):
    model.eval(); correct=total=0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred==y).sum().item()
            total += y.size(0)
    return 100.0*correct/total

def train_epoch(model, loader, device, optimizer, criterion):
    model.train(); loss_sum=0.0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward(); optimizer.step()
        loss_sum += loss.item()*x.size(0)
    return loss_sum/len(loader.dataset)

def count_params(m): return sum(p.numel() for p in m.parameters())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--momentum', type=float, default=0.9)
    ap.add_argument('--weight_decay', type=float, default=5e-4)
    ap.add_argument('--step_size', type=int, default=10)
    ap.add_argument('--gamma', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--ckpt', type=str, default='alexnet_mnist_best.pth')
    args = ap.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, test_loader = make_mnist_loaders(args.batch_size)

    model = mnist_alexnet().to(device)
    print(f'Parameters: {count_params(model):,}')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    best=0.0
    for epoch in range(1, args.epochs+1):
        loss = train_epoch(model, train_loader, device, optimizer, criterion)
        scheduler.step()
        acc = accuracy(model, test_loader, device)
        print(f'Epoch {epoch:02d}/{args.epochs} | Loss: {loss:.4f} | Test Acc: {acc:.2f}% | LR: {scheduler.get_last_lr()[0]:.5f}')
        if acc>best:
            best=acc
            torch.save({'model': model.state_dict(), 'acc': best, 'epoch': epoch}, args.ckpt)
            print(f'  -> Saved new best to {args.ckpt} (acc={best:.2f}%)')
    print(f'Best Test Accuracy: {best:.2f}%')

if __name__=='__main__':
    main()
