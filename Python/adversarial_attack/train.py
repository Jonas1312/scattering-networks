# coding:utf-8
"""
  Purpose:  Train and save network weights
"""

import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from models.model import ScattDense as Model


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    nb_samples = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        nb_samples += len(data)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)], Loss: {:.6f}'.format(
            epoch,
            nb_samples,
            len(train_loader.dataset),
            100. * (batch_idx + 1) / len(train_loader),
            loss.item()), end='\r')


def validate(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss,
        correct,
        len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss, 100. * correct / len(test_loader.dataset)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    epochs = 30

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),  # rescale between 0. and 1.
                           transforms.Normalize((0.1307,), (0.3081,))  # MNIST
                       ])),
        batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data/mnist', train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),  # rescale between 0. and 1.
                           transforms.Normalize((0.1307,), (0.3081,))  # MNIST
                       ])),
        batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model = Model().to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    best_loss = float('Inf')
    for epoch in range(1, epochs + 1):
        scheduler.step()
        train(model, device, train_loader, optimizer, epoch)
        test_loss, _ = validate(model, device, test_loader)
        if test_loss < best_loss and epoch > 2:
            best_loss = test_loss
            file_name = "{}_loss_{:.4f}.pth".format(Model.__name__, best_loss)
            torch.save(model.state_dict(), os.path.join("saved_models", file_name))
            print("Saved: ", file_name)


if __name__ == '__main__':
    main()
