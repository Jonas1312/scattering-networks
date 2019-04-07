# coding:utf-8
"""
  Purpose:  Train network and save weights
"""

import os

import torch
import torch.nn.functional as F
from torch import optim
from torchvision import datasets, transforms

from models.resnet18 import ResNet, ScattResNet


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    nb_samples = 0
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        nb_samples += len(data)
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = F.cross_entropy(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('Train Epoch: {} [{}/{} ({:.0f}%)], Loss: {:.6f}'.format(
            epoch,
            nb_samples,
            len(train_loader.dataset),
            100. * (batch_idx + 1) / len(train_loader),
            loss.item()), end='\r')
        train_loss += F.cross_entropy(output, target, reduction='sum').item()

    train_loss /= len(train_loader.dataset)
    print("Train Epoch: {} [{}/{} ({:.0f}%)], Average Loss: {:.6f}".format(
        epoch, nb_samples, len(train_loader.dataset), 100., train_loss))
    return train_loss


def validate(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print('Test set: Average loss: {:.6f}, Correct: {}/{}, Accuracy: ({:.2f}%)'.format(
        test_loss,
        correct,
        len(test_loader.dataset),
        accuracy))
    return test_loss, accuracy


def save_model(model):
    file_name = "{}.pth".format(model.__class__.__name__)
    path = os.path.join("saved_models", file_name)
    if not os.path.isfile(path):
        torch.save(model.state_dict(), path)
        print("Saved: ", file_name)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    batch_size = 64
    epochs = 40
    input_size = (32, ) * 2

    train_loader = torch.utils.data.DataLoader(
        datasets.STL10('../data/STL10', split='train', download=True,
                       transform=transforms.Compose([
                           transforms.RandomHorizontalFlip(),
                           transforms.RandomAffine(degrees=5, translate=(0.03, 0.03), scale=None),
                           transforms.CenterCrop((76, 76)),
                           transforms.Resize(input_size),
                           transforms.ToTensor(),  # rescale between 0. and 1.
                       ])),
        batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.STL10('../data/STL10', split='test',
                       transform=transforms.Compose([
                           transforms.CenterCrop((76, 76)),
                           transforms.Resize(input_size),
                           transforms.ToTensor(),  # rescale between 0. and 1.
                       ])),
        batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model = ResNet().to(device)

    optimizer = optim.SGD(model.parameters(), lr=1e-1, momentum=0.9, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 25, 30])

    for epoch in range(1, epochs + 1):
        print("######################### EPOCH {}/{} #########################".format(epoch, epochs))

        for param_group in optimizer.param_groups:
            print("Current learning rate:", param_group['lr'])

        scheduler.step()

        train_loss = train(model, device, train_loader, optimizer, epoch)
        test_loss, test_accuracy = validate(model, device, test_loader)

    save_model(model)


if __name__ == '__main__':
    main()
