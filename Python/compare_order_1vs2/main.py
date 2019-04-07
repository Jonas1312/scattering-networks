# coding:utf-8
"""
  Purpose:  Compare order 1 vs order 2 scattering transform performances
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from models.resnet34 import ResNet34


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
    epochs = 350
    delta_epoch = 20

    train_loader = torch.utils.data.DataLoader(
        datasets.STL10('../data/STL10', split='train', download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),  # rescale between 0. and 1.
                       ])),
        batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.STL10('../data/STL10', split='test',
                       transform=transforms.Compose([
                           transforms.ToTensor(),  # rescale between 0. and 1.
                       ])),
        batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    for order in [1, 2]:
        model = ResNet34(order=order).to(device)

        optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=0.0005)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.90)

        best_accuracy = 0.
        best_epoch = None
        for epoch in range(1, epochs + 1):
            print("############### Current epoch: {} ###############".format(epoch))
            scheduler.step()
            train(model, device, train_loader, optimizer, epoch)
            test_loss, accuracy = validate(model, device, test_loader)
            if best_accuracy < accuracy:
                best_accuracy = accuracy
                best_epoch = epoch
            else:
                if (epoch - best_epoch) > delta_epoch * 1.5:
                    print("Early stopping!")
                    break

            print("Order {} best accuracy: {}%".format(order, best_accuracy))


if __name__ == '__main__':
    main()
