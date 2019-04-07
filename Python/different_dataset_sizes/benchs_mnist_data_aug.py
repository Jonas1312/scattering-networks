# coding:utf-8
"""
  Purpose:  Compare CNN and Scatt + CNN models with different sizes of dataset
"""
import os

os.environ["KYMATIO_BACKEND"] = "skcuda"
import torch
import torch.nn.functional as F
import torch.optim as optim
from models.simple_model import CNN, ScattCNN
from torchvision import datasets, transforms


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
    batch_size = 128
    epochs = 800
    max_images = 60000  # nbr of images available in MNIST
    nb_samples_train = 20  # nbr of images to pick from the training set (<= max_images)
    assert (max_images >= nb_samples_train)
    patience = 300

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(
            datasets.MNIST('../data/mnist', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), shear=10, fillcolor=0),
                               transforms.ToTensor(),  # rescale between 0. and 1.
                               transforms.Normalize((0.1307,), (0.3081,)),  # MNIST
                           ])),
            torch.multinomial(torch.ones(max_images), nb_samples_train)),  # pick subset of training set
        batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data/mnist', train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),  # rescale between 0. and 1.
                           transforms.Normalize((0.1307,), (0.3081,))  # MNIST
                       ])),
        batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    for model_class in [CNN, ScattCNN]:
        model = model_class().to(device)

        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
                                                         patience=patience, verbose=True, cooldown=patience // 2)

        best_accuracy = 0.
        best_epoch = None
        for epoch in range(1, epochs + 1):
            train(model, device, train_loader, optimizer, epoch)
            test_loss, accuracy = validate(model, device, test_loader)
            scheduler.step(test_loss)
            if best_accuracy < accuracy:
                best_accuracy = accuracy
                best_epoch = epoch
            else:
                if (epoch - best_epoch) > patience * 1.5:
                    print("Early stopping!")
                    break

            print("Model {} accuracy {}%, {} samples".format(model.__class__.__name__, best_accuracy, nb_samples_train))


if __name__ == '__main__':
    main()