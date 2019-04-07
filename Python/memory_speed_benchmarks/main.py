# coding:utf-8
"""
  Purpose:  Benchmarks Scattering based Hybrid models vs CNN in terms of memory and speed
"""

import time
import torch.optim as optim

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from models.base_model import Base


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    iterations = 50

    test_loader = torch.utils.data.DataLoader(
        datasets.STL10('../data/STL10', split='train',
                       transform=transforms.Compose([
                           transforms.Resize((256, 256)),
                           transforms.ToTensor(),  # rescale between 0. and 1.
                       ])),
        batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model = Base().to(device)

    times = []
    optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=0.0005)
    for _ in range(1, iterations + 1):
        # eval mode
        model.eval()
        with torch.no_grad():
            data, _ = next(iter(test_loader))
            data = data.to(device)
            start = time.perf_counter()
            model(data)
            torch.cuda.synchronize()
            end = time.perf_counter()

        # train mode with backprop
        # model.train()
        # data, _ = next(iter(test_loader))
        # data = data.to(device)
        # optimizer.zero_grad()
        # start = time.perf_counter()
        # output = model(data)
        # target = torch.randn_like(output)
        # loss = F.mse_loss(output, target)
        # loss.backward()
        # optimizer.step()
        # torch.cuda.synchronize()
        # end = time.perf_counter()

        times.append(end - start)
        print("Took {}s".format(end - start))

    print("Median {}s".format(sorted(times)[len(times) // 2]))


if __name__ == '__main__':
    main()
