# coding:utf-8
"""
  Purpose:  Translation invariance test
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from kymatio import Scattering2D


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shape = (64,) * 2
    img_1 = np.zeros(shape, dtype=np.float32)
    # img_1[10:20, 15:20] = 1.
    img_1[15:20, 10:20] = 1.

    plt.figure(1, figsize=(12, 2))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.6)

    for i, J in enumerate(range(2, 6)):
        scattering = Scattering2D(J=J, shape=shape, max_order=1).cuda()
        img_1_scatt = scattering(torch.from_numpy(img_1).to(device)).cpu().numpy()

        means = list()
        for shift in range(shape[0] - 5):
            img = np.zeros(shape, dtype=np.float32)
            img[10:20, shift:shift + 5] = 1.
            img_scatt = scattering(torch.from_numpy(img).to(device)).cpu().numpy()
            means.append(np.mean((img_1_scatt - img_scatt) ** 2))

        num = 152 + i
        plt.subplot(num)
        plt.plot(means)
        plt.ylim([0, 0.0012])
        plt.title("J = {}".format(J))

    plt.subplot(151)
    plt.imshow(img_1, cmap='gray')
    plt.title("Image référence")
    plt.show()


if __name__ == '__main__':
    main()
