# coding:utf-8
"""
  Purpose:  Plot benchmarks
"""

import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage.filters import gaussian_filter1d


def main():
    plt.rcParams.update({'font.size': 11})
    file = "benchmarks/benchs_mnist.csv"
    df = pd.read_csv(file)
    sigma=0.5

    x_axis = df['nb_samples'].values

    y_axis = df['accuracy_cnn'].values
    y_axis = gaussian_filter1d(y_axis, sigma=sigma)
    plt.plot(x_axis, y_axis, '-', color='steelblue', label="CNN")

    y_axis = df['accuracy_scatt_cnn'].values
    y_axis = gaussian_filter1d(y_axis, sigma=sigma)
    plt.plot(x_axis, y_axis, '-', color='orange', label="Hybrid")

    # file = "benchmarks/benchs_mnist_data_aug.csv"
    # df = pd.read_csv(file)
    # x_axis = df['nb_samples'].values
    # y_axis = df['accuracy_cnn'].values
    # y_axis = gaussian_filter1d(y_axis, sigma=sigma)
    # plt.plot(x_axis, y_axis, '--', color='steelblue', label="CNN + image augmentation")
    # y_axis = df['accuracy_scatt_cnn'].values
    # y_axis = gaussian_filter1d(y_axis, sigma=sigma)
    # plt.plot(x_axis, y_axis, '--', color='orange', label="Hybrid + image augmentation")

    plt.legend(loc='lower right')
    plt.xticks(x_axis)
    plt.xscale('symlog')
    plt.xlabel("Number of samples in the training set")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.title("MNIST dataset (2xConv2D backbone)")
    plt.show()


if __name__ == '__main__':
    main()
