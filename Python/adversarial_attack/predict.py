# coding:utf-8
"""
  Purpose:  Load network weights and predict
"""

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn.functional import softmax

# from models.model import CNN as Model
# from models.model import ScattCNN as Model
from models.model import ScattDense as Model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # weights_name = "CNN_loss_0.0444.pth"
    # weights_name = "ScattCNN_loss_0.0342.pth"
    weights_name = "ScattDense_loss_0.0232.pth"

    img_name = "000000-num7_adversarial_ScattDense.png"
    base_img_name = img_name[:11] + ".png"

    # Open image
    img = cv2.imread(os.path.join("images", img_name), cv2.IMREAD_GRAYSCALE)
    base_img = cv2.imread(os.path.join("images", base_img_name), cv2.IMREAD_GRAYSCALE)
    noise = img.astype(np.float32) - base_img.astype(np.float32)
    noise = np.abs(noise)

    # Load model
    model = Model().to(device)
    model.load_state_dict(torch.load(os.path.join("saved_models", weights_name)))
    model.eval()

    # Predict
    img = img.astype(np.float32) / 255.
    img_tensor = img[np.newaxis, np.newaxis, :, :]
    img_tensor = (img_tensor - 0.1307) / 0.3081
    img_tensor = torch.from_numpy(img_tensor).to(device)
    y_pred = model(img_tensor)
    _, indices = torch.max(y_pred, 1)
    number = indices.cpu().data.numpy()[0]
    y_softmax = softmax(y_pred, dim=1).cpu().data.numpy()[0]
    prob = y_softmax[number] * 100
    print("Predicted: {} with probability {:.2f}%".format(number, prob))

    # Plot
    plt.figure(1, figsize=(10, 3))
    plt.subplot(131)
    plt.imshow(img, cmap='gray')
    plt.title("Adversarial example")
    plt.subplot(132)
    plt.imshow(noise, cmap='gray')
    plt.title("Noise")
    plt.subplot(133)
    plt.bar(np.arange(0, 10), y_softmax * 100, tick_label=np.arange(0, 10))
    plt.ylim(0, 100)
    plt.title("Predictions")
    plt.show()


if __name__ == '__main__':
    main()
