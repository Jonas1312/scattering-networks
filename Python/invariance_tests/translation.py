# coding:utf-8
"""
  Purpose:  Test translation invariance
"""

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms

# from models.resnet18 import ResNet as Model
# from models.resnet18 import ScattResNet as Model
from models.resnet18 import ScattResNetNoPool as Model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_name = "test_image_png_6000.png"
    class_index = 0
    labels = {
        0: "airplane",
        1: "bird",
        2: "car",
        3: "cat",
        4: "deer",
        5: "dog",
        6: "horse",
        7: "monkey",
        8: "ship",
        9: "truck",
    }

    # Open image
    src_img = cv2.imread(os.path.join("images", img_name), cv2.IMREAD_COLOR)
    src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)

    # Load model
    model = Model().to(device)
    weights_name = Model.__name__ + ".pth"
    model.load_state_dict(torch.load(os.path.join("saved_models", weights_name)))
    model.eval()

    accuracy_list = list()
    inter_size = 76
    for shift in range(96 - inter_size):
        input_img = src_img[
                    (96 - inter_size) // 2:(96 - inter_size) // 2 + inter_size,
                    shift:inter_size + shift,
                    ]
        assert (input_img.shape[:2] == (inter_size, inter_size))
        input_img = cv2.resize(input_img, (32, 32))
        # plt.imshow(input_img)
        # plt.show()
        input_tensor = transforms.ToTensor()(input_img)
        input_tensor = input_tensor.view(1, 3, 32, 32)
        # print(input_tensor.size())

        # Predict
        y_pred = model(input_tensor.to(device))
        y_pred = torch.softmax(y_pred, dim=-1)
        accuracy = y_pred.cpu().detach().numpy()[0, class_index] * 100
        accuracy_list.append(accuracy)
    print(accuracy_list)

    # Plot
    plt.figure(1, figsize=(12, 3))
    plt.subplots_adjust(left=None, bottom=0.2, right=None, top=None, wspace=None)

    plt.subplot(141)
    plt.imshow(cv2.resize(src_img[(96 - inter_size) // 2:(96 - inter_size) // 2 + inter_size,
                          (96 - inter_size) // 2:(96 - inter_size) // 2 + inter_size, :], (32, 32)))
    plt.axis('off')
    plt.title(r"Center $\bullet$", fontdict={"color": "green", "size": 18})

    plt.subplot(142)
    index_max = accuracy_list.index(max(accuracy_list))
    plt.imshow(cv2.resize(
        src_img[(96 - inter_size) // 2:(96 - inter_size) // 2 + inter_size, index_max:inter_size + index_max, :],
        (32, 32)))
    plt.axis('off')
    plt.title(r"Max $\bullet$", fontdict={"color": "red", "size": 18})

    plt.subplot(143)
    index_min = accuracy_list.index(min(accuracy_list))
    plt.imshow(cv2.resize(
        src_img[(96 - inter_size) // 2:(96 - inter_size) // 2 + inter_size, index_min:inter_size + index_min, :],
        (32, 32)))
    plt.axis('off')
    plt.title(r"Min $\bullet$", fontdict={"color": "blue", "size": 18})

    plt.subplot(144)
    plt.plot(list((range(-(96 - inter_size) // 2, (96 - inter_size) // 2))), np.array(accuracy_list))
    plt.plot(index_max - (96 - inter_size) // 2, accuracy_list[index_max], 'ro')
    plt.plot(index_min - (96 - inter_size) // 2, accuracy_list[index_min], 'bo')
    plt.plot(0, accuracy_list[len(accuracy_list) // 2], 'go')
    plt.ylim([-5, 105])
    plt.ylabel("P({})".format(labels[class_index]))
    plt.xlabel("Translation shift (px)")
    plt.show()


if __name__ == '__main__':
    main()
