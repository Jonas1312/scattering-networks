# coding:utf-8
"""
  Purpose:  Generate adversarial attack
"""

import os

os.environ["KYMATIO_BACKEND"] = "torch"  # scattering transform has to be differentiable
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable

from models.model import ScattCNN as Model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # weights_name = "CNN_loss_0.0444.pth"
    weights_name = "ScattCNN_loss_0.0342.pth"
    # weights_name = "ScattDense_loss_0.0232.pth"
    img_name = "005706-num5.png"
    target_class = 6
    assert (0 <= target_class < 10)
    weight = 0.9913

    # Open image
    img = cv2.imread(os.path.join("images", img_name), cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32) / 255.

    # Load model
    model = Model().to(device)
    model.load_state_dict(torch.load(os.path.join("saved_models", weights_name)))
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # Create IO
    img_tensor = img[np.newaxis, np.newaxis, :, :]
    img_tensor = (img_tensor - 0.1307) / 0.3081
    img_tensor = torch.from_numpy(img_tensor)
    img_tensor = img_tensor.to(device)
    noise_tensor = Variable(img_tensor.data.clone(), requires_grad=True).to(device)
    target_class = torch.from_numpy(np.array([target_class])).to(device)
    target_class = target_class.long()

    # Train noise tensor
    optimizer = optim.Adam([noise_tensor], lr=1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=800, verbose=True, cooldown=200)

    best_noise = None
    best_loss = float("inf")
    for iteration in range(1, 3500):
        optimizer.zero_grad()
        y_pred = model(noise_tensor)
        loss_noise = F.mse_loss(input=noise_tensor, target=img_tensor) * (1 - weight)
        loss_label = F.cross_entropy(input=y_pred, target=target_class) * weight
        loss = loss_noise + loss_label
        scheduler.step(loss)
        print("iter {}, loss_noise {:.5f}, loss_label {:.9f}".format(iteration, loss_noise.item(), loss_label.item()),
              end='\r')
        loss.backward()
        optimizer.step()
        if loss < best_loss:
            best_loss = loss
            best_noise = noise_tensor.clone().detach()
    print("")

    # Save noise image
    print("Save adversarial image")
    noise_img = best_noise.cpu().detach().numpy()
    noise_img = noise_img[0, 0, :, :]
    noise_img = (noise_img * 0.3081 + 0.1307) * 255
    noise_img = np.clip(noise_img, 0, 255)
    noise_img = noise_img.astype(np.uint8)
    cv2.imwrite(os.path.join("images", img_name[:-4] + "_adversarial_" + Model.__name__ + img_name[-4:]), noise_img)


if __name__ == '__main__':
    main()
