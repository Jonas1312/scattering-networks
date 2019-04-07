# coding:utf-8
"""
  Purpose:  Image reconstruction from scattering coefficients. Test on first and second order coefficients.
"""

import numpy as np
import torch
import torch.nn.functional as F
from kymatio import Scattering2D
from torch import optim
import cv2


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    grayscale = False
    img_name = "images/lena.jpg"

    # Load src image
    if not grayscale:
        src_img = cv2.imread(img_name, cv2.IMREAD_COLOR)
    else:
        src_img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
    src_img = np.array(src_img).astype(np.float32)
    y_true = np.array(src_img)
    src_img = src_img / 255.
    if not grayscale:
        src_img = np.moveaxis(src_img, -1, 0)
    print("img shape: ", src_img.shape)
    if grayscale:
        height, width = src_img.shape
    else:
        channels, height, width = src_img.shape

    for order in [2]:
        for J in [3, 4, 5]:
            # nb_coeffs = 1 + J * L + L ** 2 * J * (J - 1) // 2 if max_order == 2 else 1 + J * L
            scattering = Scattering2D(J=J, shape=(height, width), L=8, max_order=order)
            if device == "cuda":
                scattering = scattering.cuda()
            src_img_tensor = torch.from_numpy(src_img).to(device)
            scattering_coefficients = scattering(src_img_tensor)

            # Create random trainable input image
            input_tensor = torch.rand(src_img.shape, requires_grad=True, device=device)

            optimizer = optim.Adam([input_tensor], lr=10)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=100,
                                                             verbose=True, threshold=1e-3, cooldown=50)

            best_img = None
            best_loss = float("inf")
            for epoch in range(1, 1500):
                optimizer.zero_grad()
                new_coefficients = scattering(input_tensor)
                # loss = F.l1_loss(input=new_coefficients, target=scattering_coefficients)
                loss = F.mse_loss(input=new_coefficients, target=scattering_coefficients)
                scheduler.step(loss)
                print("Epoch {}, loss: {}".format(epoch, loss.item()), end='\r')
                loss.backward()
                optimizer.step()
                if loss < best_loss:
                    best_loss = loss
                    best_img = input_tensor.clone().detach()

            y_pred = best_img.cpu().detach().numpy()
            if not grayscale:
                y_pred = np.moveaxis(y_pred, 0, -1)

            y_pred = np.clip(y_pred, 0., 1.)
            y_pred = y_pred * 255.0

            # PSNR
            print("")
            if grayscale:
                mse = np.mean((y_true - y_pred) ** 2)
            else:
                mse_r = np.mean((y_true[:, :, 0] - y_pred[:, :, 0]) ** 2)
                mse_g = np.mean((y_true[:, :, 1] - y_pred[:, :, 1]) ** 2)
                mse_b = np.mean((y_true[:, :, 2] - y_pred[:, :, 2]) ** 2)
                mse = (mse_r + mse_g + mse_b) / 3.
            psnr = 20 * np.log10(255. / np.sqrt(mse))
            print("PSNR: {:.2f}dB for order {} and J={}".format(psnr, order, J))

            y_pred = y_pred.astype(np.uint8)
            name = img_name[:-4] + "_{}_{}_{:.2f}db".format(order, J, psnr) + img_name[-4:]
            cv2.imwrite(name, y_pred)


if __name__ == '__main__':
    main()
