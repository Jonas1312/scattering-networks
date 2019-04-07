import torch
from kymatio import Scattering2D

scattering = Scattering2D(J=3, shape=(32, 32))
x = torch.randn(1, 1, 32, 32)
Sx = scattering(x)
print(Sx.size())
