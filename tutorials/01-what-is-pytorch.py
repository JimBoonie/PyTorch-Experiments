# 01-what-is-pytorch.py

import torch

x = torch.Tensor(5, 4)
print("New tensor:")
print(x)

x = torch.rand(5, 4)
print("Random tensor:")
print(x)

y = torch.rand(4, 5)
print("Matrix product of random tensors:")
print(torch.mm(x, y))