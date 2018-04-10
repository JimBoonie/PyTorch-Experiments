# 02-autograd.py

import torch
from torch.autograd import Variable

x = Variable(torch.ones(2, 2), requires_grad=True)
print("Variable x:")
print(x)
print("Variable.data of x:")
print(x.data)

out = x.mean()
print("Output of short network:")
print(out)

out.backward()
print("Variable.grad of x after .backward():")
print(x.grad)

x = Variable(torch.randn(3), requires_grad=True)
y = x * 2
while y.data.norm() < 1000:
	y = y * 2

print(y)

gradients = torch.FloatTensor([0.1, 1.0, 0.0001])
y.backward(gradients)
print(x.grad)