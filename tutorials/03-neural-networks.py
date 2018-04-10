# 03-neural-networks.py

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)

# learnable parameters
params = list(net.parameters())
print("Number of learnable parameters: {0}".format(len(params)))
print("conv1 shape: {0}".format(params[0].size()))

# make dummy target to compute loss
input = Variable(torch.randn(1, 1, 32, 32))
output = net(input)
target = Variable(torch.arange(1, 11))
target = target.view(1, -1)
criterion = nn.MSELoss()
loss = criterion(output, target)
print("Network loss using dummy target: {0}".format(loss))

# calculate gradient
net.zero_grad() # zeroes out existing gradients
print("conv1.bias.grad before backward: {0}".format(net.conv1.bias.grad))
loss.backward()
print("conv1.bias.grad after backward: {0}".format(net.conv1.bias.grad))

# manually apply stochastic gradient descent (SGD)
# learning_rate = 0.01
# for f in net.parameters():
#     f.data.sub_(f.grad.data * learning_rate)

# use optim to apply SGD
import torch.optim as optim
optimizer = optim.SGD(net.parameters(), lr=0.01)

# apply once
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
print("conv1.bias before backward: {0}".format(net.conv1.bias))
loss.backward()
optimizer.step()    # Does the update
print("conv1.bias after backward: {0}".format(net.conv1.bias))