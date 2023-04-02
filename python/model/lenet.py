import torch
from torch import nn

class ReLU1(nn.Module):
    def forward(self, x):
        index = torch.where(x > 1)
        x[index] = 1
        index = torch.where(x < 0)
        x[index] = 0
        return x

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.Relu = ReLU1()
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c5 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)
        self.flatten = nn.Flatten()
        self.f6 = nn.Linear(120, 84)
        self.output = nn.Linear(84, 10)

    def forward(self, x):
        x = self.Relu(self.c1(x))
        x = self.s2(x)
        x = self.Relu(self.c3(x))
        x = self.s4(x)
        x = self.Relu(self.c5(x))
        x = self.flatten(x)
        x = self.Relu(self.f6(x))
        x = self.output(x)
        return x

    def eval_state_dict(self):
        return self.state_dict()
