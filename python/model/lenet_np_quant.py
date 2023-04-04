from model.np_quant import *

class LeNet():
    def __init__(self):
        self.c1 = QuantConv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.s2 = QuantAvePool2d(kernel_size=2, stride=2)
        self.c3 = QuantConv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.s4 = QuantAvePool2d(kernel_size=2, stride=2)
        self.c5 = QuantConv2d(in_channels=16, out_channels=120, kernel_size=5)
        self.f6 = QuantLinear(120, 84)
        self.output = QuantLinear(84, 10)

    def forward(self, x):
        x = ReLU1(self.c1.forward(x))
        x = self.s2.forward(x)
        x = ReLU1(self.c3.forward(x))
        x = self.s4.forward(x)
        x = ReLU1(self.c5.forward(x))
        x = x.reshape((x.shape[0], -1))
        x = ReLU1(self.f6.forward(x))
        x = self.output.forward(x)
        return x

    def load_quant(self, state_dict, quants):
        self.c1.weight = state_dict["c1.weight"].numpy()
        self.c1.bias = state_dict["c1.bias"].numpy()
        self.c1.load_quant(quants[0], quants[1], quants[2])

        self.c3.weight = state_dict["c3.weight"].numpy()
        self.c3.bias = state_dict["c3.bias"].numpy()
        self.c3.load_quant(quants[3], quants[4], quants[5])

        self.c5.weight = state_dict["c5.weight"].numpy()
        self.c5.bias = state_dict["c5.bias"].numpy()
        self.c5.load_quant(quants[6], quants[7], quants[8])

        self.f6.weight = state_dict["f6.weight"].numpy()
        self.f6.bias = state_dict["f6.bias"].numpy()
        self.f6.load_quant(quants[9], quants[10], quants[11])

        self.output.weight = state_dict["output.weight"].numpy()
        self.output.bias = state_dict["output.bias"].numpy()
        self.output.load_quant(quants[12], quants[13], quants[14])
