from model.quant import *

# 定义网络模型
class LeNet(nn.Module):
    # 初始化网络
    def __init__(self):
        super(LeNet, self).__init__()

        self.c1 = QuantConv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.Relu = QuantReLU1()
        self.s2 = QuantAvePool2d(kernel_size=2, stride=2)
        self.c3 = QuantConv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.s4 = QuantAvePool2d(kernel_size=2, stride=2)
        self.c5 = QuantConv2d(in_channels=16, out_channels=120, kernel_size=5)
        self.flatten = nn.Flatten()
        self.f6 = QuantLinear(120, 84)
        self.output = QuantLinear(84, 10)

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

    def linear_quant(self, quantize_bit=8):
        for m in self.modules():
            if isinstance(m, QuantConv2d):
                m.conv_quant(quantize_bit)
            elif isinstance(m, QuantMaxPool2d):
                m.pool_quant(quantize_bit)
            elif isinstance(m, QuantLinear):
                m.linear_quant(quantize_bit)

    def get_quant(self):
        quant_list = [
            self.c1.scale, self.c1.shift, self.c1.zero_point,
            self.c3.scale, self.c3.shift, self.c3.zero_point,
            self.c5.scale, self.c5.shift, self.c5.zero_point,
            self.f6.scale, self.f6.shift, self.f6.zero_point,
            self.output.scale, self.output.shift, self.output.zero_point,
        ]
        return quant_list

    def load_quant(self, quants):
        self.c1.load_quant(quants[0], quants[1], quants[2])
        self.s2.load_quant()
        self.c3.load_quant(quants[3], quants[4], quants[5])
        self.s4.load_quant()
        self.c5.load_quant(quants[6], quants[7], quants[8])
        self.f6.load_quant(quants[9], quants[10], quants[11])
        self.output.load_quant(quants[12], quants[13], quants[14])
