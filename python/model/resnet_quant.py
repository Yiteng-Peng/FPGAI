import torch

from model.quant import *

__all__ = [
    "ResNet",
]

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return QuantConv2d(in_planes, out_planes, kernel_size=3, padding=1, stride=stride)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return QuantConv2d(in_planes, out_planes, kernel_size=1, stride=stride)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride=stride)
        self.relu = QuantReLU1()
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # 不要对x重新赋值，保证x不变，以便于后面进行残差链接
        out = self.conv1(x)
        out = self.relu(out)

        # 如果有downsample就代表通道数不一致，那么就扩展之后再残差链接
        out = self.conv2(out)
        if out.dtype == torch.int:  # 量化模式
            out += x.int() if self.downsample is None else self.downsample(x)
        else:
            out += x if self.downsample is None else self.downsample(x)
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.relu = QuantReLU1()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out)

        out = self.conv3(out)
        if out.dtype == torch.int:  # 量化模式
            out += x.int() if self.downsample is None else self.downsample(x)
        else:
            out += x if self.downsample is None else self.downsample(x)
        out = self.relu(out)

        return out


cfgs = {
    "18": (BasicBlock, [2, 2, 2, 2]),
    "34": (BasicBlock, [3, 4, 6, 3]),
    "50": (Bottleneck, [3, 4, 6, 3]),
}


class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()
        self.block, self.layers = cfgs["50"]

        self.inplanes = 64
        self.conv1 = QuantConv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1)
        self.relu = QuantReLU1()
        self.maxpool = QuantMaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = QuantAvePool2d(kernel_size=2, stride=2)
        self.layer1 = self._make_layer(self.block, 64, self.layers[0], stride=1)
        self.layer2 = self._make_layer(self.block, 128, self.layers[1], stride=2)
        self.layer3 = self._make_layer(self.block, 256, self.layers[2], stride=2)
        self.layer4 = self._make_layer(self.block, 512, self.layers[3], stride=2)
        self.fc = QuantLinear(512 * self.block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def _make_layer(self, block, planes, block_num, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = conv1x1(self.inplanes, planes * block.expansion, stride)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))

        self.inplanes = planes * block.expansion
        for _ in range(1, block_num):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.float().mean([2, 3]).int()
        x = self.fc(x)

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
        quant_list = []

        def insert_layer_quant(quant_layer):
            quant_list.append(quant_layer.scale)
            quant_list.append(quant_layer.shift)
            quant_list.append(quant_layer.zero_point)

        # 初始层
        insert_layer_quant(self.conv1)

        layers_list = [self.layer1, self.layer2, self.layer3, self.layer4]
        if self.block == BasicBlock:
            for i in range(len(layers_list)):
                for num in range(self.layers[i]):
                    insert_layer_quant(layers_list[i][num].conv1)
                    insert_layer_quant(layers_list[i][num].conv2)
                    if layers_list[i][num].downsample:
                        insert_layer_quant(layers_list[i][num].downsample)
        elif self.block == Bottleneck:
            for i in range(len(layers_list)):
                for num in range(self.layers[i]):
                    insert_layer_quant(layers_list[i][num].conv1)
                    insert_layer_quant(layers_list[i][num].conv2)
                    insert_layer_quant(layers_list[i][num].conv3)
                    if layers_list[i][num].downsample:
                        insert_layer_quant(layers_list[i][num].downsample)

        # 全连接
        insert_layer_quant(self.fc)

        return quant_list

    def load_quant(self, quants):
        # 初始层
        index = 0
        self.conv1.load_quant(quants[index], quants[index + 1], quants[index + 2])
        index += 3

        layers_list = [self.layer1, self.layer2, self.layer3, self.layer4]
        if self.block == BasicBlock:
            for i in range(len(layers_list)):
                for num in range(self.layers[i]):
                    layers_list[i][num].conv1.load_quant(quants[index], quants[index + 1], quants[index + 2])
                    index += 3
                    layers_list[i][num].conv2.load_quant(quants[index], quants[index + 1], quants[index + 2])
                    index += 3
                    if layers_list[i][num].downsample:
                        layers_list[i][num].downsample.load_quant(quants[index], quants[index + 1], quants[index + 2])
                        index += 3
        elif self.block == Bottleneck:
            for i in range(len(layers_list)):
                for num in range(self.layers[i]):
                    layers_list[i][num].conv1.load_quant(quants[index], quants[index + 1], quants[index + 2])
                    index += 3
                    layers_list[i][num].conv2.load_quant(quants[index], quants[index + 1], quants[index + 2])
                    index += 3
                    layers_list[i][num].conv3.load_quant(quants[index], quants[index + 1], quants[index + 2])
                    index += 3
                    if layers_list[i][num].downsample:
                        layers_list[i][num].downsample.load_quant(quants[index], quants[index + 1], quants[index + 2])
                        index += 3

        # 全连接
        self.fc.load_quant(quants[index], quants[index + 1], quants[index + 2])
