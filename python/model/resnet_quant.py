from model.quant import *

__all__ = [
    "ResNet",
]

class ReLU1(nn.Module):
    def forward(self, x):
        index = torch.where(x > 1)
        x[index] = 1
        index = torch.where(x < 0)
        x[index] = 0
        return x


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
        block, layers = cfgs["18"]

        self.inplanes = 64
        self.conv1 = QuantConv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1)
        self.relu = QuantReLU1()
        self.maxpool = QuantMaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = QuantAvePool2d(kernel_size=2, stride=2)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.fc = QuantLinear(512 * block.expansion, num_classes)

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

        x = self.avgpool(x)
        x = self.fc(x)

        return x
