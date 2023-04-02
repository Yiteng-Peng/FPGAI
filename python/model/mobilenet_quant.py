from model.quant import *

# DW卷积
def Conv3x3BNReLU(in_channels, out_channels, stride, groups):
    return nn.Sequential(
        QuantConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1,
                    groups=groups, bias=False),
        QuantReLU6(inplace=True)
    )


# PW卷积
def Conv1x1BNReLU(in_channels, out_channels):
    return nn.Sequential(
        QuantConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=False),
        QuantReLU6(inplace=True)
    )


# PW卷积（没有使用激活函数）
def Conv1x1BN(in_channels, out_channels):
    return nn.Sequential(
        QuantConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=False),
    )


class Msg(nn.Module):
    def __init__(self, msg):
        super(Msg, self).__init__()
        self.msg = msg

    def forward(self, x):
        print(self.msg)
        return x


class InvertedResidual(nn.Module):
    # t = expansion_factor,也就是扩展因子，文章中取6
    def __init__(self, in_channels, out_channels, expansion_factor, stride):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        mid_channels = (in_channels * expansion_factor)
        # 先1x1卷积升维，再1x1卷积降维
        self.bottleneck = nn.Sequential(
            # 升维操作: 扩充维度是 in_channels * expansion_factor (6倍)
            Conv1x1BNReLU(in_channels, mid_channels),
            # DW卷积,降低参数量
            Conv3x3BNReLU(mid_channels, mid_channels, stride, groups=mid_channels),
            # 降维操作: 降维度 in_channels * expansion_factor(6倍) 降维到指定 out_channels 维度
            Conv1x1BN(mid_channels, out_channels)
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.bottleneck(x)
        else:
            return self.bottleneck(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=10):
        super(MobileNetV2, self).__init__()
        self.inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 1],  # Stride 2 -> 1 for CIFAR-10
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        features = []
        # 初始层，将3通道转为32通道
        features += [
            Conv3x3BNReLU(in_channels=3, out_channels=32, stride=2, groups=1),
        ]

        # building inverted residual blocks
        input_channels = 32
        for t, c, n, s in self.inverted_residual_setting:
            output_channels = c
            expansion_factor = t
            for i in range(n):
                stride = s if i == 0 else 1
                features += [
                    InvertedResidual(input_channels, output_channels, expansion_factor, stride),
                ]
                input_channels = output_channels

        # 升维，将320维升至1280维
        features += [
            Conv1x1BNReLU(320, 1280),
        ]
        self.features = nn.Sequential(*features)

        # 线性层 用来预测
        self.classifier = nn.Sequential(
            QuantLinear(1280, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
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
        index = 0
        insert_layer_quant(self.features[index][0])
        index += 1

        # 中间层
        for t, c, n, s in self.inverted_residual_setting:
            for i in range(n):
                insert_layer_quant(self.features[index].bottleneck[0][0])
                insert_layer_quant(self.features[index].bottleneck[1][0])
                insert_layer_quant(self.features[index].bottleneck[2][0])
                index += 1

        # 升维层
        insert_layer_quant(self.features[index][0])
        # 分类层
        insert_layer_quant(self.classifier[0])

    def load_quant(self, quants):
        pass
