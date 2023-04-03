import torch.nn as nn
import torch

__all__ = [
    "MobileNetV2",
]

class ReLU1(nn.Module):
    def forward(self, x):
        index = torch.where(x > 1)
        x[index] = 1
        index = torch.where(x < 0)
        x[index] = 0
        return x

# DW卷积
def Conv3x3ReLU(in_channels, out_channels, stride):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            ReLU1()
        )

# PW卷积
def Conv1x1ReLU(in_channels,out_channels):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=False),
            ReLU1()
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
            Conv1x1ReLU(in_channels, mid_channels),
            # DW卷积,降低参数量
            Conv3x3ReLU(mid_channels, mid_channels, stride),
            # 降维操作: 降维度 in_channels * expansion_factor(6倍) 降维到指定 out_channels 维度
            Conv1x1ReLU(mid_channels, out_channels)
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
            Conv3x3ReLU(in_channels=3, out_channels=32, stride=2),
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
            Conv1x1ReLU(320, 1280),
        ]
        self.features = nn.Sequential(*features)

        # 线性层 用来预测
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x

    def eval_state_dict(self):
        # 把字典重命名，把BatchNorm对应的地方弹出
        # 因为量化模型用于评估，用不上这些参数，带着反而不方便之后的操作
        eval_state_dict = self.state_dict()

        def pop_batchnorm(name:str):
            eval_state_dict.pop(name + ".weight")
            eval_state_dict.pop(name + ".bias")
            eval_state_dict.pop(name + ".running_mean")
            eval_state_dict.pop(name + ".running_var")
            eval_state_dict.pop(name + ".num_batches_tracked")

        # 初始层
        index = 0       # 在features中所处的层的位置
        pop_batchnorm("features.%s.1" % index)
        index += 1

        for t, c, n, s in self.inverted_residual_setting:
            for i in range(n):
                key = "features.%s.bottleneck.%s.1"
                pop_batchnorm(key % (index, "0"))
                pop_batchnorm(key % (index, "1"))
                pop_batchnorm(key % (index, "2"))
                index += 1

        # 升维层
        pop_batchnorm("features.%s.1" % index)

        # 分类层
        eval_state_dict["classifier.0.weight"] = \
            eval_state_dict.pop("classifier.1.weight")
        eval_state_dict["classifier.0.bias"] = \
            eval_state_dict.pop("classifier.1.bias")

        return eval_state_dict
