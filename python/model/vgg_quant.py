from model.quant import *

__all__ = [
    "VGG",
]

cfgs = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],                                       # vgg11
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],                              # vgg13
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],               # vgg16
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],# vgg19
}

def make_layers(cfg):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [QuantMaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = QuantConv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

class VGG(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG, self).__init__()
        self.vgg_type = "A"
        self.features = make_layers(cfgs[self.vgg_type])

        self.classifier = nn.Sequential(
            QuantLinear(512 * 1 * 1, 4096),
            nn.ReLU(inplace=True),
            QuantLinear(4096, 4096),
            nn.ReLU(inplace=True),
            QuantLinear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
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

        index = 0
        for v in cfgs[self.vgg_type]:
            if v == "M":
                index += 1      # Pool
            else:
                insert_layer_quant(self.features[index])
                index += 2

        insert_layer_quant(self.classifier[0])
        insert_layer_quant(self.classifier[2])
        insert_layer_quant(self.classifier[4])

        return quant_list

    def load_quant(self, quants):
        quant_index = 0

        def load_layer_quant(quant_layer):
            nonlocal quant_index
            if isinstance(quant_layer, QuantMaxPool2d):
                quant_layer.load_quant()
            else:
                quant_layer.load_quant(quants[quant_index],
                                       quants[quant_index+1],
                                       quants[quant_index+2])
                quant_index += 3

        index = 0
        for v in cfgs[self.vgg_type]:
            if v == "M":
                load_layer_quant(self.features[index])
                index += 1  # Pool
            else:
                load_layer_quant(self.features[index])
                index += 2

        load_layer_quant(self.classifier[0])
        load_layer_quant(self.classifier[2])
        load_layer_quant(self.classifier[4])
