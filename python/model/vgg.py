import torch.nn as nn

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
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

class VGG(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG, self).__init__()
        self.vgg_type = "A"
        self.features = make_layers(cfgs[self.vgg_type])

        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def eval_state_dict(self):
        eval_state_dict = self.state_dict()

        def pop_batchnorm(name:str):
            eval_state_dict.pop(name + ".weight")
            eval_state_dict.pop(name + ".bias")
            eval_state_dict.pop(name + ".running_mean")
            eval_state_dict.pop(name + ".running_var")
            eval_state_dict.pop(name + ".num_batches_tracked")

        # pop BatchNorm
        index = 0
        for v in cfgs[self.vgg_type]:
            if v == "M":
                index += 1      # Pool
            else:
                index += 1      # Conv2d
                pop_batchnorm("features.%d" % index)
                index += 2      # BatchNorm, ReLU

        def change_key(before, after):
            eval_state_dict[after + ".weight"] = eval_state_dict.pop(before + ".weight")
            eval_state_dict[after + ".bias"] = eval_state_dict.pop(before + ".bias")

        # rebuild state_dict
        before_index = 0
        after_index = 0
        for v in cfgs[self.vgg_type]:
            if v == "M":
                before_index += 1  # Pool
                after_index += 1
            else:
                change_key("features.%d" % before_index, "features.%d" % after_index)
                before_index += 3
                after_index += 2

        # classifier
        change_key("classifier.%d" % 0, "classifier.%d" % 0)
        change_key("classifier.%d" % 3, "classifier.%d" % 2)
        change_key("classifier.%d" % 6, "classifier.%d" % 4)

        return eval_state_dict
