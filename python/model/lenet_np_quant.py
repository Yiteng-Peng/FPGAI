import numpy as np

def quantize_tensor(x, num_bits=8):
    qmin = 0.
    qmax = 2.**num_bits - 1.
    min_val, max_val = x.min(), x.max()

    scale = (max_val - min_val) / (qmax - qmin)

    initial_zero_point = qmin - min_val / scale

    zero_point = 0
    if initial_zero_point < qmin:
        zero_point = qmin
    elif initial_zero_point > qmax:
        zero_point = qmax
    else:
        zero_point = initial_zero_point

    zero_point = int(zero_point)
    q_x = zero_point + x / scale
    q_x.clamp_(qmin, qmax).round_()
    q_x = q_x.round().int()
    return q_x, scale, zero_point

def dequantize_tensor(q_x, scale, zero_point):
    return scale * (q_x.float() - zero_point)

def ReLU(x):
    return np.maximum(0, x)

class QuantLinear():
    def __init__(self, in_features, out_features):
        self.scale = None
        self.shift = None
        self.zero_point = None

        self.weight = np.zeros((in_features, out_features))
        self.bias = np.zeros((in_features, out_features))
        self.quant_weight = None
        self.quant_bias = None

    def load_quant(self, scale, shift, zero_point):
        self.scale = scale
        self.shift = shift
        self.zero_point = zero_point
        self.quant_weight = self.weight.astype(np.uint8)
        self.quant_bias = np.round((self.bias * 256) / (self.scale / 2 ** self.shift)).astype(np.int32)

    def forward(self, x):
        out = (np.dot(x, self.quant_weight.T.astype(np.int32))
               - np.dot(x, np.ones(self.quant_weight.T.shape, dtype=np.int32)) * self.zero_point
               + self.quant_bias)
        return (out * self.scale) >> self.shift

class QuantAvePool2d():
    def __init__(self, kernel_size=2, stride=2):
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        N, C, H, W = x.shape    # 数据个数，通道数，高，宽
        H_out = int((H - self.kernel_size) / self.stride + 1)
        W_out = int((W - self.kernel_size) / self.stride + 1)

        out = np.zeros((N, C, H_out, W_out), dtype=np.int32)
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * self.stride
                h_end = h_start + self.kernel_size
                w_start = j * self.stride
                w_end = w_start + self.kernel_size
                out[:, :, i, j] = np.max(x[:, :, h_start:h_end, w_start:w_end], axis=(2, 3))
        return out

class QuantConv2d():
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.scale = None
        self.shift = None
        self.zero_point = None

        self.stride = stride
        self.padding = padding

        self.weight = np.zeros((out_channels, in_channels, kernel_size, kernel_size))
        self.bias = np.zeros(out_channels)
        self.quant_weight = None
        self.quant_bias = None

    def load_quant(self, scale, shift, zero_point):
        self.scale = scale
        self.shift = shift
        self.zero_point = zero_point
        self.quant_weight = self.weight.astype(np.uint8)
        self.quant_bias = np.round((self.bias * 256) / (self.scale / 2 ** self.shift)).astype(np.int32)

    def forward(self, x):
        in_N, in_C, in_H, in_W = x.shape
        out_C, _, ker_H, ker_W = self.quant_weight.shape

        out_H = int(1 + (in_H + 2 * self.padding - ker_H) / self.stride)
        out_W = int(1 + (in_W + 2 * self.padding - ker_W) / self.stride)
        x_pad = np.pad(x, ((0, 0), (0, 0),
                           (self.padding, self.padding), (self.padding, self.padding)),
                       mode='constant')

        out = np.zeros((in_N, out_C, out_H, out_W), dtype=np.int32)
        for i in range(out_H):
            for j in range(out_W):
                h_start = i * self.stride
                h_end = h_start + ker_H
                w_start = j * self.stride
                w_end = w_start + ker_W
                x_slice = x_pad[:, :, h_start:h_end, w_start:w_end]
                x_col = x_slice.reshape((in_N, -1))
                quant_weight_col = self.quant_weight.reshape((out_C, -1))
                out[:, :, i, j] = np.dot(x_col, quant_weight_col.T.astype(np.int32)) - \
                                  np.dot(x_col, np.ones(quant_weight_col.T.shape, dtype=np.int32)) * self.zero_point + \
                                  self.quant_bias

        tmp = (out * self.scale)
        tmp = tmp >> self.shift
        return tmp

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
        x = ReLU(self.c1.forward(x))
        x = self.s2.forward(x)
        x = ReLU(self.c3.forward(x))
        x = self.s4.forward(x)
        x = self.c5.forward(x)
        x = x.reshape((x.shape[0], -1))
        x = self.f6.forward(x)
        x = self.output.forward(x)
        print(x)
        return x

    def load_quant(self, state_dict: dict,
                   c1_sc: int, c1_sh: int, c1_zp: int, c3_sc: int, c3_sh: int, c3_zp: int,
                   c5_sc: int, c5_sh: int, c5_zp: int, f6_sc: int, f6_sh: int, f6_zp: int,
                   out_sc: int, out_sh: int, out_zp: int):
        self.c1.weight = state_dict["c1.weight"].numpy()
        self.c1.bias = state_dict["c1.bias"].numpy()
        self.c1.load_quant(c1_sc, c1_sh, c1_zp)
        self.c3.weight = state_dict["c3.weight"].numpy()
        self.c3.bias = state_dict["c3.bias"].numpy()
        self.c3.load_quant(c3_sc, c3_sh, c3_zp)
        self.c5.weight = state_dict["c5.weight"].numpy()
        self.c5.bias = state_dict["c5.bias"].numpy()
        self.c5.load_quant(c5_sc, c5_sh, c5_zp)
        self.f6.weight = state_dict["f6.weight"].numpy()
        self.f6.bias = state_dict["f6.bias"].numpy()
        self.f6.load_quant(f6_sc, f6_sh, f6_zp)
        self.output.weight = state_dict["output.weight"].numpy()
        self.output.bias = state_dict["output.bias"].numpy()
        self.output.load_quant(out_sc, out_sh, out_zp)

if __name__ == "__main__":
    import torch
    from torchvision import datasets, transforms

    model = LeNet()
    state_dict = torch.load("./save_model/quant_model.pth", map_location="cpu")
    model.load_quant(state_dict,
                     23, 12, 99, 13, 12, 141, 3, 11, 128, 13, 13, 126, 13, 12, 127)

    data_transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root='./data', train=False, transform=data_transform, download=True)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=10)

    with torch.no_grad():
        acc_list = []
        for X, y in test_dataloader:
            X, y = (X.cpu().numpy() * 255).astype(np.uint8), y.cpu().numpy().astype(np.uint8)
            pred = model.forward(X)
            acc_list.append((pred.argmax(axis=1) == y).mean().item())
        print(np.mean(acc_list))
