import numpy as np

def ReLU1(x):
    x = np.where(x > 255, 255, x)
    x = np.where(x < 0, 0, x)
    return x

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

class QuantMaxPool2d():
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
                out[:, :, i, j] = np.mean(x[:, :, h_start:h_end, w_start:w_end], axis=(2, 3)).astype(np.int32)
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