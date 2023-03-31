import torch
import warnings
from torch import nn
import torch.nn.functional as F

# 定义量化和反量化函数
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

# 量化scale，将浮点的scale量化为最靠近的整型scale和偏移shift
def quant_scale(scale, num_bits=8):
    max_scale = 2**num_bits

    best_scale = round(scale.item())
    best_shift = 0
    min_diff = abs(scale - best_scale)

    cur_scale = scale
    for i in range(1, 32):
        cur_scale = cur_scale * 2
        cur_quant = round(cur_scale.item())
        if cur_quant > max_scale:
            break
        cur_diff = abs(cur_scale - cur_quant)
        if cur_diff < min_diff or best_scale == 0:
            min_diff = cur_diff
            best_scale = cur_quant
            best_shift = i

    if best_scale == 0:
        warnings.warn("scale is 0, please check by manual", RuntimeWarning)
    return best_scale, best_shift

# 定义量化卷积和量化全连接
class QuantLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(QuantLinear, self).__init__(in_features, out_features, bias)
        # out = conv(in * (q_x - z_p) + bias * 256 / scale) * scale
        self.quant_flag = False
        self.scale = None
        self.shift = None
        self.zero_point = None
        self.qx_minus_zeropoint = None
        self.bias_divide_scale = None

    def linear_quant(self, quantize_bit=8):
        self.weight.data, self.scale, self.zero_point = quantize_tensor(self.weight.data, num_bits=quantize_bit)
        self.scale, self.shift = quant_scale(self.scale, quantize_bit)
        self.bias_divide_scale = (self.bias * 256) / (self.scale / 2 ** self.shift)
        self.bias_divide_scale = self.bias_divide_scale.round().int()
        self.quant_flag = True

    def load_quant(self, scale, shift, zero_point):
        # true_scale = scale >> shift
        self.scale = scale
        self.shift = shift
        self.zero_point = zero_point

        self.qx_minus_zeropoint = self.weight - self.zero_point
        try:
            self.qx_minus_zeropoint = self.qx_minus_zeropoint.round().int()
        except:
            self.qx_minus_zeropoint = self.qx_minus_zeropoint
        self.bias_divide_scale = (self.bias * 256) / (self.scale / 2 ** self.shift)
        self.bias_divide_scale = self.bias_divide_scale.round().int()

        self.quant_flag = True

    def forward(self, x):
        if self.quant_flag == True:
            # weight = dequantize_tensor(self.weight, self.scale, self.zero_point)
            # return F.linear(x, weight, self.bias)
            return (F.linear(x.float(), self.qx_minus_zeropoint.float(), self.bias_divide_scale.float()) * self.scale).round().int() >> self.shift
        else:
            return F.linear(x, self.weight, self.bias)

class QuantAvePool2d(nn.AvgPool2d):
    def __init__(self, kernel_size, stride, padding=0):
        super(QuantAvePool2d, self).__init__(kernel_size, stride, padding)
        self.quant_flag = False

    def pool_quant(self, quantize_bit=8):
        self.quant_flag = True

    def load_quant(self):
        self.quant_flag = True

    def forward(self, x):
        if self.quant_flag == True:
            return F.avg_pool2d(x.float(), self.kernel_size, self.stride, self.padding).round().int()
        else:
            return F.avg_pool2d(x, self.kernel_size, self.stride, self.padding)

class QuantMaxPool2d(nn.MaxPool2d):
    def __init__(self, kernel_size, stride, padding=0):
        super(QuantMaxPool2d, self).__init__(kernel_size, stride, padding)
        self.quant_flag = False

    def pool_quant(self, quantize_bit=8):
        self.quant_flag = True

    def load_quant(self):
        self.quant_flag = True

    def forward(self, x):
        if self.quant_flag == True:
            return F.max_pool2d(x.float(), self.kernel_size, self.stride, self.padding).round().int()
        else:
            return F.max_pool2d(x, self.kernel_size, self.stride, self.padding)

class QuantConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(QuantConv2d, self).__init__(in_channels, out_channels,
                                          kernel_size, stride, padding, dilation, groups, bias)
        self.quant_flag = False
        self.scale = None
        self.shift = None
        self.zero_point = None
        self.qx_minus_zeropoint = None
        self.bias_divide_scale = None

    def conv_quant(self, quantize_bit=8):
        self.weight.data, self.scale, self.zero_point = quantize_tensor(self.weight.data, num_bits=quantize_bit)
        self.scale, self.shift = quant_scale(self.scale, quantize_bit)
        self.bias_divide_scale = (self.bias * 256) / (self.scale / 2 ** self.shift)
        self.bias_divide_scale = self.bias_divide_scale.round().int()
        self.quant_flag = True

    def load_quant(self, scale, shift, zero_point):
        # true_scale = scale >> shift
        self.scale = scale
        self.shift = shift
        self.zero_point = zero_point
        self.qx_minus_zeropoint = self.weight - self.zero_point
        try:
            self.qx_minus_zeropoint = self.qx_minus_zeropoint.round().int()
        except:
            self.qx_minus_zeropoint = self.qx_minus_zeropoint
        self.bias_divide_scale = (self.bias * 256) / (self.scale / 2 ** self.shift)
        self.bias_divide_scale = self.bias_divide_scale.round().int()
        self.quant_flag = True

    def forward(self, x):
        if self.quant_flag == True:
            # weight = dequantize_tensor(self.weight, self.scale, self.zero_point)
            # return F.conv2d(x, weight, self.bias, self.stride,
            #                 self.padding, self.dilation, self.groups)
            return (F.conv2d(x.float(), self.qx_minus_zeropoint.float(), self.bias_divide_scale.float(), self.stride,
                            self.padding, self.dilation, self.groups) * self.scale).round().int() >> self.shift
        else:
            return F.conv2d(x, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)