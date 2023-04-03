from config import *
import numpy as np
from tqdm import tqdm
from model.quant import QuantConv2d, QuantLinear

model = QUANT_MODEL()
state_dict, quant_list = torch.load(QUANT_MODEL_PATH)
model.load_state_dict(state_dict)
model.load_quant(quant_list)


def write_txt_layer_param(fp_param, layer):
    # 每100个数据分一段
    # weight
    fp_param.write("#\n")
    if isinstance(layer, QuantConv2d):
        num = 0
        buf = ""
        for i in range(layer.weight.shape[0]):
            for j in range(layer.weight.shape[1]):
                for k in range(layer.weight.shape[2]):
                    for index in range(layer.weight.shape[3]):
                        buf += str(int(layer.weight[i][j][k][index]))
                        num += 1
                        if num % 10 == 0 or num == layer.weight.numel():
                            buf += "\n"
                            fp_param.write(buf)
                            buf = ""
                        else:
                            buf += ","
    else:
        num = 0
        buf = ""
        for i in range(layer.weight.shape[0]):
            for j in range(layer.weight.shape[1]):
                buf += str(int(layer.weight[i][j]))
                num += 1
                if num % 10 == 0 or num == layer.weight.numel():
                    buf += "\n"
                    fp_param.write(buf)
                    buf = ""
                else:
                    buf += ","
    # bias
    if layer.bias_divide_scale is not None:
        fp_param.write("$\n")
        num = 0
        buf = ""
        for i in range(layer.bias_divide_scale.shape[0]):
            buf += str(int(layer.bias_divide_scale[i]))
            num += 1
            if num % 10 == 0 or num == layer.bias_divide_scale.numel():
                buf += "\n"
                fp_param.write(buf)
                buf = ""
            else:
                buf += ","


def write_txt_layer_quant(fp_quant, layer):
    # quant
    fp_quant.write(str(layer.scale))
    fp_quant.write(",")
    fp_quant.write(str(layer.shift))
    fp_quant.write(",")
    fp_quant.write(str(layer.zero_point))
    fp_quant.write("\n")


def write_bin_layer_param(fp_weight, fp_bias, layer):
    np_weight = layer.weight.cpu().numpy()
    np_weight = np_weight.astype(np.uint8)
    np_weight.tofile(fp_weight)
    np_bias = layer.bias_divide_scale.cpu().numpy()
    np_bias = np_bias.astype(np.int32)
    np_bias.tofile(fp_bias)


def export_quant_model(model):
    if EXPORT_MODE == "txt":
        fp_param = open(WEIGHT_TXT_PATH, "w")
        fp_quant = open(QUANT_TXT_PATH, "w")
        for i, m in enumerate(tqdm(model.modules())):
            if isinstance(m, QuantConv2d) or isinstance(m, QuantLinear):
                write_txt_layer_param(fp_param, m)
                write_txt_layer_quant(fp_quant, m)
        fp_param.close()
        fp_quant.close()
    elif EXPORT_MODE == "bin":
        fp_weight = open(WEIGHT_BIN_PATH, "wb")
        fp_bias = open(BIAS_BIN_PATH, "wb")
        fp_quant = open(QUANT_TXT_PATH, "w")
        for i, m in enumerate(tqdm(model.modules())):
            if isinstance(m, QuantConv2d) or isinstance(m, QuantLinear):
                write_bin_layer_param(fp_weight, fp_bias, m)
                write_txt_layer_quant(fp_quant, m)
        fp_weight.close()
        fp_bias.close()
        fp_quant.close()
    else:
        raise ValueError("EXPORT_MODE should bin or txt")

with torch.no_grad():
    export_quant_model(model)
