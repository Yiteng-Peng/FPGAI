from config import *
from model.quant import QuantConv2d, QuantLinear

model = QUANT_MODEL
state_dict, quant_list = torch.load(QUANT_MODEL_PATH)
model.load_state_dict(state_dict)
model.load_quant(quant_list)

def write_layer(fp_param, fp_quant, layer):
    # 每10个数据分一段
    # weight
    fp_param.write("#\n")
    if isinstance(layer, QuantConv2d):
        num = 0
        for i in range(layer.weight.shape[0]):
            for j in range(layer.weight.shape[1]):
                for k in range(layer.weight.shape[2]):
                    for index in range(layer.weight.shape[3]):
                        fp_param.write(str(int(layer.weight[i][j][k][index])))
                        num += 1
                        if num % 10 == 0 or num == layer.weight.numel():
                            fp_param.write("\n")
                        else:
                            fp_param.write(",")
    else:
        num = 0
        for i in range(layer.weight.shape[0]):
            for j in range(layer.weight.shape[1]):
                fp_param.write(str(int(layer.weight[i][j])))
                num += 1
                if num % 10 == 0 or num == layer.weight.numel():
                    fp_param.write("\n")
                else:
                    fp_param.write(",")
    # bias
    fp_param.write("$\n")
    num = 0
    for i in range(layer.bias_divide_scale.shape[0]):
        fp_param.write(str(int(layer.bias_divide_scale[i])))
        num += 1
        if num % 10 == 0 or num == layer.bias_divide_scale.numel():
            fp_param.write("\n")
        else:
            fp_param.write(",")

    # quant
    fp_quant.write(str(layer.scale))
    fp_quant.write(",")
    fp_quant.write(str(layer.shift))
    fp_quant.write(",")
    fp_quant.write(str(layer.zero_point))
    fp_quant.write("\n")

def export_quant_model(model):
    fp_param = open(WEIGHT_TXT_PATH, "w")
    fp_quant = open(QUANT_TXT_PATH, "w")

    # if MODEL_NAME == "lenet":
    #     write_layer(fp_param, fp_quant, model.c1)
    #     write_layer(fp_param, fp_quant, model.c3)
    #     write_layer(fp_param, fp_quant, model.c5)
    #     write_layer(fp_param, fp_quant, model.f6)
    #     write_layer(fp_param, fp_quant, model.output)
    for i, m in enumerate(model.modules()):
        if isinstance(m, QuantConv2d) or isinstance(m, QuantLinear):
            write_layer(fp_param, fp_quant, m)
            print("layer %d", i)

    fp_param.close()
    fp_quant.close()


export_quant_model(model)
