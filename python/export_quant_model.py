import torch
from net_quant import *
import numpy as np
from torchvision import datasets, transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

model = LeNet().to(device)
model.load_state_dict(torch.load("./save_model/quant_model.pth"))
model.load_quant(23, 12, 99, 13, 12, 141, 3, 11, 128, 13, 13, 126, 13, 12, 127)
model.eval()

data_transform = transforms.Compose([transforms.ToTensor()])
test_dataset = datasets.MNIST(root='./data', train=False, transform=data_transform, download=True)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1)

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

PARAM_PATH = "./for_C/lenet_param.txt"
QUANT_PATH = "./for_C/lenet_quant.txt"
def export_model_C(model):
    fp_param = open(PARAM_PATH, "w")
    fp_quant = open(QUANT_PATH, "w")
    write_layer(fp_param, fp_quant, model.c1)
    write_layer(fp_param, fp_quant, model.c3)
    write_layer(fp_param, fp_quant, model.c5)
    write_layer(fp_param, fp_quant, model.f6)
    write_layer(fp_param, fp_quant, model.output)
    fp_param.close()
    fp_quant.close()

# export_model_C(model)
# exit(0)

with torch.no_grad():
    acc_list = []
    for X, y in test_dataloader:
        X, y = (X.to(device) * 255).round().int(), y.to(device)
        pred = model(X)
        acc_list.append((pred.argmax(dim=1) == y).float().mean().item())
    print(np.mean(acc_list))
