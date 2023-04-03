import torch
import model
from torchvision import datasets, transforms

# 设备
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 数据集:[MNIST, CIFAR10, CIFAR100]
data_transform = transforms.Compose([transforms.ToTensor()])
TRAIN_DATASET = datasets.CIFAR10(root='./data', train=True, transform=data_transform, download=True)
TEST_DATASET = datasets.CIFAR10(root='./data', train=False, transform=data_transform, download=True)

MODEL_NAME = "ResNet50"
# 训练
MODEL = model.resnet.ResNet().to(DEVICE)
EPOCH = 20
RAW_MODEL_PATH = "./pretrained/%s_raw.pth" % MODEL_NAME

# 量化
QUANT_MODEL = None  # model.lenet.LeNet().to(DEVICE)
QUANT_MODEL_PATH = "./quantization/%s_quant.tuple" % MODEL_NAME

# numpy量化
NP_QUANT_MODEL = None   # model.lenet_np_quant.LeNet()

# 测试
RAW_TEST = True
QUANT_TEST = False
NP_QUANT_TEST = False

# 导出
WEIGHT_TXT_PATH = "./txt_model/%s_weight.txt" % MODEL_NAME
QUANT_TXT_PATH = "./txt_model/%s_quant.txt" % MODEL_NAME
