import torch
from torch import nn
from net_quant import LeNet
import time
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.autograd import Variable

def read_8bit_img(filepath):
    # 读取8bit数据
    image = Image.open(filepath).convert('L')
    resize = transforms.Resize([28, 28])
    image = resize(image)
    image = np.copy(image)
    image = torch.tensor(image)
    image = Variable(torch.unsqueeze(torch.unsqueeze(image, dim=0).int(), dim=0).int()).to(device)
    image = image.clone().detach().to(device)
    return image

def read_float_img(filepath):
    image = Image.open(filepath).convert('L')
    resize = transforms.Resize([28, 28])
    image = resize(image)
    image = np.copy(image)
    image = torch.tensor(image)
    image = Variable(torch.unsqueeze(torch.unsqueeze(image, dim=0).float(), dim=0).float()).to(device)
    image = image.clone().detach().to(device)
    return image

device = "cuda" if torch.cuda.is_available() else "cpu"

# 调用net定义的模型
model1 = LeNet().to(device)
model1.load_state_dict(torch.load("./save_model/best_model.pth"))

model2 = LeNet().to(device)
model2.load_state_dict(torch.load("./save_model/quant_model.pth"))
model2.load_quant(23, 12, 99, 13, 12, 141, 3, 11, 128, 13, 13, 126, 13, 12, 127)

# 定义损失函数（交叉熵）
loss_fn = nn.CrossEntropyLoss()

# 分类类别
classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

model1.eval()
model2.eval()

float_image1 = read_float_img('data/mydata/2/2.jpg')
# float_image2 = read_float_img('data/mydata/4/4.jpg')
byte_image1 = read_8bit_img('data/mydata/5/5.jpg')
# byte_image2 = read_8bit_img('data/mydata/4/4.jpg')

# 量化前测试
print("量化前测试")
start1 = time.time()
with torch.no_grad():
    for i in range(1):
        pred = model1(float_image1)
end1 = time.time()
predicted= classes[torch.argmax(pred[0])]
print(f'predicted:"{predicted}"')
print("耗时" + str(end1 - start1))
print("#" * 20)

# 量化后测试
# model2.linear_quant()
print("量化后测试")
start2 = time.time()
with torch.no_grad():
    for i in range(1):
        pred = model2(byte_image1)
end2 = time.time()
predicted = classes[torch.argmax(pred[0])]
print(f'predicted:"{predicted}"')
print("耗时" + str(end2 - start2))

