import torch
from torch import nn, optim
import models

# device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# dataset
DATA_NAME = "CIFAR10"
NUM_CLASS = 10

TRAIN_MODE = "train"
TEST_MODE = "test"
COV_MODE = TEST_MODE

# model
MODEL_NAME = "mobilenet_v2"
MODEL = models.mobilenet_v2

# train model
LOSS_FUNC = nn.CrossEntropyLoss()
EPOCHS = 3
BATCH_SIZE = 128
SHUFFLE = True

################################# train model #####################################
# save
SAVE_MODE = "s"     # 'm' for origin model, 's' for state_dict
SAVE_PATH = "pretrained/%s.pt" % (MODEL_NAME + "_" + DATA_NAME + "_@" + SAVE_MODE)

################################ test model ########################################
# load
LOAD_MODE = "s"
LOAD_PATH = "pretrained/%s.pt" % (MODEL_NAME + "_" + DATA_NAME + "_@" + LOAD_MODE)

# test model
TEST_BATCH = 1000

################################ quantization model ################################
Q_PATH = "pretrained/%sq.pt" % (MODEL_NAME + "_" + DATA_NAME + "_@" + LOAD_MODE)
