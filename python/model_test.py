import tqdm
import numpy as np
from torch.utils.data import DataLoader

from datasets.loader import dataset
from config import *

TEST_MODEL = MODEL(pretrained=True, mode_path=Q_PATH, device=DEVICE, num_classes=NUM_CLASS)

def test(model, test_data):
    with torch.no_grad():
        testloader = DataLoader(test_data, batch_size=TEST_BATCH)
        acc_list = []
        for X, y in testloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = model(X)
            acc_list.append((pred.argmax(dim=1) == y).float().mean().item())
            
        return np.mean(acc_list)

# 先载入模型，然后修改参数到定点数，保存，然后再载入

if __name__ == "__main__":
    print(type(TEST_MODEL))
    # get model
    cur_model = TEST_MODEL

    # data get
    test_data = dataset(DATA_NAME, TEST_MODE)
    #
    # # model test
    cur_acc = test(cur_model, test_data)
    print(cur_acc)
