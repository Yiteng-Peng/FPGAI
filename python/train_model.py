from tqdm import tqdm
from torch import nn
from config import *
from torch.optim import lr_scheduler

# 加载训练数据集
train_dataset = TRAIN_DATASET
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# 调用net定义的模型
model = MODEL().to(DEVICE)
try:
    model.load_state_dict(torch.load(RAW_MODEL_PATH))
except:
    pass

# 定义损失函数（交叉熵）
loss_fn = nn.CrossEntropyLoss()

# 定义一个优化器
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

# 学习率每隔10轮，变为原来的0.5
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# 定义训练函数
def train(dataloader, model, loss_fn, optimizer):
    model.train()
    loss, current, n = 0.0, 0.0, 0
    for batch, (X, y) in enumerate(tqdm(dataloader)):
        # 前向传播
        X, y = X.to(DEVICE), y.to(DEVICE)
        output = model(X)
        cur_loss = loss_fn(output, y)
        _, pred = torch.max(output, axis=1)

        cur_acc = torch.sum(y == pred)/output.shape[0]

        optimizer.zero_grad()
        cur_loss.backward()
        optimizer.step()

        loss += cur_loss.item()
        current += cur_acc.item()
        n = n + 1

    train_loss = loss / n
    train_acc = current / n
    print("train_loss" + str(train_loss))
    print("train_acc" + str(train_acc))

    return train_loss, train_acc


if __name__ == "__main__":
    print("use ", DEVICE)
    # 开始训练
    epoch = EPOCH
    min_loss = 0

    for t in range(epoch):
        print(f'epoch{t+1}\n------------------')
        train_loss, train_acc = train(train_dataloader, model, loss_fn, optimizer)

        if t == 0:
            min_loss = train_loss

        # 保存最好的模型权重
        if train_loss <= min_loss:
            min_loss = train_loss
            print('save best model')
            torch.save(model.state_dict(), RAW_MODEL_PATH)
