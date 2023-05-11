from config import *
import numpy as np
import time
import os
os.sched_setaffinity(0, {0})

# DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

if RAW_TEST:
    torch.set_num_threads(1)
    model = MODEL().to(DEVICE)
    model.load_state_dict(torch.load(RAW_MODEL_PATH))
    model.eval()

    test_dataset = TEST_DATASET
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=4)
    with torch.no_grad():
        start_time = time.time()
        acc_list = []
        for X, y in test_dataloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = model(X)
            acc_list.append((pred.argmax(dim=1) == y).float().mean().item())
        print("cost time:\t", time.time() - start_time)
        print("raw model:\t", np.mean(acc_list))

if QUANT_TEST:
    model = QUANT_MODEL().to(DEVICE)
    state_dict, quant_list = torch.load(QUANT_MODEL_PATH)
    model.load_state_dict(state_dict)
    model.load_quant(quant_list)
    model.eval()

    test_dataset = TEST_DATASET
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=4)
    with torch.no_grad():
        start_time = time.time()
        acc_list = []
        for X, y in test_dataloader:
            X, y = (X.to(DEVICE) * 255).round().int(), y.to(DEVICE)
            pred = model(X)
            acc_list.append((pred.argmax(dim=1) == y).float().mean().item())
        print("cost time:\t", time.time() - start_time)
        print("quant model:\t", np.mean(acc_list))

if NP_QUANT_TEST:
    model = NP_QUANT_MODEL()
    state_dict, quant_list = torch.load(QUANT_MODEL_PATH, map_location="cpu")
    model.load_quant(state_dict, quant_list)

    test_dataset = TEST_DATASET
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=4)
    with torch.no_grad():
        start_time = time.time()
        acc_list = []
        for X, y in test_dataloader:
            X, y = (X.cpu().numpy() * 255).astype(np.uint8), y.cpu().numpy().astype(np.uint8)
            pred = model.forward(X)
            acc_list.append((pred.argmax(axis=1) == y).mean().item())
        print("cost time:\t", time.time() - start_time)
        print("np quant model:\t", np.mean(acc_list))
