from config import *
import numpy as np

# DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

if RAW_TEST:
    model = MODEL().to(DEVICE)
    model.eval()
    model.load_state_dict(torch.load(RAW_MODEL_PATH))
    test_dataset = TEST_DATASET
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100)
    with torch.no_grad():
        acc_list = []
        for X, y in test_dataloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = model(X)
            acc_list.append((pred.argmax(dim=1) == y).float().mean().item())
        print("raw model:\t", np.mean(acc_list))

if QUANT_TEST:
    model = QUANT_MODEL().to(DEVICE)
    model.eval()
    state_dict, quant_list = torch.load(QUANT_MODEL_PATH)
    model.load_state_dict(state_dict)
    model.load_quant(quant_list)

    test_dataset = TEST_DATASET
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100)
    with torch.no_grad():
        acc_list = []
        for X, y in test_dataloader:
            X, y = (X.to(DEVICE) * 255).round().int(), y.to(DEVICE)
            pred = model(X)
            acc_list.append((pred.argmax(dim=1) == y).float().mean().item())
        print("quant model:\t", np.mean(acc_list))

if NP_QUANT_TEST:
    model = NP_QUANT_MODEL().to(DEVICE)
    state_dict, quant_list = torch.load(QUANT_MODEL_PATH, map_location="cpu")
    model.load_quant(state_dict, quant_list)

    test_dataset = TEST_DATASET
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100)
    with torch.no_grad():
        acc_list = []
        for X, y in test_dataloader:
            X, y = (X.cpu().numpy() * 255).astype(np.uint8), y.cpu().numpy().astype(np.uint8)
            pred = model.forward(X)
            acc_list.append((pred.argmax(axis=1) == y).mean().item())
        print("np quant model:\t", np.mean(acc_list))
