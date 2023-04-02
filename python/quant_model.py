from config import *

# 调用net定义的模型
model = QUANT_MODEL
# normal
raw_model = MODEL
raw_state_dict = torch.load(RAW_MODEL_PATH)
raw_model.load_state_dict(raw_state_dict)
state_dict = raw_model.eval_state_dict()

model.load_state_dict(state_dict)

# 量化
model.linear_quant()

# 模型保存
torch.save((model.state_dict(), model.get_quant()), QUANT_MODEL_PATH)
