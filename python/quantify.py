import collections
from config import *

Q_MODEL = MODEL(pretrained=True, mode_path=LOAD_PATH, device="cpu", num_classes=NUM_CLASS)


def float2int(state_dict, bit_width):
    int_range = 2**(bit_width-1) - 1
    qstate_list = []

    s = []
    for key in state_dict:
        if "conv" in key:
            channels = state_dict[key].shape[0]
            temp_tensor = torch.zeros(state_dict[key].shape, dtype=torch.int8)
            if "weight" in key:
                for i in range(channels):
                    kernel = state_dict[key][i]
                    c_item = torch.max(kernel.min().abs(), kernel.max().abs())
                    s_item = c_item / int_range
                    s.append(s_item)
                    temp_tensor[i] = (kernel / s_item).to(torch.int8)
                qstate_list.append((key, temp_tensor))
            else:
                for i in range(channels):
                    bias = state_dict[key][i]
                    s_item = s[i]
                    temp_tensor[i] = (bias / s_item).to(torch.int8)
                s = []
                qstate_list.append((key, temp_tensor))
        else:
            if "weight" in key:
                fc_params = state_dict[key]
                c_item = torch.max(fc_params.min().abs(), fc_params.max().abs())
                s_item = c_item / int_range
                s.append(s_item)
                temp_tensor = (fc_params / s_item).to(torch.int8)
                qstate_list.append((key, temp_tensor))
            else:
                fc_params = state_dict[key]
                s_item = s[0]
                temp_tensor = (fc_params / s_item).to(torch.int8)
                s = []
                qstate_list.append((key, temp_tensor))
    return qstate_list


def float2uint(state_dict, bit_width):
    int_bias = 2**(bit_width-1) // 2 - 1
    qstate_list = float2int(state_dict, bit_width)
    for i in range(len(qstate_list)):
        qstate_list[i] = (qstate_list[i][0], (qstate_list[i][1] + int_bias).to(torch.uint8))
    return qstate_list


if __name__ == "__main__":
    
    # origin
    state_dict = torch.load(LOAD_PATH, map_location="cpu")
    print(state_dict)
    qstate_list = float2uint(state_dict, 8)
    print(qstate_list)
    torch.save(collections.OrderedDict(qstate_list), Q_PATH)
    