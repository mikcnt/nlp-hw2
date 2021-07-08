import torch
from collections import OrderedDict


def rename_parameters(checkpoint_path, new_path):
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint["state_dict"]

    new_state_dict = OrderedDict()

    for key, value in state_dict.items():
        if key.startswith("model.san"):
            new_key = key.replace("model.san", "model.attention")
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value

    checkpoint["state_dict"] = new_state_dict
    torch.save(checkpoint, new_path)
