import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import Vocab
from typing import *
import pickle


def save_pickle(data: Any, path: str) -> None:
    """Save object as pickle file."""
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_pickle(path: str) -> Any:
    """Load object from pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)


def pad_collate(batch):
    # pad x
    xx = [x["inputs"] for x in batch]
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)

    # pad y
    try:
        yy = [x["outputs"] for x in batch]
        yy_pad = pad_sequence(yy, batch_first=True, padding_value=0)
    except:
        yy_pad = None

    # lengths of inputs == lenghts of outputs
    lengths = [len(x) for x in xx]

    batch = {
        "inputs": xx_pad,
        "outputs": yy_pad,
        "lengths": lengths,
        "raw": [x["raw"] for x in batch],
    }

    return batch
