import torch
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import Vocab, Vectors
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
    # pad token indexes
    token_indexes = [x["token_indexes"] for x in batch]
    token_indexes_padded = pad_sequence(
        token_indexes, batch_first=True, padding_value=0
    )

    # lenghts and attention mask
    lengths = [len(x) for x in token_indexes]
    attention_mask = token_indexes_padded != 0

    # pad labels (except in case we're doing inference)
    try:
        labels = [x["labels"] for x in batch]
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=0)
    except:
        labels_padded = None

    # pad pos tags
    pos_tags = [x["pos_tags"] for x in batch]
    pos_tags_padded = pad_sequence(pos_tags, batch_first=True, padding_value=0)

    try:
        categories = torch.stack([x["categories"] for x in batch])
    except:
        categories = None

    return {
        "token_indexes": token_indexes_padded,
        "labels": labels_padded,
        "categories": categories,
        "pos_tags": pos_tags_padded,
        "lengths": lengths,
        "attention_mask": attention_mask,
        "tokens": [x["tokens"] for x in batch],
        "raw": [x["raw"] for x in batch],
    }


def compute_pretrained_embeddings(path: str, cache: str, vocabulary: Vocab):
    vectors = Vectors(path, cache=cache)
    pretrained_embeddings = torch.randn(len(vocabulary), vectors.dim)
    for i, w in enumerate(vocabulary.itos):
        if w in vectors.stoi:
            vec = vectors.get_vecs_by_tokens(w)
            pretrained_embeddings[i] = vec
    pretrained_embeddings[vocabulary["<pad>"]] = torch.zeros(vectors.dim)
    return pretrained_embeddings
