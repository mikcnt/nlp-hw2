import re
import nltk
import json
import torch
from collections import Counter
from tqdm import tqdm
from typing import List, Tuple, Any
from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset
from torchtext.vocab import Vocab

nltk.download("punkt")

sentiment_to_idx = {
    sentiment: idx
    for idx, sentiment in enumerate(["positive", "negative", "neutral", "conflict"])
}

idx_to_sentiment = {idx: sentiment for sentiment, idx in sentiment_to_idx.items()}


def idx_to_one_hot(idx: int, num_classes: int) -> List[int]:
    one_hot = [0] * num_classes
    one_hot[idx] = 1
    return one_hot


def tokens_position(sentence: str, target_char_positions: List[int]) -> List[int]:
    """Extract tokens positions from position in string."""
    s_pos, e_pos = target_char_positions
    n_tokens = len(word_tokenize(sentence[s_pos:e_pos]))
    s_token = len(re.findall(r" +", sentence[:s_pos]))
    return list(range(s_token, s_token + n_tokens))


def read_data(path: str) -> Tuple[list, list, list]:
    """Read data from json file."""
    with open(path) as f:
        raw_data = json.load(f)
    return raw_data


def preprocess(raw_data):
    # split data in 2 + 1 lists (3 for restaurants data, 2 for laptops data):
    # for both datasets: sentences (i.e., raw text), targets (i.e., position range, instance, sentiment),
    # for restaurant dataset: categories (i.e., category, sentiment)
    sentences = []
    targets = []
    categories = []
    for d in raw_data:
        # tokenize text data
        s = d["text"]
        sentences.append(word_tokenize(s))
        # extract targets: can either be 0, 1 or multiple
        t = [
            {
                "tokens_position": tokens_position(s, x[0]),
                # TODO: is the `instance` key necessary?
                "instance": x[1],
                # TODO: is one-hot right?
                "sentiment": idx_to_one_hot(sentiment_to_idx[x[2]]),
            }
            for x in d["targets"]
        ]
        targets.append(t)
        # extract categories. can be 0, 1 or multiple (for the restaurant dataset)
        if "categories" in d.keys():
            c = [{"category": x[0], "sentiment": x[1]} for x in d["categories"]]
            categories.append(c)

    return sentences, targets, categories


def build_vocab(sentences: List[str], min_freq: int = 1) -> Vocab:
    counter = Counter()
    for i in tqdm(range(len(sentences))):
        for token in sentences[i]:
            if token is not None:
                counter[token] += 1
    return Vocab(counter, specials=["<pad>", "<unk>"], min_freq=min_freq)


class ABSADataset(Dataset):
    def __init__(
        self,
        sentences: List[str],
        targets: List[Any],
        categories: List[Any],
        vocabulary: Vocab,
    ) -> None:
        super(ABSADataset, self).__init__()
        self.sentences = sentences
        self.targets = targets
        self.categories = categories
        self.encoded_sentences = self.encode_text(self.sentences, vocabulary)

    @staticmethod
    def encode_text(sentence: List[str], vocabulary: Vocab) -> List[int]:
        indices = []
        for w in sentence:
            # TODO: why would it be `None`?
            if w is None:
                indices.append(vocabulary["<pad>"])
            elif w in vocabulary.stoi:
                indices.append(vocabulary[w])
            else:
                indices.append(vocabulary.unk_index)
        return indices

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, index: int) -> torch.Tensor:
        pass
