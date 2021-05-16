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

def token_position(sentence, position)


def read_data(path: str) -> Tuple[list, list, list]:
    # read from json file
    with open(path) as f:
        raw_data = json.load(f)
    # split data in 2 + 1 lists (3 for restaurants data, 2 for laptops data):
    # for both datasets: sentences (i.e., raw text), targets (i.e., position range, instance, sentiment),
    # for restaurant dataset: categories (i.e., category, sentiment)
    sentences = []
    targets = []
    categories = []
    for d in raw_data:
        # tokenize text data
        sentences.append(word_tokenize(d["text"]))
        # extract targets. can be 0, 1 or multiple
        t = [
            {"position": tuple(x[0]), "instance": x[1], "sentiment": x[2]}
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
