import re
import nltk
import json
import torch
import pytorch_lightning as pl
from collections import Counter
from tqdm import tqdm
from typing import List, Tuple, Dict
from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import Vocab

from hw2.stud.utils import pad_collate

nltk.download("punkt")


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
    processed_data = {"sentences": [], "targets": []}
    # TODO: categories
    for d in raw_data:
        # extract tokens
        text = d["text"]
        tokens = word_tokenize(text)
        # possible sentiments are: positive, negative, neutral, conflict
        # `B-sentiment` means that it's the starting token of a sequence (possibly of length 1)
        # `I-sentiment` means that it's following another token (either B- or another I-) of a sequence
        # `0` means that no sentiment is involved with that the token
        sentiments = ["0"] * len(tokens)
        for start_end, instance, sentiment in d["targets"]:
            sentiment_positions = tokens_position(text, start_end)
            for i, s in enumerate(sentiment_positions):
                if i == 0:
                    sentiments[s] = "B-" + sentiment
                else:
                    sentiments[s] = "I-" + sentiment

        processed_data["sentences"].append(tokens)
        processed_data["targets"].append(sentiments)

    return processed_data


def build_vocab(data: List[str], specials: List[str], min_freq: int = 1) -> Vocab:
    counter = Counter()
    for i in tqdm(range(len(data))):
        for t in data[i]:
            if t is not None:
                counter[t] += 1
    return Vocab(counter, specials=specials, min_freq=min_freq)


def dataset_max_len(sentences: List[List[str]]) -> int:
    max_len = 0
    for s in sentences:
        length = len(s)
        if length > max_len:
            max_len = length
    return max_len


class ABSADataset(Dataset):
    def __init__(
        self,
        processed_data: Dict[str, List[List[str]]],
        vocabulary: Vocab,
        sentiments_vocabulary: Vocab,
    ) -> None:
        super(ABSADataset, self).__init__()
        self.sentences = processed_data["sentences"]
        self.targets = processed_data["targets"]
        self.encoded_data = []
        self.vocabulary = vocabulary
        self.sentiments_vocabulary = sentiments_vocabulary
        self.max_len = dataset_max_len(self.sentences)
        self.index_dataset()

    def encode_text(self, sentence: List[str]) -> List[int]:
        indices = []
        for w in sentence:
            if w in self.vocabulary.stoi:
                indices.append(self.vocabulary[w])
            else:
                indices.append(self.vocabulary.unk_index)
        return indices

    def index_dataset(self):
        assert len(self.sentences) == len(self.targets)
        for i in range(len(self.sentences)):
            sentence = self.sentences[i]
            targets = self.targets[i]
            # encode sentences and targets
            encoded_elem = self.encode_text(sentence)
            encoded_labels = [self.sentiments_vocabulary[t] for t in targets]

            encoded_elem = torch.LongTensor(encoded_elem)
            encoded_labels = torch.LongTensor(encoded_labels)

            self.encoded_data.append(
                {"inputs": encoded_elem, "outputs": encoded_labels}
            )

    def __len__(self) -> int:
        return len(self.encoded_data)

    def __getitem__(self, index: int) -> Tuple[torch.LongTensor, torch.LongTensor]:
        return self.encoded_data[index]


class DataModuleABSA(pl.LightningDataModule):
    def __init__(
        self,
        train_data,
        dev_data,
        vocabulary,
        sentiments_vocabulary,
    ):

        super().__init__()
        self.train_data = train_data
        self.dev_data = dev_data
        self.vocabulary = vocabulary
        self.sentiments_vocabulary = sentiments_vocabulary

    def setup(self, stage=None):
        self.trainset = ABSADataset(
            self.train_data,
            self.vocabulary,
            self.sentiments_vocabulary,
        )
        self.devset = ABSADataset(
            self.dev_data,
            self.vocabulary,
            self.sentiments_vocabulary,
        )

    def train_dataloader(self):
        return DataLoader(
            self.trainset, batch_size=128, shuffle=True, collate_fn=pad_collate
        )

    def val_dataloader(self):
        return DataLoader(
            self.devset, batch_size=128, shuffle=False, collate_fn=pad_collate
        )
