import re
import nltk
import json
import torch
import pytorch_lightning as pl
from collections import Counter

from nltk import TreebankWordTokenizer
from tqdm import tqdm
from typing import List, Tuple, Dict, Any, Optional, Union
from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import Vocab
from stud.utils import pad_collate
from transformers import BertTokenizer

# set up tokenizers
nltk.download("punkt")


def tokens_position(
    sentence: str,
    target_char_positions: List[int],
    tokenizer: Union[TreebankWordTokenizer, BertTokenizer],
) -> List[int]:
    """Extract tokens positions from position in string."""
    s_pos, e_pos = target_char_positions
    tokens_between_positions = tokenizer.tokenize(sentence[s_pos:e_pos])
    n_tokens = len(tokens_between_positions)
    s_token = len(tokenizer.tokenize(sentence[:s_pos]))
    return list(range(s_token, s_token + n_tokens))


def read_data(path: str) -> List[Dict[str, Any]]:
    """Read data from json file."""
    with open(path) as f:
        raw_data = json.load(f)
    return raw_data


def json_to_iob(
    raw_data,
    tokenizer: Union[TreebankWordTokenizer, BertTokenizer],
    train: bool = True,
):
    processed_data = {"sentences": [], "targets": []}
    for d in raw_data:
        # extract tokens
        text = d["text"]
        tokens = tokenizer.tokenize(text)
        processed_data["sentences"].append(tokens)
        if train:
            sentiments = ["O"] * len(tokens)
            for start_end, instance, sentiment in d["targets"]:
                sentiment_positions = tokens_position(
                    text, start_end, tokenizer=tokenizer
                )
                for i, s in enumerate(sentiment_positions):
                    if i == 0:
                        sentiments[s] = "B-" + sentiment
                    else:
                        sentiments[s] = "I-" + sentiment
            processed_data["targets"].append(sentiments)

    return processed_data


def json_to_bioes(
    raw_data,
    tokenizer: Union[TreebankWordTokenizer, BertTokenizer],
    train: bool = True,
):
    processed_data = {"sentences": [], "targets": []}
    for d in raw_data:
        # extract tokens
        text = d["text"]
        tokens = tokenizer.tokenize(text)
        processed_data["sentences"].append(tokens)
        if train:
            sentiments = ["O"] * len(tokens)
            for start_end, instance, sentiment in d["targets"]:
                sentiment_positions = tokens_position(
                    text, start_end, tokenizer=tokenizer
                )
                for i, s in enumerate(sentiment_positions):
                    if len(sentiment_positions) == 1:
                        sentiments[s] = "S-" + sentiment
                    elif i == 0:
                        sentiments[s] = "B-" + sentiment
                    elif i == len(sentiment_positions) - 1:
                        sentiments[s] = "E-" + sentiment
                    else:
                        sentiments[s] = "I-" + sentiment
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
        raw_data: List[Dict[str, Any]],
        vocabulary: Vocab,
        sentiments_vocabulary: Vocab,
        tokenizer: Union[TreebankWordTokenizer, BertTokenizer],
        tagging_schema: str,
        train: bool = True,
    ) -> None:
        super(ABSADataset, self).__init__()
        self.raw_data = raw_data
        self.tokenizer = tokenizer
        self.train = train
        preprocess = json_to_iob if tagging_schema == "IOB" else json_to_bioes
        processed_data = preprocess(raw_data, tokenizer=tokenizer, train=train)
        self.sentences = processed_data["sentences"]
        self.targets = processed_data["targets"]
        self.vocabulary = vocabulary
        self.sentiments_vocabulary = sentiments_vocabulary
        self.encoded_data = []
        self.index_dataset()

    def encode_text(self, sentence: List[str]) -> List[int]:
        if isinstance(self.tokenizer, BertTokenizer):
            indices = self.tokenizer.convert_tokens_to_ids(sentence)
        else:
            indices = []
            for w in sentence:
                if w in self.vocabulary.stoi:
                    indices.append(self.vocabulary[w])
                else:
                    indices.append(self.vocabulary.unk_index)
        return indices

    def index_dataset(self):
        for i in range(len(self.sentences)):
            data_dict = {}
            sentence = self.sentences[i]
            data_dict["raw"] = self.raw_data[i]
            data_dict["inputs"] = torch.LongTensor(self.encode_text(sentence))
            if self.train:
                targets = self.targets[i]
                data_dict["outputs"] = torch.LongTensor(
                    [self.sentiments_vocabulary[t] for t in targets]
                )

            self.encoded_data.append(data_dict)

    def __len__(self) -> int:
        return len(self.encoded_data)

    def __getitem__(self, index: int) -> Dict[str, torch.LongTensor]:
        return self.encoded_data[index]


class DataModuleABSA(pl.LightningDataModule):
    def __init__(
        self,
        train_data,
        dev_data,
        vocabulary,
        sentiments_vocabulary,
        tagging_schema: str,
        tokenizer=None,
    ):

        super(DataModuleABSA, self).__init__()
        self.train_data = train_data
        self.dev_data = dev_data
        self.vocabulary = vocabulary
        self.sentiments_vocabulary = sentiments_vocabulary
        self.tagging_schema = tagging_schema
        self.tokenizer = tokenizer

    def setup(self, stage=None):
        self.trainset = ABSADataset(
            self.train_data,
            self.vocabulary,
            self.sentiments_vocabulary,
            tagging_schema=self.tagging_schema,
            tokenizer=self.tokenizer,
        )
        self.devset = ABSADataset(
            self.dev_data,
            self.vocabulary,
            self.sentiments_vocabulary,
            tagging_schema=self.tagging_schema,
            tokenizer=self.tokenizer,
        )

    def train_dataloader(self):
        return DataLoader(
            self.trainset, batch_size=16, shuffle=True, collate_fn=pad_collate
        )

    def val_dataloader(self):
        return DataLoader(
            self.devset, batch_size=16, shuffle=False, collate_fn=pad_collate
        )
