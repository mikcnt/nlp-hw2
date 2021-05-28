import re
import nltk
import json
import torch
import pytorch_lightning as pl
from collections import Counter
from tqdm import tqdm
from typing import List, Tuple, Dict, Any, Optional
from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import Vocab
from stud.utils import pad_collate
from transformers import BertTokenizer

# set up tokenizers
nltk.download("punkt")
BERT_MODEL_NAME = "bert-base-cased"
bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)


def tokenize(sentence, use_bert=False):
    if use_bert:
        return bert_tokenizer.tokenize(sentence)
    else:
        return word_tokenize(sentence)


def tokens_position(
    sentence: str, target_char_positions: List[int], use_bert=False
) -> List[int]:
    """Extract tokens positions from position in string."""
    s_pos, e_pos = target_char_positions
    n_tokens = len(tokenize(sentence[s_pos:e_pos], use_bert=use_bert))
    s_token = len(re.findall(r" +", sentence[:s_pos]))
    return list(range(s_token, s_token + n_tokens))


def read_data(path: str) -> List[Dict[str, Any]]:
    """Read data from json file."""
    with open(path) as f:
        raw_data = json.load(f)
    return raw_data


def preprocess(raw_data, use_bert=False, preprocess_targets=True):
    # split data in 2 + 1 lists (3 for restaurants data, 2 for laptops data):
    # for both datasets: sentences (i.e., raw text), targets (i.e., position range, instance, sentiment),
    # for restaurant dataset: categories (i.e., category, sentiment)
    processed_data = {"sentences": [], "targets": []}
    # TODO: categories
    for d in raw_data:
        # extract tokens
        text = d["text"]
        tokens = tokenize(text, use_bert=use_bert)
        processed_data["sentences"].append(tokens)
        if preprocess_targets:
            # possible sentiments are: positive, negative, neutral, conflict
            # `B-sentiment` means that it's the starting token of a sequence (possibly of length 1)
            # `I-sentiment` means that it's following another token (either B- or another I-) of a sequence
            # `O` means that no sentiment is involved with that the token
            sentiments = ["O"] * len(tokens)
            for start_end, instance, sentiment in d["targets"]:
                sentiment_positions = tokens_position(text, start_end)
                for i, s in enumerate(sentiment_positions):
                    if i == 0:
                        sentiments[s] = "B-" + sentiment
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
        preprocess_targets: bool = True,
        use_bert: bool = False,
    ) -> None:
        super(ABSADataset, self).__init__()
        self.raw_data = raw_data
        self.preprocess_targets = preprocess_targets
        self.use_bert = use_bert
        processed_data = preprocess(
            raw_data, use_bert=use_bert, preprocess_targets=preprocess_targets
        )
        self.sentences = processed_data["sentences"]
        self.targets = processed_data["targets"]
        self.vocabulary = vocabulary
        self.sentiments_vocabulary = sentiments_vocabulary
        self.encoded_data = []
        self.index_dataset()

    def encode_text(self, sentence: List[str]) -> List[int]:
        if self.use_bert:
            indices = bert_tokenizer.convert_tokens_to_ids(sentence)
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
            if self.preprocess_targets:
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
        use_bert=False,
    ):

        super().__init__()
        self.train_data = train_data
        self.dev_data = dev_data
        self.vocabulary = vocabulary
        self.sentiments_vocabulary = sentiments_vocabulary
        self.use_bert = use_bert

    def setup(self, stage=None):
        self.trainset = ABSADataset(
            self.train_data,
            self.vocabulary,
            self.sentiments_vocabulary,
            use_bert=self.use_bert,
        )
        self.devset = ABSADataset(
            self.dev_data,
            self.vocabulary,
            self.sentiments_vocabulary,
            use_bert=self.use_bert,
        )

    def train_dataloader(self):
        return DataLoader(
            self.trainset, batch_size=16, shuffle=True, collate_fn=pad_collate
        )

    def val_dataloader(self):
        return DataLoader(
            self.devset, batch_size=16, shuffle=False, collate_fn=pad_collate
        )
