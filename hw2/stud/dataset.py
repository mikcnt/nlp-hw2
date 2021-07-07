import nltk
import json
import torch
import pytorch_lightning as pl
from collections import Counter

from nltk import TreebankWordTokenizer
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Tuple
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import Vocab

from stud.utils import pad_collate

# download nltk stuff
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")


def read_data(path: str) -> List[Dict[str, Any]]:
    """Read data from json file."""
    with open(path) as f:
        raw_data = json.load(f)
    return raw_data


def zero_dict() -> Dict[str, Dict[str, float]]:
    return {
        "anecdotes/miscellaneous": {
            "positive": 0.0,
            "negative": 0.0,
            "neutral": 0.0,
            "conflict": 0.0,
        },
        "price": {"positive": 0.0, "negative": 0.0, "neutral": 0.0, "conflict": 0.0},
        "food": {"positive": 0.0, "negative": 0.0, "neutral": 0.0, "conflict": 0.0},
        "ambience": {"positive": 0.0, "negative": 0.0, "neutral": 0.0, "conflict": 0.0},
        "service": {"positive": 0.0, "negative": 0.0, "neutral": 0.0, "conflict": 0.0},
    }


def preprocess_category(d: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    category_dict = zero_dict()
    for category, polarity in d["categories"]:
        category_dict[category][polarity] = 1
    return category_dict


def tokens_position(
    sentence: str,
    target_char_positions: List[int],
    tokenizer: TreebankWordTokenizer,
) -> List[int]:
    """Extract tokens with sentiments associated positions from position in string."""
    s_pos, e_pos = target_char_positions
    tokens_between_positions = tokenizer.tokenize(sentence[s_pos:e_pos])
    n_tokens = len(tokens_between_positions)
    s_token = len(tokenizer.tokenize(sentence[:s_pos]))
    return list(range(s_token, s_token + n_tokens))


def json_to_tags(example, tokenizer: TreebankWordTokenizer, tagging_schema: str):
    assert tagging_schema in ["IOB", "BIOES"], "Schema must be either `IOB` or `BIOES`."

    text = example["text"]
    targets = example["targets"]
    tokens = tokenizer.tokenize(text)

    sentiments = ["O"] * len(tokens)
    for start_end, instance, sentiment in targets:
        sentiment_positions = tokens_position(text, start_end, tokenizer=tokenizer)
        for i, s in enumerate(sentiment_positions):
            if tagging_schema == "IOB":
                if i == 0:
                    sentiments[s] = "B-" + sentiment
                else:
                    sentiments[s] = "I-" + sentiment
            elif tagging_schema == "BIOES":
                if len(sentiment_positions) == 1:
                    sentiments[s] = "S-" + sentiment
                elif i == 0:
                    sentiments[s] = "B-" + sentiment
                elif i == len(sentiment_positions) - 1:
                    sentiments[s] = "E-" + sentiment
                else:
                    sentiments[s] = "I-" + sentiment
    return sentiments


def preprocess(
    raw_data: List[Dict[str, Any]],
    tokenizer: TreebankWordTokenizer,
    tagging_schema: Optional[str] = None,
    save_categories: bool = False,
    train: bool = True,
) -> Dict[str, Any]:
    processed_data = {
        "sentences": [],
        "targets": [],
        "pos_tags": [],
    }
    if save_categories:
        processed_data["categories"] = []
    for d in raw_data:
        text = d["text"]
        tokens = tokenizer.tokenize(text)
        pos_tags = [pos[1] for pos in nltk.pos_tag(tokens)]

        processed_data["sentences"].append(tokens)
        processed_data["pos_tags"].append(pos_tags)

        # only valid for restaurants
        if save_categories:
            processed_data["categories"].append(preprocess_category(d))

        if train:
            sentiments = json_to_tags(d, tokenizer, tagging_schema)
            processed_data["targets"].append(sentiments)

    return processed_data


def tags_to_json(tokens, sentiments, tagging_schema):
    tokens2sentiments = []
    for i, (token, sentiment) in enumerate(zip(tokens, sentiments)):
        if tagging_schema == "IOB":
            # just ignore outside sentiments for the moment
            if sentiment == "O":
                tokens2sentiments.append([[token], sentiment])

            # if it is starting, then we expect something after, so we append the first one
            if sentiment.startswith("B-"):
                tokens2sentiments.append([[token], sentiment[2:]])

            # if it is inside, we have different options
            if sentiment.startswith("I-"):
                # if this is the first sentiment, then we just treat it as a beginning one
                if len(tokens2sentiments) == 0:
                    tokens2sentiments.append([[token], sentiment[2:]])
                else:
                    # otherwise, there is some other sentiment before
                    last_token, last_sentiment = tokens2sentiments[-1]
                    # if the last sentiment is not equal to the one we're considering, then we treat this
                    # again as a beginning one.
                    if last_sentiment != sentiment[2:]:
                        tokens2sentiments.append([[token], sentiment[2:]])
                    # otherwise, the sentiment before was a B or a I with the same sentiment
                    # therefore this token is part of the same target instance, with the same sentiment associated
                    else:
                        tokens2sentiments[-1] = [
                            last_token + [token],
                            sentiment[2:],
                        ]
        elif tagging_schema == "BIOES":
            # just ignore outside sentiments for the moment
            if sentiment == "O":
                tokens2sentiments.append([[token], sentiment])

            # if it is single, then we just need to append that
            if sentiment.startswith("S-"):
                tokens2sentiments.append([[token], sentiment[2:]])

            # if it is starting, then we expect something after, so we append the first one
            if sentiment.startswith("B-"):
                tokens2sentiments.append([[token], sentiment[2:]])

            # if it is inside, we have different options
            if sentiment.startswith("I-") or sentiment.startswith("E-"):
                # if this is the first sentiment, then we just treat it as a beginning one
                if len(tokens2sentiments) == 0:
                    tokens2sentiments.append([[token], sentiment[2:]])
                else:
                    # otherwise, there is some other sentiment before
                    last_token, last_sentiment = tokens2sentiments[-1]
                    # if the last sentiment is not equal to the one we're considering, then we treat this
                    # again as a beginning one.
                    if last_sentiment != sentiment[2:]:
                        tokens2sentiments.append([[token], sentiment[2:]])
                    # if the previous sentiment was a single target word or an ending one
                    # we treat the one we're considering again as a beginning one
                    elif sentiments[i - 1].startswith("S-") or sentiments[
                        i - 1
                    ].startswith("E-"):
                        tokens2sentiments.append([[token], sentiment[2:]])
                    # otherwise, the sentiment before was a B or a I with the same sentiment
                    # therefore this token is part of the same target instance, with the same sentiment associated
                    else:
                        tokens2sentiments[-1] = [
                            last_token + [token],
                            sentiment[2:],
                        ]

    return tokens2sentiments


def build_vocab(data: List[str], specials: List[str], min_freq: int = 1) -> Vocab:
    counter = Counter()
    for i in tqdm(range(len(data))):
        for t in data[i]:
            if t is not None:
                counter[t] += 1
    return Vocab(counter, specials=specials, min_freq=min_freq)


def build_vocab_category(data: List[Dict[str, Any]]) -> Tuple[Vocab, Vocab]:
    """Use with raw data, only on restaurants dataset."""
    counter_category = Counter()
    counter_polarity = Counter()
    for d in data:
        for category, polarity in d["categories"]:
            counter_category[category] += 1
            counter_polarity[polarity] += 1
    from torchtext.vocab import Vocab

    vocab_category = Vocab(counter_category, specials=())
    vocab_polarity = Vocab(counter_polarity, specials=())
    return vocab_category, vocab_polarity


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
        vocabularies: Dict[str, Vocab],
        tagging_schema: str,
        tokenizer: TreebankWordTokenizer,
        save_categories: bool = False,
        train: bool = True,
    ) -> None:
        super(ABSADataset, self).__init__()
        self.raw_data = raw_data
        self.tokenizer = tokenizer
        self.save_categories = save_categories
        self.train = train

        preprocessed_data = preprocess(
            raw_data,
            tokenizer=tokenizer,
            tagging_schema=tagging_schema,
            save_categories=save_categories,
            train=train,
        )

        self.sentences = preprocessed_data["sentences"]
        self.targets = preprocessed_data["targets"]
        self.pos_tags = preprocessed_data["pos_tags"]

        if save_categories:
            self.categories = preprocessed_data["categories"]
            self.category_vocabulary = vocabularies["category_vocabulary"]
            self.category_polarity_vocabulary = vocabularies[
                "category_polarity_vocabulary"
            ]

        self.vocabulary = vocabularies["vocabulary"]
        self.sentiments_vocabulary = vocabularies["sentiments_vocabulary"]
        self.pos_vocabulary = vocabularies["pos_vocabulary"]

        self.encoded_data = []
        self.index_dataset()

    def encode_text(self, sentence: List[str], vocab) -> List[int]:
        indices = []
        for w in sentence:
            if w in vocab.stoi:
                indices.append(vocab[w])
            else:
                indices.append(vocab.unk_index)
        return indices

    def index_dataset(self):
        for i in range(len(self.sentences)):
            data_dict = {}
            sentence = self.sentences[i]
            pos_tags = self.pos_tags[i]
            data_dict["tokens"] = sentence
            data_dict["raw"] = self.raw_data[i]
            data_dict["token_indexes"] = torch.LongTensor(
                self.encode_text(sentence, self.vocabulary)
            )
            data_dict["pos_tags"] = torch.LongTensor(
                self.encode_text(pos_tags, self.pos_vocabulary)
            )

            if self.save_categories:
                categories = self.categories[i]
                # initialize 0-tensor
                zeros = torch.zeros(
                    len(self.category_vocabulary),
                    len(self.category_polarity_vocabulary) + 1,
                )

                # place 1 in the polarity of a given category
                for category, polarities in categories.items():
                    for polarity, value in polarities.items():
                        zeros[
                            self.category_vocabulary[category],
                            self.category_polarity_vocabulary[polarity],
                        ] = value

                # if a certain category has no polarity, place 1 in the last polarity
                # (last place = no polarity)
                for row in range(zeros.shape[0]):
                    if sum(zeros[row]) == 0:
                        zeros[row, -1] = 1

                data_dict["categories"] = zeros

            if self.train:
                targets = self.targets[i]
                data_dict["labels"] = torch.LongTensor(
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
        vocabularies,
        tagging_schema: str,
        batch_size: int,
        tokenizer=None,
        save_categories: bool = False,
    ):

        super(DataModuleABSA, self).__init__()
        self.train_data = train_data
        self.dev_data = dev_data
        self.batch_size = batch_size
        self.vocabularies = vocabularies
        self.tagging_schema = tagging_schema
        self.tokenizer = tokenizer
        self.save_categories = save_categories

    def setup(self, stage=None):
        self.trainset = ABSADataset(
            self.train_data,
            self.vocabularies,
            tagging_schema=self.tagging_schema,
            tokenizer=self.tokenizer,
            save_categories=self.save_categories,
        )
        self.devset = ABSADataset(
            self.dev_data,
            self.vocabularies,
            tagging_schema=self.tagging_schema,
            tokenizer=self.tokenizer,
            save_categories=self.save_categories,
        )

    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=pad_collate,
            num_workers=8,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.devset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=pad_collate,
            num_workers=8,
            pin_memory=True,
        )
