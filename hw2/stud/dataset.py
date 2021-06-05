import nltk
import json
import torch
import pytorch_lightning as pl
from collections import Counter

from nltk import TreebankWordTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from tqdm import tqdm
from typing import List, Dict, Any, Union, Optional
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import Vocab

from stud.bert_embedder import BERTEmbedder
from stud.utils import pad_collate
from transformers import BertTokenizer, BertModel, BertConfig

# set up tokenizers
nltk.download("punkt")


def read_data(path: str) -> List[Dict[str, Any]]:
    """Read data from json file."""
    with open(path) as f:
        raw_data = json.load(f)
    return raw_data


def tokens_position(
    sentence: str,
    target_char_positions: List[int],
    tokenizer: Union[TreebankWordTokenizer, BertTokenizer],
) -> List[int]:
    """Extract tokens with sentiments associated positions from position in string."""
    s_pos, e_pos = target_char_positions
    tokens_between_positions = tokenizer.tokenize(sentence[s_pos:e_pos])
    n_tokens = len(tokens_between_positions)
    s_token = len(tokenizer.tokenize(sentence[:s_pos]))
    return list(range(s_token, s_token + n_tokens))


def json_to_tags(
    example, tokenizer: Union[TreebankWordTokenizer, BertTokenizer], tagging_schema: str
):
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
    tokenizer: Union[TreebankWordTokenizer, BertTokenizer],
    tagging_schema: Optional[str] = None,
    bert_embedder=None,
    train: bool = True,
):
    """Convert data in JSON format to IOB schema."""
    processed_data = {"sentences": [], "targets": [], "bert_embeddings": []}
    for d in raw_data:
        text = d["text"]
        tokens = tokenizer.tokenize(text)
        if bert_embedder is not None:
            bert_embeddings = bert_embedder.embed_sentences([tokens])[0].to("cpu")
            processed_data["bert_embeddings"].append(bert_embeddings)

        processed_data["sentences"].append(tokens)
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
        tagging_schema: str,
        tokenizer: Union[TreebankWordTokenizer, BertTokenizer],
        use_bert: bool = True,
        train: bool = True,
    ) -> None:
        super(ABSADataset, self).__init__()
        self.raw_data = raw_data
        self.tokenizer = tokenizer
        self.use_bert = use_bert
        if use_bert:
            bert_config = BertConfig.from_pretrained(
                "bert-base-cased", output_hidden_states=True
            )
            bert_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
            bert_model = BertModel.from_pretrained(
                "bert-base-cased", config=bert_config
            )
            bert_embedder = BERTEmbedder(
                bert_model=bert_model,
                bert_tokenizer=bert_tokenizer,
                device="cuda",
            )
        else:
            bert_embedder = None
        self.train = train
        preprocessed_data = preprocess(
            raw_data,
            tokenizer=tokenizer,
            tagging_schema=tagging_schema,
            bert_embedder=bert_embedder,
            train=train,
        )
        self.sentences = preprocessed_data["sentences"]
        self.targets = preprocessed_data["targets"]
        self.bert_embeddings = preprocessed_data["bert_embeddings"]
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

            if self.use_bert:
                data_dict["bert_embeddings"] = self.bert_embeddings[i]

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
        batch_size: int,
        tokenizer=None,
    ):

        super(DataModuleABSA, self).__init__()
        self.train_data = train_data
        self.dev_data = dev_data
        self.batch_size = batch_size
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
