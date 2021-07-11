import os

import numpy as np
from typing import List, Dict

from nltk import TreebankWordTokenizer
from torch.utils.data import DataLoader

from stud.dataset import ABSADataset
from stud.extra import PlABSACategoryModel
from stud.pl_models import PlABSAModel
from stud.utils import load_pickle, pad_collate, compute_pretrained_embeddings
from model import Model
import random

os.environ["TRANSFORMERS_CACHE"] = "../../model/"


def build_model_b(device: str) -> Model:
    """
    The implementation of this function is MANDATORY.
    Args:
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements aspect sentiment analysis of the ABSA pipeline.
            b: Aspect sentiment analysis.
    """
    return StudentModel(device)


def build_model_ab(device: str) -> Model:
    """
    The implementation of this function is MANDATORY.
    Args:
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements both aspect identification and sentiment analysis of the ABSA pipeline.
            a: Aspect identification.
            b: Aspect sentiment analysis.

    """
    return StudentModel(device)


def build_model_cd(device: str) -> Model:
    """
    The implementation of this function is OPTIONAL.
    Args:
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements both aspect identification and sentiment analysis of the ABSA pipeline
        as well as Category identification and sentiment analysis.
            c: Category identification.
            d: Category sentiment analysis.
    """
    # return RandomBaseline(mode='cd')
    return StudentModelExtra(device)


class RandomBaseline(Model):

    options_sent = [
        ("positive", 793 + 1794),
        ("negative", 701 + 638),
        ("neutral", 365 + 507),
        ("conflict", 39 + 72),
    ]

    options = [
        (0, 452),
        (1, 1597),
        (2, 821),
        (3, 524),
    ]

    options_cat_n = [
        (1, 2027),
        (2, 402),
        (3, 65),
        (4, 6),
    ]

    options_sent_cat = [
        ("positive", 1801),
        ("negative", 672),
        ("neutral", 411),
        ("conflict", 164),
    ]

    options_cat = [
        ("anecdotes/miscellaneous", 939),
        ("price", 268),
        ("food", 1008),
        ("ambience", 355),
    ]

    def __init__(self, mode="b"):

        self._options_sent = [option[0] for option in self.options_sent]
        self._weights_sent = np.array([option[1] for option in self.options_sent])
        self._weights_sent = self._weights_sent / self._weights_sent.sum()

        if mode == "ab":
            self._options = [option[0] for option in self.options]
            self._weights = np.array([option[1] for option in self.options])
            self._weights = self._weights / self._weights.sum()
        elif mode == "cd":
            self._options_cat_n = [option[0] for option in self.options_cat_n]
            self._weights_cat_n = np.array([option[1] for option in self.options_cat_n])
            self._weights_cat_n = self._weights_cat_n / self._weights_cat_n.sum()

            self._options_sent_cat = [option[0] for option in self.options_sent_cat]
            self._weights_sent_cat = np.array(
                [option[1] for option in self.options_sent_cat]
            )
            self._weights_sent_cat = (
                self._weights_sent_cat / self._weights_sent_cat.sum()
            )

            self._options_cat = [option[0] for option in self.options_cat]
            self._weights_cat = np.array([option[1] for option in self.options_cat])
            self._weights_cat = self._weights_cat / self._weights_cat.sum()

        self.mode = mode

    def predict(self, samples: List[Dict]) -> List[Dict]:
        preds = []
        for sample in samples:
            pred_sample = {}
            words = None
            if self.mode == "ab":
                n_preds = np.random.choice(self._options, 1, p=self._weights)[0]
                if n_preds > 0 and len(sample["text"].split(" ")) > n_preds:
                    words = random.sample(sample["text"].split(" "), n_preds)
                elif n_preds > 0:
                    words = sample["text"].split(" ")
            elif self.mode == "b":
                if len(sample["targets"]) > 0:
                    words = [word[1] for word in sample["targets"]]
            if words:
                pred_sample["targets"] = [
                    (
                        word,
                        str(
                            np.random.choice(
                                self._options_sent, 1, p=self._weights_sent
                            )[0]
                        ),
                    )
                    for word in words
                ]
            else:
                pred_sample["targets"] = []
            if self.mode == "cd":
                n_preds = np.random.choice(
                    self._options_cat_n, 1, p=self._weights_cat_n
                )[0]
                pred_sample["categories"] = []
                for i in range(n_preds):
                    category = str(
                        np.random.choice(self._options_cat, 1, p=self._weights_cat)[0]
                    )
                    sentiment = str(
                        np.random.choice(
                            self._options_sent_cat, 1, p=self._weights_sent_cat
                        )[0]
                    )
                    pred_sample["categories"].append((category, sentiment))
            preds.append(pred_sample)
        return preds


class StudentModel(Model):
    def __init__(self, device):
        path = "model/M3.ckpt"
        vocabulary_path = "model/vocabulary.pkl"
        sentiments_vocabulary_path = "model/sentiments_vocabulary.pkl"
        pos_vocabulary_path = "model/pos_vocabulary.pkl"
        embeddings_path = "model/glove.6B.300d.txt"
        cache_path = "model/.vector_cache/"
        self.vocabulary = load_pickle(vocabulary_path)
        self.sentiments_vocabulary = load_pickle(sentiments_vocabulary_path)
        self.pos_vocabulary = load_pickle(pos_vocabulary_path)
        self.vocabularies = {
            "vocabulary": self.vocabulary,
            "sentiments_vocabulary": self.sentiments_vocabulary,
            "pos_vocabulary": self.pos_vocabulary,
        }
        self.tagging_schema = "IOB"
        self.tokenizer = TreebankWordTokenizer()
        pretrained_embeddings = compute_pretrained_embeddings(
            path=embeddings_path,
            cache=cache_path,
            vocabulary=self.vocabulary,
        )

        self.model = PlABSAModel.load_from_checkpoint(
            path,
            map_location=device,
            vocabularies=self.vocabularies,
            embeddings=pretrained_embeddings,
            tokenizer=TreebankWordTokenizer(),
            train=False,
        )
        self.model.eval()

    def predict(self, samples: List[Dict]) -> List[Dict]:
        data = ABSADataset(
            samples,
            vocabularies=self.vocabularies,
            tagging_schema=self.tagging_schema,
            tokenizer=self.tokenizer,
            save_categories=False,
            train=False,
        )
        loader = DataLoader(data, batch_size=1, shuffle=False, collate_fn=pad_collate)
        all_outputs = []
        for batch in loader:
            output = self.model.predict(batch)
            all_outputs += output
        return all_outputs


class StudentModelExtra(Model):
    def __init__(self, device):
        path = "model/M3_cd.ckpt"
        vocabulary_path = "model/vocabulary_extra.pkl"
        sentiments_vocabulary_path = "model/sentiments_vocabulary_extra.pkl"
        pos_vocabulary_path = "model/pos_vocabulary_extra.pkl"
        category_vocabulary_path = "model/categories_vocabulary.pkl"
        category_polarities_vocabulary_path = (
            "model/categories_polarities_vocabulary.pkl"
        )

        embeddings_path = "model/glove.6B.300d.txt"
        cache_path = "model/.vector_cache/"
        self.vocabulary = load_pickle(vocabulary_path)
        self.sentiments_vocabulary = load_pickle(sentiments_vocabulary_path)
        self.pos_vocabulary = load_pickle(pos_vocabulary_path)
        self.cat_vocab = load_pickle(category_vocabulary_path)
        self.cat_pol_vocab = load_pickle(category_polarities_vocabulary_path)
        self.vocabularies = {
            "vocabulary": self.vocabulary,
            "sentiments_vocabulary": self.sentiments_vocabulary,
            "pos_vocabulary": self.pos_vocabulary,
            "category_vocabulary": self.cat_vocab,
            "category_polarity_vocabulary": self.cat_pol_vocab,
        }
        self.tagging_schema = "IOB"
        self.tokenizer = TreebankWordTokenizer()
        pretrained_embeddings = compute_pretrained_embeddings(
            path=embeddings_path,
            cache=cache_path,
            vocabulary=self.vocabulary,
        )

        self.model = PlABSACategoryModel.load_from_checkpoint(
            path,
            map_location=device,
            vocabularies=self.vocabularies,
            embeddings=pretrained_embeddings,
            tokenizer=TreebankWordTokenizer(),
            train=False,
        )
        self.model.eval()

    def predict(self, samples: List[Dict]) -> List[Dict]:
        data = ABSADataset(
            samples,
            vocabularies=self.vocabularies,
            tagging_schema=self.tagging_schema,
            tokenizer=self.tokenizer,
            save_categories=True,
            train=False,
        )
        loader = DataLoader(data, batch_size=1, shuffle=False, collate_fn=pad_collate)
        all_outputs = []
        for batch in loader:
            output = self.model.predict(batch)
            all_outputs += output
        return all_outputs
