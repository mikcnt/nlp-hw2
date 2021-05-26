import numpy as np
from typing import List, Tuple, Dict

import torch
from torch.utils.data import DataLoader
from torchtext.vocab import Vectors

from stud.dataset import preprocess, ABSADataset
from stud.pl_models import PlABSAModel
from stud.utils import load_pickle, pad_collate
from model import Model
import random


def build_model_b(device: str) -> Model:
    """
    The implementation of this function is MANDATORY.
    Args:
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements aspect sentiment analysis of the ABSA pipeline.
            b: Aspect sentiment analysis.
    """
    return RandomBaseline()


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
    # return RandomBaseline(mode="ab")
    # raise NotImplementedError
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
    raise NotImplementedError


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
        path = "model/epoch=19_f1_val=14.8297.ckpt"
        vocabulary_path = "model/vocabulary.pkl"
        sentiments_vocabulary_path = "model/sentiments_vocabulary.pkl"
        embeddings_path = "model/glove.6B.300d.txt"
        cache_path = "model/.vector_cache/"
        self.vocabulary = load_pickle(vocabulary_path)

        self.sentiments_vocabulary = load_pickle(sentiments_vocabulary_path)
        vocabularies = {
            "vocabulary": self.vocabulary,
            "sentiments_vocabulary": self.sentiments_vocabulary,
        }
        vectors = Vectors(embeddings_path, cache=cache_path)
        pretrained_embeddings = torch.randn(len(self.vocabulary), vectors.dim)
        for i, w in enumerate(self.vocabulary.itos):
            if w in vectors.stoi:
                vec = vectors.get_vecs_by_tokens(w)
                pretrained_embeddings[i] = vec
        pretrained_embeddings[self.vocabulary["<pad>"]] = torch.zeros(vectors.dim)
        self.model = PlABSAModel.load_from_checkpoint(
            path,
            map_location=device,
            vocabularies=vocabularies,
            embeddings=pretrained_embeddings,
        )
        self.model.eval()

    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary

    def predict(self, samples: List[Dict]) -> List[Dict]:
        """
        --> !!! STUDENT: implement here your predict function !!! <--
        Args:
            - If you are doing model_b (ie. aspect sentiment analysis):
                sentence: a dictionary that represents an input sentence as well as the target words (aspects), for example:
                    [
                        {
                            "text": "I love their pasta but I hate their Ananas Pizza.",
                            "targets": [[13, 17], "pasta"], [[36, 47], "Ananas Pizza"]]
                        },
                        {
                            "text": "The people there is so kind and the taste is exceptional, I'll come back for sure!"
                            "targets": [[4, 9], "people", [[36, 40], "taste"]]
                        }
                    ]
            - If you are doing model_ab or model_cd:
                sentence: a dictionary that represents an input sentence, for example:
                    [
                        {
                            "text": "I love their pasta but I hate their Ananas Pizza."
                        },
                        {
                            "text": "The people there is so kind and the taste is exceptional, I'll come back for sure!"
                        }
                    ]
        Returns:
            A List of dictionaries with your predictions:
                - If you are doing target word identification + target polarity classification:
                    [
                        {
                            "targets": [("pasta", "positive"), ("Ananas Pizza", "negative")] # A list having a tuple for each target word
                        },
                        {
                            "targets": [("people", "positive"), ("taste", "positive")] # A list having a tuple for each target word
                        }
                    ]
                - If you are doing target word identification + target polarity classification + aspect category identification + aspect category polarity classification:
                    [
                        {
                            "targets": [("pasta", "positive"), ("Ananas Pizza", "negative")], # A list having a tuple for each target word
                            "categories": [("food", "conflict")]
                        },
                        {
                            "targets": [("people", "positive"), ("taste", "positive")], # A list having a tuple for each target word
                            "categories": [("service", "positive"), ("food", "positive")]
                        }
                    ]
        """
        # data = preprocess(samples, preprocess_targets=False)
        data = ABSADataset(
            samples,
            self.vocabulary,
            self.sentiments_vocabulary,
            preprocess_targets=False,
        )
        loader = DataLoader(data, batch_size=1, shuffle=False, collate_fn=pad_collate)
        all_outputs = []
        for batch in loader:
            output = self.model.forward_processed(batch)
            all_outputs += output
        return all_outputs
