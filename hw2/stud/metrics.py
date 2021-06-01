from collections import defaultdict
import torch
from nltk.tokenize.treebank import TreebankWordDetokenizer, TreebankWordTokenizer
from torchmetrics import Metric
from typing import *

from transformers import BertTokenizer


class TokenToSentimentsConverter(object):
    def __init__(
        self,
        vocabulary,
        sentiments_vocabulary,
        tokenizer: Union[TreebankWordTokenizer, BertTokenizer],
    ):
        self.vocabulary = vocabulary
        self.sentiments_vocabulary = sentiments_vocabulary
        self.tokenizer = tokenizer

    def begin_sentiment(
        self, tokens: List[Tuple[str, str]], token: str, sentiment: str
    ) -> None:
        tokens.append((token, sentiment))

    def inside_sentiment(
        self, tokens: List[Tuple[str, str]], token: str, sentiment: str
    ) -> None:
        if len(tokens) == 0:
            self.begin_sentiment(tokens, token, sentiment)
            return
        last_token, last_sentiment = tokens[-1]
        if last_sentiment != sentiment:
            self.begin_sentiment(tokens, token, sentiment)
        else:
            tokens[-1] = (f"{last_token} {token}", last_sentiment)

    def pick_sentiment(
        self, tokens: List[Tuple[str, str]], token: str, sentiment: str
    ) -> None:
        if sentiment == "O":
            self.begin_sentiment(tokens, token, sentiment)
        elif sentiment.startswith("B-"):
            self.begin_sentiment(tokens, token, sentiment[2:])
        elif sentiment.startswith("I-"):
            self.inside_sentiment(tokens, token, sentiment[2:])

    def compute_sentiments(
        self,
        input_tokens: List[str],
        output_sentiments: List[str],
    ) -> Dict[str, List[Tuple[str, str]]]:

        tokens2sentiments = []
        for i, (token, sentiment) in enumerate(zip(input_tokens, output_sentiments)):
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
                    elif output_sentiments[-1].startswith("S-") or output_sentiments[
                        -1
                    ].startswith("E-"):
                        tokens2sentiments.append([[token], sentiment[2:]])
                    # otherwise, the sentiment before was a B or a I with the same sentiment
                    # therefore this token is part of the same target instance, with the same sentiment associated
                    else:
                        tokens2sentiments[-1] = [last_token + [token], sentiment[2:]]
        if isinstance(self.tokenizer, TreebankWordTokenizer):
            detokenizer = TreebankWordDetokenizer()
            return {
                "targets": [
                    (detokenizer.detokenize(tk), sentiment)
                    for tk, sentiment in tokens2sentiments
                    if sentiment != "O"
                ]
            }
        else:
            return {
                "targets": [
                    (self.tokenizer.convert_tokens_to_string(tk), sentiment)
                    for tk, sentiment in tokens2sentiments
                    if sentiment != "O"
                ]
            }

    def postprocess(self, sentences, to_process, lengths):
        to_process_list: List[List[int]] = to_process.tolist()

        # remove padded elements
        for i, length in enumerate(lengths):
            to_process_list[i] = to_process_list[i][:length]

        # extract tokens and associated sentiments
        tokens = [self.tokenizer.tokenize(x) for x in sentences]

        # convert indexes to tokens + IOB format sentiments
        processed_iob_sentiments = [
            [self.sentiments_vocabulary.itos[x] for x in batch]
            for batch in to_process_list
        ]

        # convert IOB sentiments to simple | target words - sentiment | format
        words_to_sentiment = [
            self.compute_sentiments(token, output)
            for token, output in zip(tokens, processed_iob_sentiments)
        ]

        return words_to_sentiment


class F1SentimentExtraction(Metric):
    def __init__(
        self,
        vocabulary,
        sentiments_vocabulary,
        tokenizer: Union[TreebankWordTokenizer, BertTokenizer],
    ):
        super().__init__(dist_sync_on_step=False)
        self.vocabulary = vocabulary
        self.sentiments_vocabulary = sentiments_vocabulary
        self.sentiments_converter = TokenToSentimentsConverter(
            vocabulary, sentiments_vocabulary, tokenizer
        )
        self.add_state("tp", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.tensor(0.0), dist_reduce_fx="sum")

    # noinspection PyMethodOverriding
    def update(
        self,
        sentences: torch.Tensor,
        preds: torch.Tensor,
        gt_raw_data: torch.Tensor,
        lengths: List[int],
    ):
        sentences = [x["text"] for x in gt_raw_data]
        token_pred_sentiments = self.sentiments_converter.postprocess(
            sentences, preds, lengths
        )

        for label, pred in zip(gt_raw_data, token_pred_sentiments):
            pred_terms = {term_pred[0] for term_pred in pred["targets"]}
            gt_terms = {term_gt[1] for term_gt in label["targets"]}
            self.tp += len(pred_terms & gt_terms)
            self.fp += len(pred_terms - gt_terms)
            self.fn += len(gt_terms - pred_terms)

    def compute(self):
        precision = (
            self.tp / (self.tp + self.fp)
            if self.fp != 0
            else torch.tensor(1.0, device="cuda")
        )
        recall = (
            self.tp / (self.tp + self.fn)
            if self.fn != 0
            else torch.tensor(1.0, device="cuda")
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall != 0.0
            else torch.tensor(0.0, device="cuda")
        )
        # self.reset()
        return f1

    def reset(self):
        self.tp = torch.tensor(0, device="cuda")
        self.fp = torch.tensor(0, device="cuda")
        self.fn = torch.tensor(0, device="cuda")


class F1SentimentEvaluation(Metric):
    def __init__(
        self,
        vocabulary,
        sentiments_vocabulary,
        mode="Aspect Sentiment",
        tokenizer=None,
    ):
        super().__init__(dist_sync_on_step=False)
        self.vocabulary = vocabulary
        self.sentiments_vocabulary = sentiments_vocabulary
        self.tokenizer = tokenizer
        self.sentiments_converter = TokenToSentimentsConverter(
            vocabulary, sentiments_vocabulary, tokenizer=tokenizer
        )
        self.mode = mode
        if mode == "Category Extraction":
            self.sentiment_types = [
                "anecdotes/miscellaneous",
                "price",
                "food",
                "ambience",
            ]
        else:
            self.sentiment_types = ["positive", "negative", "neutral", "conflict"]

        for sent in self.sentiment_types + ["ALL"]:
            self.add_state(f"tp_{sent}", default=torch.tensor(0), dist_reduce_fx="sum")
            self.add_state(f"fp_{sent}", default=torch.tensor(0), dist_reduce_fx="sum")
            self.add_state(f"fn_{sent}", default=torch.tensor(0), dist_reduce_fx="sum")

    def compute_sent(self, sentiment, label, pred):
        if self.mode == "Aspect Sentiment":
            pred_sent = {
                (term_pred[0], term_pred[1])
                for term_pred in pred["targets"]
                if term_pred[1] == sentiment
            }
            gt_sent = {
                (term_pred[1], term_pred[2])
                for term_pred in label["targets"]
                if term_pred[2] == sentiment
            }
        elif self.mode == "Category Extraction" and "categories" in label:
            pred_sent = {
                (term_pred[0])
                for term_pred in pred["categories"]
                if term_pred[0] == sentiment
            }
            gt_sent = {
                (term_pred[0])
                for term_pred in label["categories"]
                if term_pred[0] == sentiment
            }
        elif "categories" in label:
            pred_sent = {
                (term_pred[0], term_pred[1])
                for term_pred in pred["categories"]
                if term_pred[1] == sentiment
            }
            gt_sent = {
                (term_pred[0], term_pred[1])
                for term_pred in label["categories"]
                if term_pred[1] == sentiment
            }
        else:
            return None

        return pred_sent, gt_sent

    def get_state(self, name):
        return getattr(self, name)

    def set_state(self, name, value):
        setattr(self, name, value)

    def tp(self, sent):
        return self.get_state(f"tp_{sent}")

    def fp(self, sent):
        return self.get_state(f"fp_{sent}")

    def fn(self, sent):
        return self.get_state(f"fn_{sent}")

    # noinspection PyMethodOverriding
    def update(
        self,
        sentences: torch.Tensor,
        preds: torch.Tensor,
        gt_raw_data: torch.Tensor,
        lengths: List[int],
    ):

        sentences = [x["text"] for x in gt_raw_data]
        token_pred_sentiments = self.sentiments_converter.postprocess(
            sentences, preds, lengths
        )

        for label, pred in zip(gt_raw_data, token_pred_sentiments):
            for sent_type in self.sentiment_types + ["ALL"]:
                pred_sent, gt_sent = self.compute_sent(sent_type, label, pred)
                self.set_state(
                    f"tp_{sent_type}", self.tp(sent_type) + len(pred_sent & gt_sent)
                )
                self.set_state(
                    f"fp_{sent_type}", self.fp(sent_type) + len(pred_sent - gt_sent)
                )
                self.set_state(
                    f"fn_{sent_type}", self.fn(sent_type) + len(gt_sent - pred_sent)
                )

    def compute(self):
        scores = defaultdict(dict)
        for sent_type in self.sentiment_types:
            if self.tp(sent_type):
                scores[sent_type]["p"] = self.tp(sent_type) / (
                    self.fp(sent_type) + self.tp(sent_type)
                )
                scores[sent_type]["r"] = self.tp(sent_type) / (
                    self.fn(sent_type) + self.tp(sent_type)
                )
            else:
                scores[sent_type]["p"] = torch.tensor(0.0)
                scores[sent_type]["r"] = torch.tensor(0.0)

            if not scores[sent_type]["p"] + scores[sent_type]["r"] == 0:
                scores[sent_type]["f1"] = (
                    2
                    * scores[sent_type]["p"]
                    * scores[sent_type]["r"]
                    / (scores[sent_type]["p"] + scores[sent_type]["r"])
                )
            else:
                scores[sent_type]["f1"] = torch.tensor(0.0)

        # Compute micro F1 Scores
        tp = sum([self.tp(sent_type) for sent_type in self.sentiment_types])
        fp = sum([self.fp(sent_type) for sent_type in self.sentiment_types])
        fn = sum([self.fn(sent_type) for sent_type in self.sentiment_types])

        if tp:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * precision * recall / (precision + recall)

        else:
            precision, recall, f1 = (
                torch.tensor(0.0),
                torch.tensor(0.0),
                torch.tensor(0.0),
            )

        scores["ALL"]["p"] = precision
        scores["ALL"]["r"] = recall
        scores["ALL"]["f1"] = f1

        # Compute Macro F1 Scores
        scores["ALL"]["Macro_f1"] = sum(
            [scores[ent_type]["f1"] for ent_type in self.sentiment_types]
        ) / len(self.sentiment_types)
        scores["ALL"]["Macro_p"] = sum(
            [scores[ent_type]["p"] for ent_type in self.sentiment_types]
        ) / len(self.sentiment_types)
        scores["ALL"]["Macro_r"] = sum(
            [scores[ent_type]["r"] for ent_type in self.sentiment_types]
        ) / len(self.sentiment_types)

        return scores["ALL"]["Macro_f1"]

    def reset(self):
        for sent in self.sentiment_types + ["ALL"]:
            self.add_state(
                f"tp_{sent}", default=torch.tensor(0.0), dist_reduce_fx="sum"
            )
            self.add_state(
                f"fp_{sent}", default=torch.tensor(0.0), dist_reduce_fx="sum"
            )
            self.add_state(
                f"fn_{sent}", default=torch.tensor(0.0), dist_reduce_fx="sum"
            )
