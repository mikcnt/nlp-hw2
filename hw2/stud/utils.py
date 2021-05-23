import torch
from torchtext.vocab import Vocab
from typing import *


class TokenToSentimentsConverter(object):
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
        if sentiment == "0":
            self.begin_sentiment(tokens, token, sentiment)
        elif sentiment.startswith("B-"):
            self.begin_sentiment(tokens, token, sentiment[2:])
        elif sentiment.startswith("I-"):
            self.inside_sentiment(tokens, token, sentiment[2:])

    def compute_sentiments(
        self, input_tokens: List[str], output_sentiments: List[str]
    ) -> Dict[str, List[Tuple[str, str]]]:
        tokens2sentiments = []
        for token, sentiment in zip(input_tokens, output_sentiments):
            self.pick_sentiment(tokens2sentiments, token, sentiment)
        return {
            "targets": [
                (tk, sentiment)
                for tk, sentiment in tokens2sentiments
                if sentiment != "0"
            ]
        }


def evaluate_extraction(samples, predictions_b):
    scores = {"tp": 0, "fp": 0, "fn": 0}
    for label, pred in zip(samples, predictions_b):
        pred_terms = {term_pred[0] for term_pred in pred["targets"]}
        gt_terms = {term_gt[1] for term_gt in label["targets"]}

        scores["tp"] += len(pred_terms & gt_terms)
        scores["fp"] += len(pred_terms - gt_terms)
        scores["fn"] += len(gt_terms - pred_terms)

    precision = (
        scores["tp"] / (scores["tp"] + scores["fp"]) if scores["fp"] != 0 else 1.0
    )
    recall = scores["tp"] / (scores["tp"] + scores["fn"]) if scores["fn"] != 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision or recall else 0.0
    return f1


def evaluate_sentiment(samples, predictions_b, mode="Aspect Sentiment"):
    scores = {}
    if mode == "Category Extraction":
        sentiment_types = ["anecdotes/miscellaneous", "price", "food", "ambience"]
    else:
        sentiment_types = ["positive", "negative", "neutral", "conflict"]
    scores = {sent: {"tp": 0, "fp": 0, "fn": 0} for sent in sentiment_types + ["ALL"]}
    for label, pred in zip(samples, predictions_b):
        for sentiment in sentiment_types:
            if mode == "Aspect Sentiment":
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
            elif mode == "Category Extraction" and "categories" in label:
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
                continue

            scores[sentiment]["tp"] += len(pred_sent & gt_sent)
            scores[sentiment]["fp"] += len(pred_sent - gt_sent)
            scores[sentiment]["fn"] += len(gt_sent - pred_sent)

    # Compute per sentiment Precision / Recall / F1
    for sent_type in scores.keys():
        if scores[sent_type]["tp"]:
            scores[sent_type]["p"] = (
                100
                * scores[sent_type]["tp"]
                / (scores[sent_type]["fp"] + scores[sent_type]["tp"])
            )
            scores[sent_type]["r"] = (
                100
                * scores[sent_type]["tp"]
                / (scores[sent_type]["fn"] + scores[sent_type]["tp"])
            )
        else:
            scores[sent_type]["p"], scores[sent_type]["r"] = 0, 0

        if not scores[sent_type]["p"] + scores[sent_type]["r"] == 0:
            scores[sent_type]["f1"] = (
                2
                * scores[sent_type]["p"]
                * scores[sent_type]["r"]
                / (scores[sent_type]["p"] + scores[sent_type]["r"])
            )
        else:
            scores[sent_type]["f1"] = 0

    # Compute micro F1 Scores
    tp = sum([scores[sent_type]["tp"] for sent_type in sentiment_types])
    fp = sum([scores[sent_type]["fp"] for sent_type in sentiment_types])
    fn = sum([scores[sent_type]["fn"] for sent_type in sentiment_types])

    if tp:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)

    else:
        precision, recall, f1 = 0, 0, 0

    scores["ALL"]["p"] = precision
    scores["ALL"]["r"] = recall
    scores["ALL"]["f1"] = f1
    scores["ALL"]["tp"] = tp
    scores["ALL"]["fp"] = fp
    scores["ALL"]["fn"] = fn

    # Compute Macro F1 Scores
    scores["ALL"]["Macro_f1"] = sum(
        [scores[ent_type]["f1"] for ent_type in sentiment_types]
    ) / len(sentiment_types)
    scores["ALL"]["Macro_p"] = sum(
        [scores[ent_type]["p"] for ent_type in sentiment_types]
    ) / len(sentiment_types)
    scores["ALL"]["Macro_r"] = sum(
        [scores[ent_type]["r"] for ent_type in sentiment_types]
    ) / len(sentiment_types)

    return f1
