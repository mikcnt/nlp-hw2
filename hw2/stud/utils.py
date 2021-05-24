import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import Vocab
from typing import *
import pickle


def save_pickle(data: Any, path: str) -> None:
    """Save object as pickle file."""
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_pickle(path: str) -> Any:
    """Load object from pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)


def pad_collate(batch):
    # pad x
    xx = [x["inputs"] for x in batch]
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)

    # pad y
    try:
        yy = [x["outputs"] for x in batch]
        yy_pad = pad_sequence(yy, batch_first=True, padding_value=0)
    except:
        yy_pad = None

    # lengths of inputs == lenghts of outputs
    lengths = [len(x) for x in xx]

    batch = {
        "inputs": xx_pad,
        "outputs": yy_pad,
        "lengths": lengths,
    }

    return batch


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
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall != 0
        else np.nan
    )
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
