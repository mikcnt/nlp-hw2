from collections import defaultdict
import torch
from torchmetrics import Metric
from typing import *


class F1SentimentExtraction(Metric):
    def __init__(self, device):
        super().__init__(dist_sync_on_step=False)
        self.device = device
        self.add_state("tp", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.tensor(0.0), dist_reduce_fx="sum")

    # noinspection PyMethodOverriding
    def update(self, preds_text: torch.Tensor, gt_text: torch.Tensor):
        for label, pred in zip(gt_text, preds_text):
            pred_terms = {term_pred[0] for term_pred in pred["targets"]}
            gt_terms = {term_gt[1] for term_gt in label["targets"]}
            self.tp += len(pred_terms & gt_terms)
            self.fp += len(pred_terms - gt_terms)
            self.fn += len(gt_terms - pred_terms)

    def compute(self):
        precision = (
            self.tp / (self.tp + self.fp)
            if self.fp != 0
            else torch.tensor(1.0, device=self.device)
        )
        recall = (
            self.tp / (self.tp + self.fn)
            if self.fn != 0
            else torch.tensor(1.0, device=self.device)
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall != 0.0
            else torch.tensor(0.0, device=self.device)
        )
        return f1

    def reset(self):
        self.add_state("tp", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.tensor(0.0), dist_reduce_fx="sum")


class F1SentimentEvaluation(Metric):
    def __init__(
        self,
        device,
        mode="Aspect Sentiment",
    ):
        super().__init__(dist_sync_on_step=False)
        self.device = device
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
            self.add_state(
                f"tp_{sent}",
                default=torch.tensor(0, device=self.device),
                dist_reduce_fx="sum",
            )
            self.add_state(
                f"fp_{sent}",
                default=torch.tensor(0, device=self.device),
                dist_reduce_fx="sum",
            )
            self.add_state(
                f"fn_{sent}",
                default=torch.tensor(0, device=self.device),
                dist_reduce_fx="sum",
            )

    # noinspection PyMethodOverriding
    def update(self, preds_text: torch.Tensor, gt_text: torch.Tensor):

        for label, pred in zip(gt_text, preds_text):
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
                scores[sent_type]["p"] = torch.tensor(0.0, device=self.device)
                scores[sent_type]["r"] = torch.tensor(0.0, device=self.device)

            if not scores[sent_type]["p"] + scores[sent_type]["r"] == 0:
                scores[sent_type]["f1"] = (
                    2
                    * scores[sent_type]["p"]
                    * scores[sent_type]["r"]
                    / (scores[sent_type]["p"] + scores[sent_type]["r"])
                )
            else:
                scores[sent_type]["f1"] = torch.tensor(0.0, device=self.device)

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
                torch.tensor(0.0, device=self.device),
                torch.tensor(0.0, device=self.device),
                torch.tensor(0.0, device=self.device),
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

        return torch.tensor(scores["ALL"]["Macro_f1"], device=self.device)

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
