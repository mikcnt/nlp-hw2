import torch
from torchmetrics import Metric
from typing import *


class TokenToSentimentsConverter(object):
    def __init__(self, vocabulary, sentiments_vocabulary):
        self.vocabulary = vocabulary
        self.sentiments_vocabulary = sentiments_vocabulary

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

    def postprocess(self, sentences, to_process, lengths):
        sentences_list: List[List[int]] = sentences.tolist()
        to_process_list: List[List[int]] = to_process.tolist()

        # remove padded elements
        for i, length in enumerate(lengths):
            sentences_list[i] = sentences_list[i][:length]
            to_process_list[i] = to_process_list[i][:length]

        # extract tokens and associated sentiments
        tokens = [[self.vocabulary.itos[x] for x in batch] for batch in sentences_list]

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
    def __init__(self, vocabulary, sentiments_vocabulary, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.vocabulary = vocabulary
        self.sentiments_vocabulary = sentiments_vocabulary
        self.sentiments_converter = TokenToSentimentsConverter(
            vocabulary, sentiments_vocabulary
        )
        self.add_state("tp", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.tensor(0.0), dist_reduce_fx="sum")

    # noinspection PyMethodOverriding
    def update(
        self,
        sentences: torch.Tensor,
        preds: torch.Tensor,
        labels: torch.Tensor,
        lengths: List[int],
    ):

        token_label_sentiments = self.sentiments_converter.postprocess(
            sentences, labels, lengths
        )
        token_pred_sentiments = self.sentiments_converter.postprocess(
            sentences, preds, lengths
        )

        for label, pred in zip(token_label_sentiments, token_pred_sentiments):
            pred_terms = {term_pred[0] for term_pred in pred["targets"]}
            gt_terms = {term_gt[0] for term_gt in label["targets"]}
            self.tp += torch.tensor(len(pred_terms & gt_terms))
            self.fp += torch.tensor(len(pred_terms - gt_terms))
            self.fn += torch.tensor(len(gt_terms - pred_terms))

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
        return f1
