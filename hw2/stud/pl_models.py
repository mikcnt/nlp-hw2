from typing import *
import torch
import pytorch_lightning as pl
from nltk import TreebankWordTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from torch import nn, optim
from torchtext.vocab import Vocab
from transformers import BertTokenizer, AdamW

from stud.dataset import tags_to_json
from stud.metrics import (
    F1SentimentExtraction,
    F1SentimentEvaluation,
)
from stud.models import ABSAModel, ABSABert


class PlABSAModel(pl.LightningModule):
    def __init__(
        self,
        hparams: Dict[str, Any],
        vocabularies: Dict[str, Vocab],
        embeddings: torch.Tensor,
        tokenizer: Union[TreebankWordTokenizer, BertTokenizer],
        train=True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(hparams)
        vocabulary = vocabularies["vocabulary"]
        self.sentiments_vocabulary = vocabularies["sentiments_vocabulary"]
        self.tokenizer = tokenizer
        self.loss_function = nn.CrossEntropyLoss(
            ignore_index=self.sentiments_vocabulary["<pad>"]
        )
        if train:
            self.f1_extraction = F1SentimentExtraction(device=self.device)
            self.f1_evaluation = F1SentimentEvaluation(device=self.device)

        self.model = (
            ABSAModel(self.hparams, embeddings)
            if not self.hparams.use_bert
            else ABSABert(self.hparams, embeddings)
        )

    def forward(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, Union[torch.Tensor, List]]:
        sentences = batch["inputs"]
        lengths = batch["lengths"]
        attention_mask = batch["attention_mask"]
        raw_data = batch["raw"]
        sentences_raw = [x["text"] for x in raw_data]
        bert_embeddings = batch['bert_embeddings']

        logits = self.model(sentences, lengths, attention_mask, bert_embeddings)
        if not self.hparams.use_crf:
            predictions = torch.argmax(logits, -1)
        else:
            predictions = torch.tensor(
                self.model.crf.decode(logits), device=self.device
            )
        text_predictions = self._batch_sentiments_to_tags(
            sentences_raw, predictions, lengths
        )
        return {
            "logits": logits,
            "predictions": predictions,
            "text_predictions": text_predictions,
        }

    def predict(self, batch: Dict[str, torch.Tensor]) -> List[Dict[str, Any]]:
        text_predictions = self(batch)["text_predictions"]
        return text_predictions

    def step(
        self, batch: Dict[str, torch.Tensor], compute_f1=False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        output = self(batch)
        labels = batch["outputs"]
        logits = output["logits"]
        text_predictions = output["text_predictions"]
        text_gt = batch["raw"]

        # compute loss and f1 score
        if self.hparams.use_crf:
            mask = batch["attention_mask"]
            loss = -1 * self.model.crf(logits, labels, mask=mask)
        else:
            logits = logits.view(-1, self.hparams.num_classes)
            labels = labels.view(-1)
            loss = self.loss_function(logits, labels)

        if compute_f1:
            f1_extraction = self.f1_extraction(text_predictions, text_gt)
            f1_evaluation = self.f1_evaluation(text_predictions, text_gt)
            return loss, f1_extraction, f1_evaluation
        else:
            return loss

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: Optional[int]
    ) -> torch.Tensor:
        train_loss = self.step(batch)
        # Log loss
        self.log_dict(
            {
                "train_loss": train_loss,
            },
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

        # Return loss to update weights
        return train_loss

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: Optional[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        val_loss, f1_extraction, f1_evaluation = self.step(batch, compute_f1=True)
        # Log loss and f1 score
        self.log_dict(
            {
                "val_loss": val_loss,
            },
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        return f1_extraction, f1_evaluation

    def validation_epoch_end(self, outputs):
        f1_extraction = self.f1_extraction.compute()
        f1_evaluation = self.f1_evaluation.compute()
        self.log_dict(
            {"f1_extraction": f1_extraction, "f1_evaluation": f1_evaluation},
            prog_bar=True,
        )
        self.f1_extraction.reset()
        self.f1_evaluation.reset()

    def configure_optimizers(self) -> optim.Optimizer:
        if not self.hparams.use_bert:
            return optim.Adam(
                self.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
            )
        else:
            return AdamW(
                self.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
                # correct_bias=False,
            )

    def _postprocess(self, tokens: List[str], sentiments: List[str]) -> Dict[str, Any]:
        tokens2sentiments = tags_to_json(
            tokens, sentiments, self.hparams.tagging_schema
        )

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

    def _batch_sentiments_to_tags(
        self,
        raw_data: List[Dict[str, Any]],
        batch_sentiments: torch.Tensor,
        lengths: List[int],
    ) -> List[Dict[str, Any]]:
        # convert sentiments to list
        sentiments_list = batch_sentiments.tolist()

        # remove padded elements
        for i, length in enumerate(lengths):
            sentiments_list[i] = sentiments_list[i][:length]

        # extract tokens and associated sentiments
        tokens = [self.tokenizer.tokenize(x) for x in raw_data]

        # convert indexes to tokens + IOB format sentiments
        processed_iob_sentiments = [
            [self.sentiments_vocabulary.itos[x] for x in batch]
            for batch in sentiments_list
        ]

        # convert IOB sentiments to simple | target words - sentiment | format
        words_to_sentiment = [
            self._postprocess(token, output)
            for token, output in zip(tokens, processed_iob_sentiments)
        ]

        return words_to_sentiment
