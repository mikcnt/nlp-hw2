from typing import *
import torch
import pytorch_lightning as pl
import transformers
from nltk import TreebankWordTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from torch import nn, optim
from torchtext.vocab import Vocab

from stud.dataset import tags_to_json
from stud.metrics import (
    F1SentimentExtraction,
    F1SentimentEvaluation,
)
from stud.models import ABSAModel


class PlABSAModel(pl.LightningModule):
    def __init__(
        self,
        hparams: Dict[str, Any],
        vocabularies: Dict[str, Vocab],
        embeddings: torch.Tensor,
        tokenizer: TreebankWordTokenizer,
        train=True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(hparams)
        self.sentiments_vocabulary = vocabularies["sentiments_vocabulary"]
        self.tokenizer = tokenizer
        self.detokenizer = TreebankWordDetokenizer()
        self.loss_function = nn.CrossEntropyLoss(
            ignore_index=self.sentiments_vocabulary["<pad>"]
        )
        # use f1 metrics only during training
        if train:
            self.f1_extraction_t = F1SentimentExtraction(device=self.device)
            self.f1_evaluation_t = F1SentimentEvaluation(device=self.device)
            self.f1_extraction_v = F1SentimentExtraction(device=self.device)
            self.f1_evaluation_v = F1SentimentEvaluation(device=self.device)

        self.model = ABSAModel(self.hparams, embeddings)

    def forward(
        self, batch: Dict[str, Union[torch.Tensor, List[int]]]
    ) -> Dict[str, Union[torch.Tensor, List]]:
        lengths = batch["lengths"]
        raw_data = batch["raw"]
        sentences_raw = [x["text"] for x in raw_data]

        logits = self.model(batch)

        # if CRF is used, the Viterbi algorithm is used to compute the predictions
        # otherwise we just take the argmax
        if not self.hparams.use_crf:
            predictions = torch.argmax(logits, -1)
        else:
            predictions = torch.tensor(
                self.model.crf.decode(logits), device=self.device
            )

        # compute `text_predictions`, that are just the predictions in readable format
        text_predictions = self._batch_sentiments_to_tags(
            sentences_raw, predictions, lengths
        )
        return {
            "logits": logits,
            "predictions": predictions,
            "text_predictions": text_predictions,
        }

    def predict(self, batch: Dict[str, torch.Tensor]) -> List[Dict[str, Any]]:
        """This method returns the predictions in their original form, as found in the dataset.
        This way, we can compute the f1 score between the predictions and the ground truths.
        Use this function during inference time."""
        text_predictions = self(batch)["text_predictions"]
        return text_predictions

    def step(self, batch: Dict[str, torch.Tensor], phase: str) -> Dict[str, Any]:
        output = self(batch)
        labels = batch["labels"]
        logits = output["logits"]
        text_predictions = output["text_predictions"]
        text_gt = batch["raw"]

        step_output = {}

        # compute loss and f1 score
        if self.hparams.use_crf:
            mask = batch["attention_mask"]
            step_output["loss"] = -1 * self.model.crf(logits, labels, mask=mask)
        else:
            logits = logits.view(-1, self.hparams.num_classes)
            labels = labels.view(-1)
            step_output["loss"] = self.loss_function(logits, labels)

        if phase == "train":
            f1_extraction = self.f1_extraction_t
            f1_evaluation = self.f1_evaluation_t
        else:
            f1_extraction = self.f1_extraction_v
            f1_evaluation = self.f1_evaluation_v

        step_output["f1_extraction"] = f1_extraction(text_predictions, text_gt)
        step_output["f1_evaluation"] = f1_evaluation(text_predictions, text_gt)

        return step_output

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: Optional[int]
    ) -> Dict[str, torch.Tensor]:
        step_output = self.step(batch, phase="train")
        train_loss = step_output["loss"]
        f1_extraction = step_output["f1_extraction"]
        f1_evaluation = step_output["f1_evaluation"]
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
        return {
            "loss": train_loss,
            "f1_extraction": f1_extraction,
            "f1_evaluation": f1_evaluation,
        }

    def training_epoch_end(self, outputs):
        f1_extraction = self.f1_extraction_t.compute()
        f1_evaluation = self.f1_evaluation_t.compute()
        self.log_dict(
            {
                "f1_extraction_train": f1_extraction,
                "f1_evaluation_train": f1_evaluation,
            },
            prog_bar=False,
        )
        self.f1_extraction_t.reset()
        self.f1_evaluation_t.reset()

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: Optional[int]
    ) -> Dict[str, torch.Tensor]:
        step_output = self.step(batch, phase="val")
        val_loss = step_output["loss"]
        f1_extraction = step_output["f1_extraction"]
        f1_evaluation = step_output["f1_evaluation"]
        # Log loss and f1 score
        self.log_dict(
            {
                "val_loss": val_loss,
            },
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        return {"f1_extraction": f1_extraction, "f1_evaluation": f1_evaluation}

    def validation_epoch_end(self, outputs):
        f1_extraction = self.f1_extraction_v.compute()
        f1_evaluation = self.f1_evaluation_v.compute()
        self.log_dict(
            {"f1_extraction_val": f1_extraction, "f1_evaluation_val": f1_evaluation},
            prog_bar=True,
        )
        self.f1_extraction_v.reset()
        self.f1_evaluation_v.reset()

    def configure_optimizers(self) -> optim.Optimizer:
        # dynamically instantiate optimizer
        optimizer = eval(self.hparams.optimizer)
        return optimizer(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

    def _postprocess(self, tokens: List[str], sentiments: List[str]) -> Dict[str, Any]:
        """Postprocess the tokens and sentiments to convert them back into the JSON format of the dataset."""
        tokens2sentiments = tags_to_json(
            tokens, sentiments, self.hparams.tagging_schema
        )

        return {
            "targets": [
                (self.detokenizer.detokenize(tk), sentiment)
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
