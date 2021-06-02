from typing import *
import torch
import pytorch_lightning as pl
from nltk import TreebankWordTokenizer
from torch import nn, optim
from torchtext.vocab import Vocab
from transformers import BertTokenizer, AdamW

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
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(hparams)
        vocabulary = vocabularies["vocabulary"]
        sentiments_vocabulary = vocabularies["sentiments_vocabulary"]
        self.sentiments_vocabulary = sentiments_vocabulary
        # self.sentiments_converter = TokenToSentimentsConverter(
        #     vocabulary, sentiments_vocabulary, tokenizer=tokenizer
        # )
        self.loss_function = nn.CrossEntropyLoss(
            ignore_index=sentiments_vocabulary["<pad>"]
        )
        self.f1_extraction = F1SentimentExtraction(
            vocabulary=vocabulary,
            sentiments_vocabulary=sentiments_vocabulary,
            tokenizer=tokenizer,
            tagging_schema=self.hparams.tagging_schema,
        )
        self.f1_evaluation = F1SentimentEvaluation(
            vocabulary=vocabulary,
            sentiments_vocabulary=sentiments_vocabulary,
            tokenizer=tokenizer,
            tagging_schema=self.hparams.tagging_schema,
        )

        self.model = (
            ABSAModel(self.hparams, embeddings)
            if not self.hparams.use_bert
            else ABSABert(self.hparams)
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        sentences = batch["inputs"]
        lengths = batch["lengths"]
        logits = self.model(sentences, lengths)
        predictions = torch.argmax(logits, -1)
        return {"logits": logits, "predictions": predictions}

    def forward_processed(self, batch: Dict[str, torch.Tensor]):
        sentences = batch["inputs"]
        lengths = batch["lengths"]
        predictions = self(batch)["predictions"]
        processed_output = self.sentiments_converter.postprocess(
            sentences, predictions, lengths
        )
        return processed_output

    def step(
        self, batch: Dict[str, torch.Tensor], compute_f1=False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        sentences = batch["inputs"]
        lengths = batch["lengths"]
        labels = batch["outputs"]
        raw_data = batch["raw"]
        # We receive one batch of data and perform a forward pass:
        logits = self.model(sentences, lengths)

        # compute loss and f1 score
        if self.hparams.use_crf:
            mask = labels != self.sentiments_vocabulary["<pad>"]
            loss = -1 * self.model.crf(logits, labels, mask=mask)
            predictions = torch.tensor(
                self.model.crf.decode(logits), device=self.device
            )
        else:
            loss = self.loss_function(
                logits.view(-1, logits.shape[-1]), labels.view(-1)
            )
            predictions = torch.argmax(logits, -1)
        if compute_f1:
            f1_extraction = self.f1_extraction(
                sentences, predictions, raw_data, lengths
            )
            f1_evaluation = self.f1_evaluation(
                sentences, predictions, raw_data, lengths
            )
            return loss, f1_extraction, f1_evaluation
        else:
            return loss

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: Optional[int]
    ) -> torch.Tensor:
        train_loss = self.step(batch)
        # Log loss and f1 score
        self.log_dict(
            {
                "train_loss": train_loss,
            },
            prog_bar=True,
        )

        # Return loss to update weights
        return train_loss

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: Optional[int]
    ) -> Tuple[Any, Any]:
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
            return AdamW(self.parameters(), lr=2e-5)
