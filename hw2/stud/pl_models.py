from typing import *
import torch
import pytorch_lightning as pl
from torch import nn, optim
from torchtext.vocab import Vocab
from transformers import BertTokenizer, AdamW

from stud.metrics import (
    F1SentimentExtraction,
    TokenToSentimentsConverter,
    F1SentimentEvaluation,
)
from stud.models import ABSAModel, ABSABert


class PlABSAModel(pl.LightningModule):
    def __init__(
        self,
        hparams: Dict[str, Any],
        vocabularies: Dict[str, Vocab],
        embeddings: torch.Tensor,
        *args,
        **kwargs
    ) -> None:
        super().__init__()
        self.save_hyperparameters(hparams)
        vocabulary = vocabularies["vocabulary"]
        sentiments_vocabulary = vocabularies["sentiments_vocabulary"]
        self.sentiments_converter = TokenToSentimentsConverter(
            vocabulary, sentiments_vocabulary
        )
        self.loss_function = nn.CrossEntropyLoss(
            ignore_index=sentiments_vocabulary["<pad>"]
        )
        tokenizer = (
            BertTokenizer.from_pretrained("bert-base-cased")
            if self.hparams.use_bert
            else None
        )
        self.f1_extraction = F1SentimentExtraction(
            vocabulary=vocabulary,
            sentiments_vocabulary=sentiments_vocabulary,
            tokenizer=tokenizer,
        )
        self.f1_evaluation = F1SentimentEvaluation(
            vocabulary=vocabulary,
            sentiments_vocabulary=sentiments_vocabulary,
            tokenizer=tokenizer,
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
        predictions = torch.argmax(logits, -1)

        # We adapt the logits and labels to fit the format required for the loss function
        logits = logits.view(-1, logits.shape[-1])
        # compute loss and f1 score
        loss = self.loss_function(logits, labels.view(-1))
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
                # "f1_train": train_f1_score,
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
                # "f1_val": val_f1_score,
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
            return AdamW(self.parameters(), lr=5e-5)
