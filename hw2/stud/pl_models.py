import torch
import pytorch_lightning as pl
from typing import *
from torch import nn, optim

from hw2.stud.metrics import F1SentimentExtraction
from hw2.stud.models import ABSAModel


class PlABSAModel(pl.LightningModule):
    def __init__(self, embeddings, raw_data, hparams, *args, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters(hparams)
        self.loss_function = nn.CrossEntropyLoss(
            ignore_index=self.hparams.sentiments_vocabulary["<pad>"]
        )
        self.f1_extraction = F1SentimentExtraction(
            vocabulary=self.hparams.vocabulary,
            sentiments_vocabulary=self.hparams.sentiments_vocabulary,
        )
        self.model = ABSAModel(self.hparams, embeddings)
        self.train_raw_data = raw_data["train"]
        self.dev_raw_data = raw_data["dev"]

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        logits = self.model(batch)
        predictions = torch.argmax(logits, -1)
        return {"logits": logits, "predictions": predictions}

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: Optional[int]
    ) -> torch.Tensor:
        sentences = batch["inputs"]
        labels = batch["outputs"]
        lengths = batch["lengths"]
        # We receive one batch of data and perform a forward pass:
        output_batch = self.forward(batch)
        logits = output_batch["logits"]
        predictions = output_batch["predictions"]
        # We adapt the logits and labels to fit the format required for the loss function
        logits = logits.view(-1, logits.shape[-1])
        # compute loss
        train_loss = self.loss_function(logits, labels.view(-1))

        # Log the loss
        self.log_dict(
            {
                "train_loss": train_loss,
                "f1_train": self.f1_extraction(sentences, predictions, labels, lengths),
            },
            prog_bar=True,
        )

        # Return loss to update weights
        return train_loss

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: Optional[int]
    ) -> torch.Tensor:
        sentences = batch["inputs"]
        labels = batch["outputs"]
        lengths = batch["lengths"]
        # We receive one batch of data and perform a forward pass:
        output_batch = self.forward(batch)
        logits = output_batch["logits"]
        predictions = output_batch["predictions"]
        # We adapt the logits and labels to fit the format required for the loss function
        logits = logits.view(-1, logits.shape[-1])
        # compute loss
        val_loss = self.loss_function(logits, labels.view(-1))

        # Log the loss
        self.log_dict(
            {
                "val_loss": val_loss,
                "f1_val": self.f1_extraction(sentences, predictions, labels, lengths),
            },
            prog_bar=True,
        )

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
