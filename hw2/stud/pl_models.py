import torch
import pytorch_lightning as pl
from typing import *
from torch import nn, optim
from hw2.stud.models import ABSAModel
from hw2.stud.utils import (
    TokenToSentimentsConverter,
    evaluate_extraction,
    evaluate_sentiment,
)


class PlABSAModel(pl.LightningModule):
    def __init__(self, embeddings, hparams) -> None:
        super().__init__()
        self.save_hyperparameters(hparams)
        self.loss_function = nn.CrossEntropyLoss(
            ignore_index=self.hparams.sentiments_vocabulary["<pad>"]
        )
        self.model = ABSAModel(self.hparams, embeddings)
        self.sentiments_converter = TokenToSentimentsConverter()

    def forward(self, input_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        lengths = torch.count_nonzero(input_tensor, -1)
        logits = self.model(input_tensor)
        predictions = torch.argmax(logits, -1)
        return {"logits": logits, "predictions": predictions, "lengths": lengths}

    def forward_sentiments(
        self,
        input_tensor: torch.Tensor,
    ) -> List[Dict[str, List[Tuple[str, str]]]]:
        # compute outputs
        outputs = self.forward(input_tensor)
        predictions = outputs["predictions"]
        lengths = outputs["lengths"]
        # convert input and output to list
        input_list: List[List[int]] = input_tensor.tolist()
        output_list: List[List[int]] = predictions.tolist()

        # remove padded elements
        for i, length in enumerate(lengths):
            input_list[i] = input_list[i][:length]
            output_list[i] = output_list[i][:length]

        # extract tokens and associated sentiments
        input_tokens = [
            [self.hparams.vocabulary.itos[x] for x in batch] for batch in input_list
        ]

        output_sentiments = [
            [self.hparams.sentiments_vocabulary.itos[x] for x in batch]
            for batch in output_list
        ]

        return [
            self.sentiments_converter.compute_sentiments(in_tokens, out_sentiments)
            for in_tokens, out_sentiments in zip(input_tokens, output_sentiments)
        ]

    def step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        inputs = batch["inputs"]
        labels = batch["outputs"]
        lengths = (labels != 0).sum(-1)
        # We receive one batch of data and perform a forward pass:
        outputs = self.forward(inputs)
        logits = outputs["logits"]
        # We adapt the logits and labels to fit the format required for the loss function
        logits = logits.view(-1, logits.shape[-1])
        labels = labels.view(-1)
        # compute loss
        loss = self.loss_function(logits, labels)
        return loss

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: Optional[int]
    ) -> torch.Tensor:
        # Compute loss
        train_loss = self.step(batch)
        # Log the loss
        self.log_dict(
            {"train_loss": train_loss},
            prog_bar=True,
        )
        # Return loss to update weights
        return train_loss

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: Optional[int]
    ) -> List[Dict[str, List[Tuple[str, str]]]]:
        # Compute loss
        valid_loss = self.step(batch)
        self.log_dict(
            {"valid_loss": valid_loss},
            prog_bar=True,
        )
        tokens2sentiments = self.forward_sentiments(batch["inputs"])
        return tokens2sentiments

    def validation_epoch_end(
        self, outputs: List[List[Dict[str, List[Tuple[str, str]]]]]
    ) -> None:
        outputs = [item for sublist in outputs for item in sublist]
        f1_extraction = evaluate_extraction(self.hparams.dev_raw_data, outputs)
        f1_sentiment = evaluate_sentiment(self.hparams.dev_raw_data, outputs)
        self.log_dict(
            {"f1_extraction": f1_extraction, "f1_sentiment": f1_sentiment},
            prog_bar=True,
        )

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.Adam(self.parameters())
