import torch
import torchmetrics
import pytorch_lightning as pl
from torch import nn, optim
from typing import *

from hw2.stud.utils import (
    TokenToSentimentsConverter,
    evaluate_extraction,
    evaluate_sentiment,
)


class ABSAModel(nn.Module):
    # we provide the hyperparameters as input
    def __init__(self, hparams, embeddings=None):
        super(ABSAModel, self).__init__()
        # Embedding layer: a mat∂rix vocab_size x embedding_dim where each index
        # correspond to a word in the vocabulary and the i-th row corresponds to
        # a latent representation of the i-th word in the vocabulary.
        self.word_embedding = nn.Embedding(hparams.vocab_size, hparams.embedding_dim)
        if embeddings is not None:
            print("initializing embeddings from pretrained")
            self.word_embedding.weight.data.copy_(embeddings)

        # LSTM layer: an LSTM neural network that process the input text
        # (encoded with word embeddings) from left to right and outputs
        # a new **contextual** representation of each word that depend
        # on the preciding words.
        self.lstm = nn.LSTM(
            hparams.embedding_dim,
            hparams.hidden_dim,
            bidirectional=hparams.bidirectional,
            num_layers=hparams.num_layers,
            dropout=hparams.dropout if hparams.num_layers > 1 else 0,
        )
        # Hidden layer: transforms the input value/scalar into
        # a hidden vector representation.
        lstm_output_dim = (
            hparams.hidden_dim
            if hparams.bidirectional is False
            else hparams.hidden_dim * 2
        )

        # During training, randomly zeroes some of the elements of the
        # input tensor with probability hparams.dropout using samples
        # from a Bernoulli distribution. Each channel will be zeroed out
        # independently on every forward call.
        # This has proven to be an effective technique for regularization and
        # preventing the co-adaptation of neurons
        self.dropout = nn.Dropout(hparams.dropout)
        self.classifier = nn.Linear(lstm_output_dim, hparams.num_classes)

    def forward(self, x):
        embeddings = self.word_embedding(x)
        embeddings = self.dropout(embeddings)
        o, (h, c) = self.lstm(embeddings)
        o = self.dropout(o)
        output = self.classifier(o)
        return output


class PlABSAModel(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.loss_function = nn.CrossEntropyLoss(
            ignore_index=self.hparams.sentiments_vocabulary["<pad>"]
        )
        self.model = ABSAModel(self.hparams, self.hparams.embeddings)
        self.sentiments_converter = TokenToSentimentsConverter()

    # This performs a forward pass of the model, as well as returning the predicted index.
    def forward(self, x):
        lengths = (x != 0).sum(-1)
        logits = self.model(x)
        predictions = torch.argmax(logits, -1)
        return {"logits": logits, "predictions": predictions, "lengths": lengths}

    def forward_sentiments(
        self,
        input_tensor: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ):
        # compute outputs
        output_tensor = self.forward(input_tensor)["predictions"]
        # convert input and output to list
        input_list: List[List[int]] = input_tensor.tolist()
        output_list: List[List[int]] = output_tensor.tolist()
        if lengths is not None:
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

    # This runs the model in training mode mode, ie. activates dropout
    # and gradient computation. It defines a single training step.
    def training_step(self, batch, batch_nb):
        inputs = batch["inputs"]
        labels = batch["outputs"]
        # We receive one batch of data and perform a forward pass:
        outputs = self.forward(inputs)
        logits = outputs["logits"]
        # We adapt the logits and labels to fit the format required for the loss function
        logits = logits.view(-1, logits.shape[-1])
        labels = labels.view(-1)
        predictions = outputs["predictions"].view(-1)

        # Compute the loss:
        loss = self.loss_function(logits, labels)
        # Log it:
        self.log_dict(
            {"train_loss": loss},
            prog_bar=True,
        )
        # Very important for PL to return the loss that will be used to update the weights:
        return loss

    # This runs the model in eval mode, ie. sets dropout to 0
    # and deactivates grad. Needed when we are in inference mode.
    def validation_step(self, batch, batch_nb):
        inputs = batch["inputs"]
        labels = batch["outputs"]
        lengths = (labels != 0).sum(-1)
        outputs = self.forward(inputs)
        logits = outputs["logits"]
        predictions = outputs["predictions"].view(-1)
        # We adapt the logits and labels to fit the format required for the loss function
        logits = logits.view(-1, logits.shape[-1])
        labels = labels.view(-1)
        sample_loss = self.loss_function(logits, labels)
        self.log_dict(
            {"valid_loss": sample_loss},
            prog_bar=True,
        )
        tokens2sentiments = self.forward_sentiments(inputs, lengths)
        return tokens2sentiments

    def validation_epoch_end(self, outputs) -> None:
        outputs = [item for sublist in outputs for item in sublist]
        f1_extraction = evaluate_extraction(self.hparams.dev_raw_data, outputs)
        f1_sentiment = evaluate_sentiment(self.hparams.dev_raw_data, outputs)
        self.log_dict(
            {"f1_extraction": f1_extraction, "f1_sentiment": f1_sentiment},
            prog_bar=True,
        )

    def configure_optimizers(self):
        return optim.Adam(self.parameters())
