from torch import nn
from typing import *


class ABSAModel(nn.Module):
    # we provide the hyperparameters as input
    def __init__(self, hparams, embeddings=None):
        super(ABSAModel, self).__init__()
        # Embedding layer: a matrix vocab_size x embedding_dim where each index
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
