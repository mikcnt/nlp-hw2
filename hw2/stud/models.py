import torch
from torch import nn
from typing import *
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from transformers import BertModel, BertForTokenClassification
from torchcrf import CRF


def lstm_padded(
    lstm_layer: nn.Module, x: torch.Tensor, lengths: List[int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    x_packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
    o_packed, (h, c) = lstm_layer(x_packed)
    return pad_packed_sequence(o_packed, batch_first=True)


class ABSAModel(nn.Module):
    # we provide the hyperparameters as input
    def __init__(self, hparams, embeddings=None):
        super(ABSAModel, self).__init__()
        self.word_embedding = nn.Embedding(hparams.vocab_size, hparams.embedding_dim)
        if embeddings is not None:
            print("initializing embeddings from pretrained")
            self.word_embedding.weight.data.copy_(embeddings)

        self.lstm = nn.LSTM(
            hparams.embedding_dim,
            hparams.hidden_dim,
            bidirectional=hparams.bidirectional,
            num_layers=hparams.num_layers,
            dropout=hparams.dropout if hparams.num_layers > 1 else 0,
        )
        lstm_output_dim = (
            hparams.hidden_dim
            if hparams.bidirectional is False
            else hparams.hidden_dim * 2
        )

        self.dropout = nn.Dropout(hparams.dropout)
        self.classifier = nn.Linear(lstm_output_dim, hparams.num_classes)

    def forward(self, x, x_lengths):
        embeddings = self.word_embedding(x)
        embeddings = self.dropout(embeddings)
        o, o_lengths = lstm_padded(self.lstm, embeddings, x_lengths)
        o = self.dropout(o)
        output = self.classifier(o)
        return output


class NER_WORD_MODEL_CRF(nn.Module):
    # we provide the hyperparameters as input
    def __init__(self, hparams, embeddings=None):
        super(NER_WORD_MODEL_CRF, self).__init__()
        # Embedding layer: a mat∂rix vocab_size x embedding_dim where each index
        # correspond to a word in the vocabulary and the i-th row corresponds to
        # a latent representation of the i-th word in the vocabulary.

        self.word_embedding = nn.Embedding(
            hparams.vocab_size, hparams.embedding_dim, padding_idx=0
        )
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
            batch_first=True,
        )
        # Hidden layer: transforms the input value/scalar into
        # a hidden vector representation.
        lstm_output_dim = (
            hparams.hidden_dim
            if hparams.bidirectional is False
            else hparams.hidden_dim * 2
        )
        self.linear_word = nn.Linear(lstm_output_dim, lstm_output_dim)

        self.dropout = nn.Dropout(hparams.dropout)
        self.concat = nn.Linear(lstm_output_dim, lstm_output_dim)
        self.concat2 = nn.Linear(lstm_output_dim, lstm_output_dim)
        self.concat3 = nn.Linear(lstm_output_dim, lstm_output_dim)

        self.fc1 = nn.Linear(lstm_output_dim, lstm_output_dim // 2)
        self.fc2 = nn.Linear(lstm_output_dim // 2, lstm_output_dim // 4)
        self.fc3 = nn.Linear(lstm_output_dim // 4, lstm_output_dim // 4)
        self.classifier = nn.Linear(lstm_output_dim // 4, hparams.num_classes)
        self.relu = nn.ReLU()

    def forward(self, x, x_lengths):

        embeddings = self.word_embedding(x)
        embeddings = self.dropout(embeddings)
        o, _ = lstm_padded(self.lstm, embeddings, x_lengths)
        o = self.linear_word(o)
        o = self.dropout(o)
        o = self.concat(o)
        o = self.relu(o)
        o = self.concat2(o)
        o = self.relu(o)
        o = self.fc1(o)
        o = self.relu(o)
        o = self.fc2(o)
        o = self.dropout(o)
        o = self.relu(o)
        o = self.fc3(o)
        o = self.relu(o)
        output = self.classifier(o)
        return output


class ABSABert(nn.Module):
    def __init__(self, hparams):
        super(ABSABert, self).__init__()
        # self.model = BertForTokenClassification.from_pretrained(
        #     "bert-base-cased", num_labels=hparams.num_classes
        # )

        self.bert = BertModel.from_pretrained("bert-base-cased")
        bert_output_dim = self.bert.config.hidden_size
        # self.lstm = nn.LSTM(bert_output_dim, 300)
        self.dropout = nn.Dropout(hparams.dropout)

        self.classifier = nn.Linear(bert_output_dim, hparams.num_classes)

    def forward(self, x, x_lengths):
        attention_mask = torch.ones_like(x)
        for i in x_lengths:
            attention_mask[..., i:] = 0

        output = self.bert(x, attention_mask)["last_hidden_state"]
        output = self.dropout(output)

        # output, _ = self.lstm(output)
        output = self.classifier(output)
        return output

    # def forward(self, x, x_lengths):
    #     attention_mask = torch.ones_like(x)
    #     for i in x_lengths:
    #         attention_mask[..., i:] = 0
    #
    #     output = self.model(x, attention_mask)["logits"]
    #
    #     return output
