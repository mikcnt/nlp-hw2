import torch
from torch import nn
from typing import *
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torchcrf import CRF
from transformers import BertModel


def lstm_padded(
    lstm_layer: nn.Module, x: torch.Tensor, lengths: List[int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    x_packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
    o_packed, _ = lstm_layer(x_packed)
    return pad_packed_sequence(o_packed, batch_first=True)


class ABSAModel(nn.Module):
    def __init__(self, hparams, embeddings=None):
        super(ABSAModel, self).__init__()
        self.word_embedding = nn.Embedding(hparams.vocab_size, hparams.embedding_dim)
        if embeddings is not None:
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

        self.crf = CRF(num_tags=hparams.num_classes, batch_first=True)

        self.dropout = nn.Dropout(hparams.dropout)
        self.classifier = nn.Linear(lstm_output_dim, hparams.num_classes)

    def forward(self, x, x_lengths, attention_mask=None):
        embeddings = self.word_embedding(x)
        embeddings = self.dropout(embeddings)
        o, o_lengths = lstm_padded(self.lstm, embeddings, x_lengths)
        o = self.dropout(o)
        output = self.classifier(o)
        return output


class ABSABert(nn.Module):
    def __init__(self, hparams, embeddings=None):
        super(ABSABert, self).__init__()
        self.hparams = hparams
        self.word_embedding = nn.Embedding(hparams.vocab_size, hparams.embedding_dim)
        if embeddings is not None:
            self.word_embedding.weight.data.copy_(embeddings)

        if hparams.use_pos:
            self.pos_embedding = nn.Embedding(
                hparams.pos_vocab_size, hparams.pos_embedding_dim
            )

        bert_output_dim = 768
        self.dropout = nn.Dropout(hparams.dropout)

        lstm_input_size = bert_output_dim + hparams.embedding_dim

        if hparams.use_pos:
            lstm_input_size += hparams.pos_embedding_dim

        self.lstm = nn.LSTM(
            lstm_input_size,
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
        self.classifier = nn.Linear(lstm_output_dim, hparams.num_classes)

        self.crf = CRF(num_tags=hparams.num_classes, batch_first=True)

    def forward(
        self,
        x,
        x_lengths,
        pos_tags,
        attention_mask=None,
        bert_embeddings=None,
    ):
        bert_embeddings = self.dropout(bert_embeddings)
        embeddings = self.word_embedding(x)
        embeddings = self.dropout(embeddings)
        all_embeddings = torch.cat((bert_embeddings, embeddings), dim=-1)
        if self.hparams.use_pos:
            pos_embeddings = self.pos_embedding(pos_tags)
            pos_embeddings = self.dropout(pos_embeddings)
            all_embeddings = torch.cat((all_embeddings, pos_embeddings), dim=1)
        output, _ = lstm_padded(self.lstm, all_embeddings, x_lengths)
        output = self.dropout(output)
        output = self.classifier(output)
        return output
