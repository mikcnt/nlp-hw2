import torch
from torch import nn
from typing import *
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torchcrf import CRF


def lstm_padded(
    lstm_layer: nn.Module, x: torch.Tensor, lengths: List[int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    x_packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
    o_packed, _ = lstm_layer(x_packed)
    return pad_packed_sequence(o_packed, batch_first=True)


class ABSAModel(nn.Module):
    def __init__(self, hparams, embeddings=None):
        super(ABSAModel, self).__init__()
        self.hparams = hparams
        self.word_embedding = nn.Embedding(hparams.vocab_size, hparams.embedding_dim)
        if embeddings is not None:
            self.word_embedding.weight.data.copy_(embeddings)

        lstm_input_size = hparams.embedding_dim

        if hparams.use_pos:
            self.pos_embedding = nn.Embedding(
                hparams.pos_vocab_size, hparams.pos_embedding_dim
            )
            self.pos_lstm = nn.LSTM(
                hparams.pos_embedding_dim,
                hparams.hidden_dim,
                bidirectional=hparams.bidirectional,
                num_layers=hparams.num_layers,
                dropout=hparams.dropout if hparams.num_layers > 1 else 0,
                batch_first=True,
            )
            lstm_output_dim = (
                hparams.hidden_dim
                if hparams.bidirectional is False
                else hparams.hidden_dim * 2
            )
            lstm_input_size += lstm_output_dim

        bert_output_dim = 768

        self.bert_lstm = nn.LSTM(
            bert_output_dim,
            hparams.hidden_dim,
            bidirectional=hparams.bidirectional,
            num_layers=hparams.num_layers,
            dropout=hparams.dropout if hparams.num_layers > 1 else 0,
            batch_first=True,
        )

        lstm_output_dim = (
            hparams.hidden_dim
            if hparams.bidirectional is False
            else hparams.hidden_dim * 2
        )
        lstm_input_size += lstm_output_dim

        self.dropout = nn.Dropout(hparams.dropout)

        self.lstm = nn.LSTM(
            lstm_input_size,
            hparams.hidden_dim,
            bidirectional=hparams.bidirectional,
            num_layers=hparams.num_layers,
            dropout=hparams.dropout if hparams.num_layers > 1 else 0,
            batch_first=True,
        )
        lstm_output_dim = (
            hparams.hidden_dim
            if hparams.bidirectional is False
            else hparams.hidden_dim * 2
        )
        self.classifier = nn.Linear(lstm_output_dim, hparams.num_classes)

        self.crf = CRF(num_tags=hparams.num_classes, batch_first=True)

    def forward(self, batch: Dict[str, Union[torch.Tensor, List[int]]]):
        token_indexes = batch["inputs"]
        lengths = batch["lengths"]
        pos_tags = batch["pos_tags"]

        token_embeddings = self.dropout(self.word_embedding(token_indexes))
        output = token_embeddings

        if self.hparams.use_bert:
            bert_embeddings = self.dropout(batch["bert_embeddings"])
            lstm_bert_out, _ = lstm_padded(self.bert_lstm, bert_embeddings, lengths)
            output = torch.cat((output, lstm_bert_out), dim=-1)

        if self.hparams.use_pos:
            pos_embeddings = self.dropout(self.pos_embedding(pos_tags))
            lstm_pos_out, _ = lstm_padded(self.pos_lstm, pos_embeddings, lengths)
            output = torch.cat((output, lstm_pos_out), dim=-1)

        output, _ = lstm_padded(self.lstm, output, lengths)
        output = self.dropout(output)
        output = self.classifier(output)
        return output
