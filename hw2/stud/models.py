import torch
from torch import nn
from typing import *
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torchcrf import CRF
from transformers import (
    BertModel,
    BertForTokenClassification,
    BertForSequenceClassification,
)


def lstm_padded(
    lstm_layer: nn.Module, x: torch.Tensor, lengths: List[int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    x_packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
    o_packed, _ = lstm_layer(x_packed)
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

        self.crf = CRF(num_tags=hparams.num_classes, batch_first=True)

        self.dropout = nn.Dropout(hparams.dropout)
        self.classifier = nn.Linear(lstm_output_dim, hparams.num_classes)

    def forward(self, x, x_lengths):
        embeddings = self.word_embedding(x)
        embeddings = self.dropout(embeddings)
        o, o_lengths = lstm_padded(self.lstm, embeddings, x_lengths)
        o = self.dropout(o)
        output = self.classifier(o)
        return output


class ABSABert(nn.Module):
    def __init__(self, hparams):
        super(ABSABert, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-cased")
        # self.bert = BertForTokenClassification.from_pretrained(
        #     "bert-base-cased", num_labels=hparams.num_classes
        # )
        bert_output_dim = self.bert.config.hidden_size
        self.dropout = nn.Dropout(hparams.dropout)
        self.classifier = nn.Linear(bert_output_dim, hparams.num_classes)

        self.crf = CRF(num_tags=hparams.num_classes, batch_first=True)

    def forward(self, x, x_lengths):
        attention_mask = torch.ones_like(x)
        for i, l in enumerate(x_lengths):
            attention_mask[i, l:] = 0

        output = self.bert(x, attention_mask)["last_hidden_state"]
        output = self.dropout(output)

        output = self.classifier(output)
        return output

        # output = self.bert(x, attention_mask=attention_mask)["logits"]
        # return output
