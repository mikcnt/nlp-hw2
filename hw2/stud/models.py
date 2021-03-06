import torch
from torch import nn
from typing import *
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torchcrf import CRF

from stud.bert_embedder import BertEmbedder


def lstm_padded(
    lstm_layer: nn.Module, x: torch.Tensor, lengths: List[int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    x_packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
    o_packed, _ = lstm_layer(x_packed)
    return pad_packed_sequence(o_packed, batch_first=True)


class Attention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, src, attn_mask=None, key_padding_mask=None):
        src2, _ = self.self_attn(
            src, src, src, attn_mask=attn_mask, key_padding_mask=key_padding_mask
        )
        src = self.dropout(src2)
        src = self.layernorm(src)
        return src


class ABSAModel(nn.Module):
    def __init__(self, hparams, embeddings=None):
        super().__init__()
        self.hparams = hparams

        # GloVe embeddings
        self.word_embedding = nn.Embedding(hparams.vocab_size, hparams.embedding_dim)
        if embeddings is not None:
            self.word_embedding.weight.data.copy_(embeddings)

        lstm_input_size = 0
        self.glove_linear = nn.Linear(hparams.embedding_dim, hparams.embedding_dim)
        if self.hparams.use_attention:
            self.attention_glove = Attention(
                d_model=hparams.embedding_dim, nhead=12, dropout=0.1
            )
        lstm_input_size += hparams.embedding_dim

        # BERT embeddings
        if self.hparams.use_bert:
            bert_output_dim = 768
            self.bert_embedder = BertEmbedder()
            self.bert_linear = nn.Linear(bert_output_dim, bert_output_dim)
            if self.hparams.use_attention:
                self.attention_bert = Attention(
                    d_model=bert_output_dim, nhead=24, dropout=0.1
                )
            lstm_input_size += bert_output_dim

        # POS embeddings
        if self.hparams.use_pos:
            self.pos_embedding = nn.Embedding(
                hparams.pos_vocab_size, hparams.pos_embedding_dim
            )
            self.pos_linear = nn.Linear(
                hparams.pos_embedding_dim, hparams.pos_embedding_dim
            )
            if self.hparams.use_attention:
                self.attention_pos = Attention(
                    d_model=hparams.pos_embedding_dim, nhead=12, dropout=0.1
                )
            lstm_input_size += hparams.pos_embedding_dim

        # stacked BiLSTM layers
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

        # classification head
        self.classifier = nn.Linear(lstm_output_dim, hparams.num_classes)
        self.dropout = nn.Dropout(hparams.dropout)
        self.relu = nn.ReLU()

        # CRF
        self.crf = CRF(num_tags=hparams.num_classes, batch_first=True)

    def forward(self, batch: Dict[str, Union[torch.Tensor, List[int]]]):
        token_indexes = batch["token_indexes"]
        lengths = batch["lengths"]
        pos_tags = batch["pos_tags"]

        # compute GloVe embeddings
        glove_embeddings = self.word_embedding(token_indexes)
        glove_embeddings = self.dropout(glove_embeddings)
        glove_embeddings = self.relu(self.glove_linear(glove_embeddings))

        if self.hparams.use_attention:
            glove_embeddings = self.attention_glove(
                glove_embeddings.transpose(0, 1),
                key_padding_mask=~batch["attention_mask"],
            ).transpose(0, 1)
        glove_embeddings = self.dropout(glove_embeddings)

        output = glove_embeddings

        # compute BERT embeddings
        if self.hparams.use_bert:
            if not self.hparams.finetune_bert:
                with torch.no_grad():
                    bert_embeddings = self.bert_embedder(batch)
            else:
                bert_embeddings = self.bert_embedder(batch)
            bert_embeddings = self.dropout(bert_embeddings)
            bert_embeddings = self.relu(self.bert_linear(bert_embeddings))
            if self.hparams.use_attention:
                bert_embeddings = self.attention_bert(
                    bert_embeddings.transpose(0, 1),
                    key_padding_mask=~batch["attention_mask"],
                ).transpose(0, 1)
            bert_embeddings = self.dropout(bert_embeddings)
            # concatenate BERT embeddings to the GloVe embeddings
            output = torch.cat((output, bert_embeddings), dim=-1)

        # compute POS embeddings
        if self.hparams.use_pos:
            pos_embeddings = self.pos_embedding(pos_tags)
            pos_embeddings = self.dropout(pos_embeddings)
            pos_embeddings = self.relu(self.pos_linear(pos_embeddings))
            if self.hparams.use_attention:
                pos_embeddings = self.attention_pos(
                    pos_embeddings.transpose(0, 1),
                    key_padding_mask=~batch["attention_mask"],
                ).transpose(0, 1)
            pos_embeddings = self.dropout(pos_embeddings)
            # concatenate POS embeddings to the GloVe + BERT embeddings
            output = torch.cat((output, pos_embeddings), dim=-1)

        # pass output to the BiLSTM
        output, _ = lstm_padded(self.lstm, output, lengths)
        # classification head
        output = self.classifier(output)
        return output
