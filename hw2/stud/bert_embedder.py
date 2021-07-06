from typing import List

import torch
import pytorch_lightning as pl
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, AutoTokenizer
import numpy as np

from itertools import groupby


class BertEmbedder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        transformer_type = "bert-base-cased"
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_type)
        self.model = AutoModel.from_pretrained(
            transformer_type, output_hidden_states=True, return_dict=True
        ).to(self.device)
        self.model.eval()

        self.average_last_k_layers = 3

    def count_subsequent(self, sequence):
        lengths = [
            sum(1 for _ in group) for idx, group in groupby(sequence) if idx is not None
        ]
        # CLS and SEP
        lengths.insert(0, 1)
        lengths.append(1)
        return lengths

    def forward(self, batch):
        tokens = batch["tokens"]
        attention_mask = batch["attention_mask"]
        lengths = batch["lengths"]

        encoding = self.tokenizer(
            tokens,
            return_tensors="pt",
            padding=True,
            truncation=True,
            is_split_into_words=True,
        )

        # calculate the BPE-length of each token
        sentence_word_lengths = [
            self.count_subsequent(sample.word_ids) for sample in encoding.encodings
        ]

        # transform the list of word lengths into a list of cumulative offsets
        offsets = [
            [0] + np.cumsum(word_lengths).tolist()
            for word_lengths in sentence_word_lengths
        ]

        max_offsets_seq = max(len(seq) for seq in offsets)

        *bpe_indices, bpe_values = list(
            zip(
                *[
                    (i_sentence, i, offset, 1 / (end_offset - start_offset))
                    for i_sentence, sentence_offsets in enumerate(offsets)
                    for i, (start_offset, end_offset) in enumerate(
                        zip(sentence_offsets[:-1], sentence_offsets[1:])
                    )
                    for offset in range(start_offset, end_offset)
                ]
            )
        )

        bpe_indices = torch.LongTensor(bpe_indices)
        bpe_values = torch.FloatTensor(bpe_values)

        bpe_weights_size = torch.Size(
            (
                encoding.input_ids.size(0),
                max_offsets_seq - 1,
                encoding.input_ids.size(1),
            )
        )

        zeros = torch.zeros(encoding.input_ids.size(0), 1, device=attention_mask.device)

        # add exclusion of CLS and SEP
        encoding_mask = torch.cat([zeros, attention_mask, zeros], dim=1).bool()

        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        transformer_out = self.model(**encoding)

        bert_embeddings: torch.Tensor = torch.stack(
            transformer_out["hidden_states"][-self.average_last_k_layers :],
            dim=2,
        ).sum(dim=2)

        bpe_weights = torch.sparse.FloatTensor(
            bpe_indices, bpe_values, bpe_weights_size
        ).to(self.device)

        bert_embeddings = torch.bmm(bpe_weights, bert_embeddings)

        # padding and special tokens (CLS & SEP) removal
        bert_embeddings = bert_embeddings[encoding_mask]

        # bert_embeddings is now 1-D, we need to split it again
        bert_embeddings: List[torch.Tensor] = torch.split(bert_embeddings, lengths)

        # pad bert_embeddings
        bert_embeddings_padded = pad_sequence(
            bert_embeddings, batch_first=True, padding_value=0
        )

        return bert_embeddings_padded
