import torch
import pytorch_lightning as pl
import transformers
from nltk import TreebankWordTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from torch import nn, optim
from typing import *
from torchtext.vocab import Vocab

from stud.bert_embedder import BertEmbedder
from stud.metrics import F1SentimentEvaluation
from stud.models import Attention, lstm_padded


class ABSACategoryModel(nn.Module):
    def __init__(self, hparams, embeddings=None):
        super().__init__()
        self.hparams = hparams

        # final BiLSTM layer input size
        lstm_input_size = 0

        # GloVe embeddings
        self.word_embedding = nn.Embedding(hparams.vocab_size, hparams.embedding_dim)
        if embeddings is not None:
            self.word_embedding.weight.data.copy_(embeddings)

        self.glove_linear = nn.Linear(hparams.embedding_dim, hparams.embedding_dim)
        # GloVe multihead attention
        self.attention_glove = Attention(
            d_model=hparams.embedding_dim, nhead=12, dropout=0.1
        )
        lstm_input_size += hparams.embedding_dim

        # BERT embeddings
        bert_output_dim = 768
        self.bert_embedder = BertEmbedder()
        self.bert_linear = nn.Linear(bert_output_dim, bert_output_dim)
        # BERT multihead attention
        self.attention_bert = Attention(d_model=bert_output_dim, nhead=24, dropout=0.1)
        lstm_input_size += bert_output_dim

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

        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def lstm_last(self, x, lengths):
        out, _ = lstm_padded(self.lstm, x, lengths)
        out_forward = out[:, -1, : self.hparams.hidden_dim]
        if not self.hparams.bidirectional:
            return out_forward
        else:
            out_reverse = out[:, 0, self.hparams.hidden_dim :]
            return torch.cat((out_forward, out_reverse), dim=1)

    def forward(self, batch: Dict[str, Union[torch.Tensor, List[int]]]):
        # inputs
        token_indexes = batch["token_indexes"]
        lengths = batch["lengths"]

        # compute glove embeddings
        glove_embeddings = self.word_embedding(token_indexes)
        glove_embeddings = self.dropout(glove_embeddings)
        glove_embeddings = self.relu(self.glove_linear(glove_embeddings))

        glove_embeddings = self.attention_glove(
            glove_embeddings.transpose(0, 1),
            key_padding_mask=~batch["attention_mask"],
        ).transpose(0, 1)
        glove_embeddings = self.dropout(glove_embeddings)

        output = glove_embeddings

        # compute BERT embeddings
        if self.hparams.use_bert:
            bert_embeddings = self.bert_embedder(batch)
            bert_embeddings = self.dropout(bert_embeddings)
            bert_embeddings = self.relu(self.bert_linear(bert_embeddings))
            bert_embeddings = self.attention_bert(
                bert_embeddings.transpose(0, 1),
                key_padding_mask=~batch["attention_mask"],
            ).transpose(0, 1)
            bert_embeddings = self.dropout(bert_embeddings)
            # concatenate BERT embeddings to GloVe embeddings
            output = torch.cat((output, bert_embeddings), dim=-1)

        # apply BiLSTM
        output = self.lstm_last(output, lengths)

        # classification head
        output = self.classifier(output)
        output_softmax = self.logsoftmax(output)

        # think the output as 5 independent classifications
        return output_softmax.reshape(
            -1,
            self.hparams.category_vocab_size,
            self.hparams.category_polarity_vocab_size + 1,
        )


class PlABSACategoryModel(pl.LightningModule):
    def __init__(
        self,
        hparams: Dict[str, Any],
        vocabularies: Dict[str, Vocab],
        embeddings: torch.Tensor,
        tokenizer: TreebankWordTokenizer,
        train=True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(hparams)
        self.category_vocabulary = vocabularies["category_vocabulary"]
        self.category_polarity_vocabulary = vocabularies["category_polarity_vocabulary"]
        self.tokenizer = tokenizer
        self.detokenizer = TreebankWordDetokenizer()
        self.loss_function = nn.NLLLoss()

        # use f1 metrics only during training
        if train:
            self.f1_extraction_t = F1SentimentEvaluation(
                device=self.device, mode="Category Extraction"
            )
            self.f1_evaluation_t = F1SentimentEvaluation(
                device=self.device, mode="Category Sentiment"
            )

            self.f1_extraction_v = F1SentimentEvaluation(
                device=self.device, mode="Category Extraction"
            )
            self.f1_evaluation_v = F1SentimentEvaluation(
                device=self.device, mode="Category Sentiment"
            )

        self.model = ABSACategoryModel(self.hparams, embeddings)

    def forward(
        self, batch: Dict[str, Union[torch.Tensor, List[int]]]
    ) -> Dict[str, Union[torch.Tensor, List]]:

        logits = self.model(batch)
        text_predictions = self.process_category_prediction(logits)
        return {
            "logits": logits,
            "text_predictions": text_predictions,
        }

    def predict(self, batch: Dict[str, torch.Tensor]) -> List[Dict[str, Any]]:
        """This method returns the predictions in their original form, as found in the dataset.
        This way, we can compute the f1 score between the predictions and the ground truths.
        Use this function during inference time."""
        text_predictions = self(batch)["text_predictions"]
        return text_predictions

    def step(self, batch: Dict[str, torch.Tensor], phase: str) -> Dict[str, Any]:
        output = self(batch)
        logits = output["logits"]
        # for each category, get the polarity that has value = 1 (can also be "O", meaning that
        # the sentence doesn't have that particulary category in the ground truth)
        labels = torch.argmax(batch["categories"], dim=-1)
        text_predictions = output["text_predictions"]
        text_gt = batch["raw"]

        step_output = {}

        # compute loss:
        # the output of our model can be thought as it was the vector containing 5 independent predictions, for each category
        # for this reason, we are going to compare, for each category, the logits and the labels, then we sum the results
        total_loss = 0
        for i in range(logits.shape[1]):
            total_loss += self.loss_function(logits[:, i], labels[:, i])
        # finally, we normalize the loss by the number of categories
        step_output["loss"] = 1 / self.hparams.category_vocab_size * total_loss

        if phase == "train":
            f1_extraction = self.f1_extraction_t
            f1_evaluation = self.f1_evaluation_t
        else:
            f1_extraction = self.f1_extraction_v
            f1_evaluation = self.f1_evaluation_v

        step_output["f1_extraction"] = f1_extraction(text_predictions, text_gt)
        step_output["f1_evaluation"] = f1_evaluation(text_predictions, text_gt)

        return step_output

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: Optional[int]
    ) -> Dict[str, torch.Tensor]:
        step_output = self.step(batch, phase="train")
        train_loss = step_output["loss"]
        f1_extraction = step_output["f1_extraction"]
        f1_evaluation = step_output["f1_evaluation"]
        # Log loss
        self.log_dict(
            {
                "train_loss": train_loss,
            },
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        # Return loss to update weights
        return {
            "loss": train_loss,
            "f1_extraction": f1_extraction,
            "f1_evaluation": f1_evaluation,
        }

    def training_epoch_end(self, outputs):
        f1_extraction = self.f1_extraction_t.compute()
        f1_evaluation = self.f1_evaluation_t.compute()
        self.log_dict(
            {
                "f1_extraction_train": f1_extraction,
                "f1_evaluation_train": f1_evaluation,
            },
            prog_bar=False,
        )
        self.f1_extraction_t.reset()
        self.f1_evaluation_t.reset()

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: Optional[int]
    ) -> Dict[str, torch.Tensor]:
        step_output = self.step(batch, phase="val")
        val_loss = step_output["loss"]
        f1_extraction = step_output["f1_extraction"]
        f1_evaluation = step_output["f1_evaluation"]
        # Log loss and f1 score
        self.log_dict(
            {
                "val_loss": val_loss,
            },
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        return {"f1_extraction": f1_extraction, "f1_evaluation": f1_evaluation}

    def validation_epoch_end(self, outputs):
        f1_extraction = self.f1_extraction_v.compute()
        f1_evaluation = self.f1_evaluation_v.compute()
        self.log_dict(
            {"f1_extraction_val": f1_extraction, "f1_evaluation_val": f1_evaluation},
            prog_bar=True,
        )
        self.f1_extraction_v.reset()
        self.f1_evaluation_v.reset()

    def configure_optimizers(self) -> optim.Optimizer:
        # dynamically instantiate optimizer
        optimizer = eval(self.hparams.optimizer)
        return optimizer(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

    def process_category_prediction(self, predictions):
        """Postprocess the logits to extract the predictions in the same format of the dataset."""
        processed_outputs = []
        # `predictions` is a batch, so we iterate in it
        for pred in predictions:
            # given the logits, `max_pred` are, for each category, the predicted polarity
            max_pred = torch.argmax(pred, dim=-1)

            # `no_category_idx` represents the index for the non-polarity "O"
            # which we arbitrarly choose to be the last one
            no_category_idx = len(self.category_polarity_vocabulary)

            # now we start to translate the predictions in a more readable format
            output = {"categories": []}
            # for each category (represented by the index `i`)
            # and its predicted polarity (represented by the index `pol_idx`)
            for i, pol_idx in enumerate(max_pred):
                # if the predicted polarity for the category is the non-polarity, we skip the category
                # since it means that the model predicted that the sentence doesn't have this category
                if pol_idx == no_category_idx:
                    continue
                # we retrieve the category with the vocabulary containing all the categories
                category = self.category_vocabulary.itos[i]
                # we retrieve the polarity with the vocabulary containing all the categories
                polarity = self.category_polarity_vocabulary.itos[pol_idx]
                # we append the prediction
                output["categories"].append([category, polarity])

            # once the sentence is finished, we append the predictions of this sentence
            # and proceed with the next on the batch
            processed_outputs.append(output)

        return processed_outputs
