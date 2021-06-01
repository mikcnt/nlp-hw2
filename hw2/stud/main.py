import torch
import pytorch_lightning as pl
from nltk import TreebankWordTokenizer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torchtext.vocab import Vectors
from transformers import BertTokenizer

from stud.dataset import (
    read_data,
    json_to_iob,
    json_to_bioes,
    build_vocab,
    DataModuleABSA,
)

from stud.pl_models import PlABSAModel
from stud.utils import save_pickle

if __name__ == "__main__":
    # --------- NECESSARY BOILER ---------
    # set seeds for reproducibility
    pl.seed_everything(42)
    # paths
    train_path = "../../data/laptops_train.json"
    dev_path = "../../data/laptops_dev.json"
    # read raw data
    train_raw_data = read_data(train_path)
    dev_raw_data = read_data(dev_path)

    # --------- DATA ---------
    USE_BERT = False
    TAGGING_SCHEMA = "IOB"
    tokenizer = (
        BertTokenizer.from_pretrained("bert-base-cased")
        if USE_BERT
        else TreebankWordTokenizer()
    )
    # preprocess data: convert json format to either IOB or BIOES
    assert TAGGING_SCHEMA in [
        "IOB",
        "BIOES",
    ], "Tagging schema should be either IOB or BIOES."
    preprocess = json_to_iob if TAGGING_SCHEMA == "IOB" else json_to_bioes
    train_data = preprocess(train_raw_data, tokenizer=tokenizer)
    dev_data = preprocess(dev_raw_data, tokenizer=tokenizer)
    # build vocabularies (for both sentences and labels)
    vocabulary = build_vocab(
        train_data["sentences"], specials=["<pad>", "<unk>"], min_freq=2
    )
    sentiments_vocabulary = build_vocab(train_data["targets"], specials=["<pad>"])
    vocabularies = {
        "vocabulary": vocabulary,
        "sentiments_vocabulary": sentiments_vocabulary,
    }
    # save vocabularies to file
    save_pickle(vocabulary, "../../model/vocabulary.pkl")
    save_pickle(sentiments_vocabulary, "../../model/sentiments_vocabulary.pkl")

    # --------- EMBEDDINGS ---------
    # load pretrained word embeddings
    vectors = Vectors(
        "../../model/glove.6B.300d.txt", cache="../../model/.vector_cache/"
    )
    pretrained_embeddings = torch.randn(len(vocabulary), vectors.dim)
    for i, w in enumerate(vocabulary.itos):
        if w in vectors.stoi:
            vec = vectors.get_vecs_by_tokens(w)
            pretrained_embeddings[i] = vec
    pretrained_embeddings[vocabulary["<pad>"]] = torch.zeros(vectors.dim)

    # --------- HYPERPARAMETERS ---------
    hparams = {
        "vocab_size": len(vocabulary),
        "hidden_dim": 300,
        "embedding_dim": vectors.dim,
        "num_classes": len(sentiments_vocabulary),
        "bidirectional": True,
        "num_layers": 3,
        "dropout": 0.5,
        "lr": 0.001,
        "weight_decay": 0.0,
        "use_bert": USE_BERT,
        "tagging_schema": TAGGING_SCHEMA,
    }

    # --------- TRAINER ---------
    # load data
    data_module = DataModuleABSA(
        train_raw_data,
        dev_raw_data,
        vocabulary,
        sentiments_vocabulary,
        tagging_schema=TAGGING_SCHEMA,
        tokenizer=tokenizer,
    )
    # define model
    model = PlABSAModel(hparams, vocabularies, embeddings=pretrained_embeddings)

    # callbacks
    early_stop_callback = EarlyStopping(
        monitor="f1_extraction",
        min_delta=0.00,
        patience=1000,
        verbose=False,
        mode="max",
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath="./saved_checkpoints",
        filename="{epoch}_{f1_extraction:.4f}_{f1_evaluation:.4f}",
        monitor="f1_extraction",
        save_top_k=0,
        save_last=False,
        mode="max",
    )

    # define trainer and train
    trainer = pl.Trainer(
        gpus=1,
        val_check_interval=1.0,
        max_epochs=1000,
        callbacks=[early_stop_callback, checkpoint_callback],
        num_sanity_val_steps=0,
        # overfit_batches=1,
    )
    trainer.fit(model, datamodule=data_module)
