import pytorch_lightning as pl
import torch.optim.optimizer
from nltk import TreebankWordTokenizer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from transformers import BertTokenizer

from stud.dataset import (
    read_data,
    build_vocab,
    DataModuleABSA,
    preprocess,
)

from stud.pl_models import PlABSAModel
from stud.utils import save_pickle, compute_pretrained_embeddings

if __name__ == "__main__":
    # --------- REPRODUCIBILITY AND PATHS ---------
    # set seeds for reproducibility
    pl.seed_everything(42)
    # paths
    restaurants_train_path = "../../data/restaurants_train.json"
    restaurants_dev_path = "../../data/restaurants_dev.json"
    laptops_train_path = "../../data/laptops_train.json"
    laptops_dev_path = "../../data/laptops_dev.json"
    # read raw data
    restaurants_train_raw_data = read_data(restaurants_train_path)
    restaurants_dev_raw_data = read_data(restaurants_dev_path)
    laptops_train_raw_data = read_data(laptops_train_path)
    laptops_dev_raw_data = read_data(laptops_dev_path)
    # train_raw_data = restaurants_train_raw_data + laptops_train_raw_data
    # dev_raw_data = restaurants_dev_raw_data + laptops_dev_raw_data
    train_raw_data = laptops_train_raw_data
    dev_raw_data = laptops_dev_raw_data

    # --------- CONSTANTS ---------
    USE_BERT = True
    TAGGING_SCHEMA = "BIOES"
    assert TAGGING_SCHEMA in [
        "IOB",
        "BIOES",
    ], "Tagging schema should be either IOB or BIOES."

    # --------- TOKENIZER ---------
    tokenizer = (
        BertTokenizer.from_pretrained("bert-base-cased")
        if USE_BERT
        else TreebankWordTokenizer()
    )

    # --------- TAG DATA: IOB OR BIOES ---------
    train_data = preprocess(
        train_raw_data, tokenizer=tokenizer, tagging_schema=TAGGING_SCHEMA
    )

    # --------- VOCABULARIES ---------
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

    # --------- PRETRAINED EMBEDDINGS ---------
    pretrained_embeddings = compute_pretrained_embeddings(
        path="../../model/glove.6B.300d.txt",
        cache="../../model/.vector_cache/",
        vocabulary=vocabulary,
    )

    # --------- HYPERPARAMETERS ---------
    hparams = {
        "vocab_size": len(vocabulary),
        "hidden_dim": 300,
        "embedding_dim": pretrained_embeddings.shape[1],
        "num_classes": len(sentiments_vocabulary),
        "bidirectional": True,
        "num_layers": 2,
        "dropout": 0.5,
        "lr": 2e-5 if USE_BERT else 1e-3,
        "weight_decay": 0.0,
        "batch_size": 16,
        "use_bert": USE_BERT,
        "tagging_schema": TAGGING_SCHEMA,
        "use_crf": False,
    }

    # --------- TRAINER ---------
    # load data
    data_module = DataModuleABSA(
        train_raw_data,
        dev_raw_data,
        vocabulary,
        sentiments_vocabulary,
        batch_size=hparams["batch_size"],
        tagging_schema=TAGGING_SCHEMA,
        tokenizer=tokenizer,
    )
    # define model
    model = PlABSAModel(
        hparams, vocabularies, embeddings=pretrained_embeddings, tokenizer=tokenizer
    )
    # callbacks
    early_stop_callback = EarlyStopping(
        monitor="f1_extraction",
        patience=1000,
        verbose=False,
        mode="max",
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath="./saved_checkpoints",
        filename="{epoch}_{f1_extraction:.4f}_{f1_evaluation:.4f}",
        monitor="f1_evaluation",
        save_top_k=0,
        save_last=False,
        mode="max",
    )

    # logger
    overfit_batches = 0
    wandb_logger = WandbLogger(offline=True, project="nlp-hw2")

    # define trainer and train
    trainer = pl.Trainer(
        gpus=1,
        val_check_interval=1.0,
        max_epochs=1000,
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=0,
        logger=wandb_logger,
        overfit_batches=overfit_batches,
        resume_from_checkpoint=None,
    )
    trainer.fit(model, datamodule=data_module)
