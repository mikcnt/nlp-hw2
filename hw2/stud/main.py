import os
import pytorch_lightning as pl
from nltk import TreebankWordTokenizer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from stud.dataset import (
    read_data,
    build_vocab,
    DataModuleABSA,
    preprocess,
)

from stud.pl_models import PlABSAModel
from stud.utils import save_pickle, compute_pretrained_embeddings

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
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
    train_raw_data = restaurants_train_raw_data + laptops_train_raw_data
    dev_raw_data = restaurants_dev_raw_data + laptops_dev_raw_data

    # --------- CONSTANTS ---------
    USE_BERT = True
    TAGGING_SCHEMA = "IOB"

    # --------- TOKENIZER ---------
    tokenizer = TreebankWordTokenizer()

    # --------- TAG DATA: IOB OR BIOES ---------
    train_data = preprocess(
        train_raw_data, tokenizer=tokenizer, tagging_schema=TAGGING_SCHEMA
    )

    # --------- VOCABULARIES ---------
    vocabulary = build_vocab(
        train_data["sentences"], specials=["<pad>", "<unk>"], min_freq=2
    )
    sentiments_vocabulary = build_vocab(train_data["targets"], specials=["<pad>"])
    pos_vocabulary = build_vocab(train_data["pos_tags"], specials=["<pad>", "<unk>"])
    vocabularies = {
        "vocabulary": vocabulary,
        "sentiments_vocabulary": sentiments_vocabulary,
        "pos_vocabulary": pos_vocabulary,
    }
    # save vocabularies to file
    save_pickle(vocabulary, "../../model/vocabulary.pkl")
    save_pickle(sentiments_vocabulary, "../../model/sentiments_vocabulary.pkl")

    # --------- PRETRAINED EMBEDDINGS ---------
    pretrained_embeddings = compute_pretrained_embeddings(
        path="../../model/wiki-news-300d-1M.vec",
        cache="../../model/.vector_cache/",
        vocabulary=vocabulary,
    )

    # --------- HYPERPARAMETERS ---------
    hparams = {
        # general parameters
        "batch_size": 16,
        "num_classes": len(sentiments_vocabulary),
        "tagging_schema": TAGGING_SCHEMA,
        "use_bert": USE_BERT,
        "use_crf": False,
        "use_pos": False,
        # optimizer parameters
        "optimizer": "transformers.AdamW",
        "lr": 2e-5,
        "weight_decay": 0.0,
        # vocabularies
        "vocab_size": len(vocabulary),
        "pos_vocab_size": len(pos_vocabulary),
        # network parameters
        "num_layers": 2,
        "hidden_dim": 300,
        "bidirectional": True,
        "dropout": 0.5,
        "pos_embedding_dim": 300,
        # Glove embeddings
        "embedding_dim": pretrained_embeddings.shape[1],
    }

    # --------- TRAINER ---------
    # load data
    data_module = DataModuleABSA(
        train_raw_data,
        dev_raw_data,
        vocabularies,
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
        callbacks=[early_stop_callback, checkpoint_callback],
        num_sanity_val_steps=0,
        logger=wandb_logger,
        overfit_batches=overfit_batches,
        resume_from_checkpoint=None,
    )
    trainer.fit(model, datamodule=data_module)
