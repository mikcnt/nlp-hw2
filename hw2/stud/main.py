import pytorch_lightning as pl
from nltk import TreebankWordTokenizer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from transformers import BertTokenizer

from stud.dataset import (
    read_data,
    json_to_iob,
    json_to_bioes,
    build_vocab,
    DataModuleABSA,
)

from stud.pl_models import PlABSAModel
from stud.utils import save_pickle, compute_pretrained_embeddings

if __name__ == "__main__":
    # --------- REPRODUCIBILITY AND PATHS ---------
    # set seeds for reproducibility
    pl.seed_everything(42)
    # paths
    train_path = "../../data/laptops_train.json"
    dev_path = "../../data/laptops_dev.json"
    # read raw data
    train_raw_data = read_data(train_path)
    dev_raw_data = read_data(dev_path)

    # --------- CONSTANTS ---------
    USE_BERT = False
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
    preprocess_fun = json_to_iob if TAGGING_SCHEMA == "IOB" else json_to_bioes
    train_data = preprocess_fun(train_raw_data, tokenizer=tokenizer)

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
    model = PlABSAModel(
        hparams, vocabularies, embeddings=pretrained_embeddings, tokenizer=tokenizer
    )
    # callbacks
    early_stop_callback = EarlyStopping(
        monitor="f1_evaluation",
        patience=15,
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
