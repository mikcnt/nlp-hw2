import torch
import pytorch_lightning as pl
from dataset import read_data, preprocess, build_vocab, ABSADataset, DataModuleABSA
from models import PlABSAModel

if __name__ == "__main__":
    pl.seed_everything(42)
    # paths
    train_path = "../../data/laptops_train.json"
    dev_path = "../../data/laptops_dev.json"
    # raw data
    train_raw_data = read_data(train_path)
    dev_raw_data = read_data(dev_path)
    # preprocess data: split sentences and labels
    train_data = preprocess(train_raw_data)
    dev_data = preprocess(dev_raw_data)
    # build vocabularies (for both sentences and labels)
    vocabulary = build_vocab(train_data["sentences"], specials=["<pad>", "<unk>"])
    sentiments_vocabulary = build_vocab(train_data["targets"], specials=["<pad>"])

    hparams = {
        "vocab_size": len(vocabulary),
        "hidden_dim": 128,
        "embedding_dim": 100,
        "num_classes": len(sentiments_vocabulary),
        "bidirectional": False,
        "num_layers": 1,
        "dropout": 0.0,
        "embeddings": None,
        "vocabulary": vocabulary,
        "sentiments_vocabulary": sentiments_vocabulary,
    }

    data_module = DataModuleABSA(
        train_data,
        dev_data,
        vocabulary,
        sentiments_vocabulary,
    )
    trainer = pl.Trainer(gpus=1, val_check_interval=1.0, max_epochs=25)
    model = PlABSAModel(**hparams)
    trainer.fit(model, datamodule=data_module)
