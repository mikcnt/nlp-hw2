import torch
import pytorch_lightning as pl
from torchtext.vocab import Vectors
from hw2.stud.dataset import (
    read_data,
    preprocess,
    build_vocab,
    DataModuleABSA,
)
from pl_models import PlABSAModel

if __name__ == "__main__":
    # set seeds for reproducibility
    pl.seed_everything(42)
    # paths
    train_path = "../../data/laptops_train.json"
    dev_path = "../../data/laptops_dev.json"
    # read raw data
    train_raw_data = read_data(train_path)
    dev_raw_data = read_data(dev_path)
    # preprocess data
    train_data = preprocess(train_raw_data)
    dev_data = preprocess(dev_raw_data)
    # build vocabularies (for both sentences and labels)
    vocabulary = build_vocab(train_data["sentences"], specials=["<pad>", "<unk>"])
    sentiments_vocabulary = build_vocab(train_data["targets"], specials=["<pad>"])
    # load pretrained word embeddings
    vectors = Vectors("../../model/glove.6B.300d.txt")
    pretrained_embeddings = torch.randn(len(vocabulary), vectors.dim)
    for i, w in enumerate(vocabulary.itos):
        if w in vectors.stoi:
            vec = vectors.get_vecs_by_tokens(w)
            pretrained_embeddings[i] = vec
    pretrained_embeddings[vocabulary["<pad>"]] = torch.zeros(vectors.dim)
    # define hyper parameters
    hparams = {
        "vocab_size": len(vocabulary),
        "hidden_dim": 128,
        "embedding_dim": vectors.dim,
        "num_classes": len(sentiments_vocabulary),
        "bidirectional": False,
        "num_layers": 1,
        "dropout": 0.5,
        "train_raw_data": train_raw_data,
        "dev_raw_data": dev_raw_data,
        "vocabulary": vocabulary,
        "sentiments_vocabulary": sentiments_vocabulary,
        "lr": 0.001,
        "weight_decay": 0.0,
    }
    # load data
    data_module = DataModuleABSA(
        train_data,
        dev_data,
        vocabulary,
        sentiments_vocabulary,
    )
    # define model
    model = PlABSAModel(pretrained_embeddings, hparams)
    # define trainer and start training
    trainer = pl.Trainer(gpus=1, val_check_interval=1.0, max_epochs=100)
    trainer.fit(model, datamodule=data_module)
