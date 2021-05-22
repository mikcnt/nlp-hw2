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
    # load pretrained word embeddings
    vectors = Vectors("../../model/glove.6B.300d.txt")
    pretrained_embeddings = torch.randn(len(vocabulary), vectors.dim)
    for i, w in enumerate(vocabulary.itos):
        if w in vectors.stoi:
            vec = vectors.get_vecs_by_tokens(w)
            pretrained_embeddings[i] = vec
    pretrained_embeddings[vocabulary["<pad>"]] = torch.zeros(vectors.dim)
    hparams = {
        "vocab_size": len(vocabulary),
        "hidden_dim": 256,
        "embedding_dim": vectors.dim,
        "num_classes": len(sentiments_vocabulary),
        "bidirectional": True,
        "num_layers": 2,
        "dropout": 0.5,
        "train_raw_data": train_raw_data,
        "dev_raw_data": dev_raw_data,
        "vocabulary": vocabulary,
        "sentiments_vocabulary": sentiments_vocabulary,
    }

    data_module = DataModuleABSA(
        train_data,
        dev_data,
        vocabulary,
        sentiments_vocabulary,
    )
    trainer = pl.Trainer(gpus=1, val_check_interval=1.0, max_epochs=100)
    model = PlABSAModel(pretrained_embeddings, hparams)
    trainer.fit(model, datamodule=data_module)
