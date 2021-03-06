import numpy as np
import torch
from matplotlib import pyplot as plt
from nltk import TreebankWordTokenizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader
from tqdm import tqdm

from stud.dataset import (
    read_data,
    ABSADataset,
)

from stud.pl_models import PlABSAModel
from stud.utils import (
    compute_pretrained_embeddings,
    load_pickle,
    pad_collate,
)

"""Evaluation script for the AB task of a given model. Used to produce confusion matrices."""

ab_iob_polarities = [
    "<pad>",
    "O",
    "B-positive",
    "B-negative",
    "I-positive",
    "B-neutral",
    "I-negative",
    "I-neutral",
    "B-conflict",
    "I-conflict",
]
ab_polarities = ["O", "positive", "negative", "neutral", "conflict"]


def remove_iob(array):
    new_array = np.zeros_like(array)

    for i, pol in enumerate(ab_polarities):
        if pol == "O":
            continue
        polarity_indexes = [
            ab_iob_polarities.index(x) for x in [f"B-{pol}", f"I-{pol}"]
        ]
        for idx in polarity_indexes:
            new_array[array == idx] = i
    return new_array


if __name__ == "__main__":
    # paths
    restaurants_dev_path = "../../data/restaurants_dev.json"
    laptops_dev_path = "../../data/laptops_dev.json"

    # read raw data
    restaurants_dev_raw_data = read_data(restaurants_dev_path)
    laptops_dev_raw_data = read_data(laptops_dev_path)
    dev_raw_data = restaurants_dev_raw_data + laptops_dev_raw_data

    model_path = "../../model/M0.ckpt"
    vocabulary_path = "../../model/vocabulary.pkl"
    sentiments_vocabulary_path = "../../model/sentiments_vocabulary.pkl"
    pos_vocabulary_path = "../../model/pos_vocabulary.pkl"

    # read vocabularies
    vocabulary = load_pickle(vocabulary_path)
    sentiments_vocabulary = load_pickle(sentiments_vocabulary_path)
    pos_vocabulary = load_pickle(pos_vocabulary_path)
    vocabularies = {
        "vocabulary": vocabulary,
        "sentiments_vocabulary": sentiments_vocabulary,
        "pos_vocabulary": pos_vocabulary,
    }

    # load pretrained embeddings
    pretrained_embeddings = compute_pretrained_embeddings(
        path="../../model/glove.6B.300d.txt",
        cache="../../model/.vector_cache/",
        vocabulary=vocabulary,
    )

    # load model
    model = PlABSAModel.load_from_checkpoint(
        model_path,
        map_location="cuda",
        vocabularies=vocabularies,
        embeddings=pretrained_embeddings,
        tokenizer=TreebankWordTokenizer(),
    )
    model.eval()

    # load data
    dataset = ABSADataset(
        dev_raw_data,
        vocabularies,
        tagging_schema="IOB",
        tokenizer=TreebankWordTokenizer(),
        save_categories=False,
    )

    # create dataloader
    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=pad_collate,
    )

    all_labels = []
    all_predictions = []
    for i, batch in enumerate(tqdm(loader)):
        with torch.no_grad():
            output = model(batch)
            labels = batch["labels"].reshape(-1)
            predictions = output["predictions"].reshape(-1)
            all_labels.append(labels)
            all_predictions.append(predictions)

    all_labels = torch.cat(all_labels).numpy()
    all_predictions = torch.cat(all_predictions).numpy()

    # remove padded elements
    all_labels_no_pad = all_labels[all_labels != 0]
    all_predictions_no_pad = all_predictions[all_labels != 0]

    # remove IOB tags (e.g., B-pos -> pos, I-pos -> pos, etc.)
    all_labels_no_pad_new = remove_iob(all_labels_no_pad)
    all_predictions_no_pad_new = remove_iob(all_predictions_no_pad)

    # compute confusion matrix
    cm = confusion_matrix(all_labels_no_pad_new, all_predictions_no_pad_new)
    # normalize confusion matrix
    cm = cm / cm.astype(float).sum(axis=1)

    # save img
    fig, ax = plt.subplots(dpi=200)
    ConfusionMatrixDisplay(cm, display_labels=ab_polarities).plot(cmap="Blues", ax=ax)
    plt.savefig("confusion_matrix.png")
