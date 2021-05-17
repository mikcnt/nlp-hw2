import re
import nltk
import json
import torch
from collections import Counter
from tqdm import tqdm
from typing import List, Tuple, Any
from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset
from torchtext.vocab import Vocab

nltk.download("punkt")


def tokens_position(sentence: str, target_char_positions: List[int]) -> List[int]:
    """Extract tokens positions from position in string."""
    s_pos, e_pos = target_char_positions
    n_tokens = len(word_tokenize(sentence[s_pos:e_pos]))
    s_token = len(re.findall(r" +", sentence[:s_pos]))
    return list(range(s_token, s_token + n_tokens))


def read_data(path: str) -> Tuple[list, list, list]:
    """Read data from json file."""
    with open(path) as f:
        raw_data = json.load(f)
    return raw_data


def preprocess(raw_data):
    # split data in 2 + 1 lists (3 for restaurants data, 2 for laptops data):
    # for both datasets: sentences (i.e., raw text), targets (i.e., position range, instance, sentiment),
    # for restaurant dataset: categories (i.e., category, sentiment)
    sentences = []
    targets = []
    # TODO: categories
    for d in raw_data:
        # extract tokens
        text = d["text"]
        tokens = word_tokenize(text)
        # possible sentiments are: positive, negative, neutral, conflict
        # `B-sentiment` means that it's the starting token of a sequence (possibly of length 1)
        # `I-sentiment` means that it's following another token (either B- or another I-) of a sequence
        # `0` means that no sentiment is involved with that the token
        sentiments = ["0"] * len(tokens)
        for start_end, instance, sentiment in d["targets"]:
            sentiment_positions = tokens_position(text, start_end)
            for i, s in enumerate(sentiment_positions):
                if i == 0:
                    sentiments[s] = "B-" + sentiment
                else:
                    sentiments[s] = "I-" + sentiment

        sentences.append(tokens)
        targets.append(sentiments)

    return sentences, targets


def build_vocab(data: List[str], specials: List[str], min_freq: int = 1) -> Vocab:
    counter = Counter()
    for i in tqdm(range(len(data))):
        for t in data[i]:
            if t is not None:
                counter[t] += 1
    return Vocab(counter, specials=specials, min_freq=min_freq)


class ABSADataset(Dataset):
    def __init__(
        self,
        sentences: List[List[str]],
        targets: List[Any],
        vocabulary: Vocab,
        sentiments_vocabulary: Vocab,
        max_len: int,
    ) -> None:
        super(ABSADataset, self).__init__()
        self.sentences = sentences
        self.targets = targets
        self.encoded_data = []
        self.vocabulary = vocabulary
        self.sentiments_vocabulary = sentiments_vocabulary
        self.max_len = max_len
        self.index_dataset()

    def encode_text(self, sentence: List[str]) -> List[int]:
        indices = []
        for w in sentence:
            if w in self.vocabulary.stoi:
                indices.append(self.vocabulary[w])
            else:
                indices.append(self.vocabulary.unk_index)
        return indices

    def index_dataset(self):
        assert len(self.sentences) == len(self.targets)
        for i in range(len(self.sentences)):
            sentence = self.sentences[i]
            targets = self.targets[i]
            # encode sentences and targets
            encoded_elem = self.encode_text(sentence)
            encoded_labels = [self.sentiments_vocabulary[t] for t in targets]
            # pad sequences
            encoded_elem = torch.LongTensor(
                self.pad_sequence(encoded_elem, pad_token=self.vocabulary["<pad>"])
            )
            encoded_labels = torch.LongTensor(
                self.pad_sequence(
                    encoded_labels, pad_token=self.sentiments_vocabulary["<pad>"]
                )
            )
            self.encoded_data.append(
                {"inputs": encoded_elem, "outputs": encoded_labels}
            )

    def pad_sequence(self, sequence, pad_token: int) -> List[int]:
        padded_sequence = [pad_token] * self.max_len
        for i, tk_idx in enumerate(sequence):
            if i >= self.max_len:
                break
            padded_sequence[i] = tk_idx
        return padded_sequence

    def __len__(self) -> int:
        return len(self.encoded_data)

    def __getitem__(self, index: int) -> Tuple[torch.LongTensor, torch.LongTensor]:
        return self.encoded_data[index]
