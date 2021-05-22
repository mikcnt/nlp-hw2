import torch
from torchtext.vocab import Vocab
from typing import *


class TokenToSentimentsConverter(object):
    def begin_sentiment(
        self, tokens: List[Tuple[str, str]], token: str, sentiment: str
    ) -> None:
        tokens.append((token, sentiment))

    def inside_sentiment(
        self, tokens: List[Tuple[str, str]], token: str, sentiment: str
    ) -> None:
        if len(tokens) == 0:
            self.begin_sentiment(tokens, token, sentiment)
            return
        last_token, last_sentiment = tokens[-1]
        if last_sentiment != sentiment:
            self.begin_sentiment(tokens, token, sentiment)
        else:
            tokens[-1] = (f"{last_token} {token}", last_sentiment)

    def pick_sentiment(
        self, tokens: List[Tuple[str, str]], token: str, sentiment: str
    ) -> None:
        if sentiment == "0":
            self.begin_sentiment(tokens, token, sentiment)
        elif sentiment.startswith("B-"):
            self.begin_sentiment(tokens, token, sentiment[2:])
        elif sentiment.startswith("I-"):
            self.inside_sentiment(tokens, token, sentiment[2:])

    def compute_sentiments(
        self, input_tokens: List[str], output_sentiments: List[str]
    ) -> List[Tuple[str, str]]:
        tokens2sentiments = []
        for token, sentiment in zip(input_tokens, output_sentiments):
            self.pick_sentiment(tokens2sentiments, token, sentiment)
        return [
            (tk, sentiment) for tk, sentiment in tokens2sentiments if sentiment != "0"
        ]


def forward_sentiments(
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    vocabulary: Vocab,
    sentiments_vocabulary: Vocab,
):
    # convert input and output to list
    input_list: List[List[int]] = input_tensor.tolist()
    output_list: List[List[int]] = output_tensor.tolist()

    # extract tokens and associated sentiments
    input_tokens = [[vocabulary.itos[x] for x in batch] for batch in input_list]

    output_sentiments = [
        [sentiments_vocabulary.itos[x] for x in batch] for batch in output_list
    ]

    return [
        self.sentiments_converter.compute_sentiments(in_tokens, out_sentiments)
        for in_tokens, out_sentiments in zip(input_tokens, output_sentiments)
    ]


if __name__ == "__main__":
    input_tokens = ["I", "love", "pasta", "Ananas", "Pizza"]

    output_sentiments = ["0", "0", "B-positive", "B-negative", "I-negative"]

    tokens2sentiments = compute_sentiments(input_tokens, output_sentiments)
    print(tokens2sentiments)
