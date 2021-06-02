from typing import Union, List, Dict, Any, Optional

from nltk import TreebankWordTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from transformers import BertTokenizer


class TokenToSentimentsConverter(object):
    def __init__(
        self,
        vocabulary,
        sentiments_vocabulary,
        tokenizer: Union[TreebankWordTokenizer, BertTokenizer],
        tagging_schema: str,
    ):
        self.vocabulary = vocabulary
        self.sentiments_vocabulary = sentiments_vocabulary
        self.tokenizer = tokenizer
        self.tagging_schema = tagging_schema

    def tokens_position(
        self,
        sentence: str,
        target_char_positions: List[int],
    ) -> List[int]:
        """Extract tokens with sentiments associated positions from position in string."""
        s_pos, e_pos = target_char_positions
        tokens_between_positions = self.tokenizer.tokenize(sentence[s_pos:e_pos])
        n_tokens = len(tokens_between_positions)
        s_token = len(self.tokenizer.tokenize(sentence[:s_pos]))
        return list(range(s_token, s_token + n_tokens))

    def json_to_tags(self, example):

        text = example["text"]
        targets = example["targets"]
        tokens = self.tokenizer.tokenize(text)

        sentiments = ["O"] * len(tokens)
        for start_end, instance, sentiment in targets:
            sentiment_positions = self.tokens_position(
                text,
                start_end,
            )
            for i, s in enumerate(sentiment_positions):
                if self.tagging_schema == "IOB":
                    if i == 0:
                        sentiments[s] = "B-" + sentiment
                    else:
                        sentiments[s] = "I-" + sentiment
                elif self.tagging_schema == "BIOES":
                    if len(sentiment_positions) == 1:
                        sentiments[s] = "S-" + sentiment
                    elif i == 0:
                        sentiments[s] = "B-" + sentiment
                    elif i == len(sentiment_positions) - 1:
                        sentiments[s] = "E-" + sentiment
                    else:
                        sentiments[s] = "I-" + sentiment
        return sentiments

    def preprocess(
        self,
        raw_data: List[Dict[str, Any]],
        train: bool = True,
    ):
        """Convert data in JSON format to IOB schema."""
        processed_data = {"sentences": [], "targets": []}
        for d in raw_data:
            text = d["text"]
            tokens = self.tokenizer.tokenize(text)
            processed_data["sentences"].append(tokens)
            if train:
                sentiments = self.json_to_tags(d)
                processed_data["targets"].append(sentiments)

        return processed_data

    def tags_to_json(self, tokens, sentiments):

        tokens2sentiments = []
        for i, (token, sentiment) in enumerate(zip(tokens, sentiments)):
            if self.tagging_schema == "IOB":
                # just ignore outside sentiments for the moment
                if sentiment == "O":
                    tokens2sentiments.append([[token], sentiment])

                # if it is starting, then we expect something after, so we append the first one
                if sentiment.startswith("B-"):
                    tokens2sentiments.append([[token], sentiment[2:]])

                # if it is inside, we have different options
                if sentiment.startswith("I-"):
                    # if this is the first sentiment, then we just treat it as a beginning one
                    if len(tokens2sentiments) == 0:
                        tokens2sentiments.append([[token], sentiment[2:]])
                    else:
                        # otherwise, there is some other sentiment before
                        last_token, last_sentiment = tokens2sentiments[-1]
                        # if the last sentiment is not equal to the one we're considering, then we treat this
                        # again as a beginning one.
                        if last_sentiment != sentiment[2:]:
                            tokens2sentiments.append([[token], sentiment[2:]])
                        # otherwise, the sentiment before was a B or a I with the same sentiment
                        # therefore this token is part of the same target instance, with the same sentiment associated
                        else:
                            tokens2sentiments[-1] = [
                                last_token + [token],
                                sentiment[2:],
                            ]
            elif self.tagging_schema == "BIOES":
                # just ignore outside sentiments for the moment
                if sentiment == "O":
                    tokens2sentiments.append([[token], sentiment])

                # if it is single, then we just need to append that
                if sentiment.startswith("S-"):
                    tokens2sentiments.append([[token], sentiment[2:]])

                # if it is starting, then we expect something after, so we append the first one
                if sentiment.startswith("B-"):
                    tokens2sentiments.append([[token], sentiment[2:]])

                # if it is inside, we have different options
                if sentiment.startswith("I-") or sentiment.startswith("E-"):
                    # if this is the first sentiment, then we just treat it as a beginning one
                    if len(tokens2sentiments) == 0:
                        tokens2sentiments.append([[token], sentiment[2:]])
                    else:
                        # otherwise, there is some other sentiment before
                        last_token, last_sentiment = tokens2sentiments[-1]
                        # if the last sentiment is not equal to the one we're considering, then we treat this
                        # again as a beginning one.
                        if last_sentiment != sentiment[2:]:
                            tokens2sentiments.append([[token], sentiment[2:]])
                        # if the previous sentiment was a single target word or an ending one
                        # we treat the one we're considering again as a beginning one
                        elif sentiments[i - 1].startswith("S-") or sentiments[
                            i - 1
                        ].startswith("E-"):
                            tokens2sentiments.append([[token], sentiment[2:]])
                        # otherwise, the sentiment before was a B or a I with the same sentiment
                        # therefore this token is part of the same target instance, with the same sentiment associated
                        else:
                            tokens2sentiments[-1] = [
                                last_token + [token],
                                sentiment[2:],
                            ]

        return tokens2sentiments

    def postprocess(self, tokens: List[str], sentiments: List[str]):
        tokens2sentiments = self.tags_to_json(tokens, sentiments)

        if isinstance(self.tokenizer, TreebankWordTokenizer):
            detokenizer = TreebankWordDetokenizer()
            return {
                "targets": [
                    (detokenizer.detokenize(tk), sentiment)
                    for tk, sentiment in tokens2sentiments
                    if sentiment != "O"
                ]
            }
        else:
            return {
                "targets": [
                    (self.tokenizer.convert_tokens_to_string(tk), sentiment)
                    for tk, sentiment in tokens2sentiments
                    if sentiment != "O"
                ]
            }

    def batch_sentiments_to_tags(self, sentences, batch_sentiments, lengths):
        to_process_list: List[List[int]] = batch_sentiments.tolist()

        # remove padded elements
        for i, length in enumerate(lengths):
            to_process_list[i] = to_process_list[i][:length]

        # extract tokens and associated sentiments
        tokens = [self.tokenizer.tokenize(x) for x in sentences]

        # convert indexes to tokens + IOB format sentiments
        processed_iob_sentiments = [
            [self.sentiments_vocabulary.itos[x] for x in batch]
            for batch in to_process_list
        ]

        # convert IOB sentiments to simple | target words - sentiment | format
        words_to_sentiment = [
            self.postprocess(token, output)
            for token, output in zip(tokens, processed_iob_sentiments)
        ]

        return words_to_sentiment
