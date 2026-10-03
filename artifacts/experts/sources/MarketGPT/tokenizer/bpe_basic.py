"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

But:
- Does not handle the regular expression splitting pattern.
- Does not handle any special tokens.

Based on minBPE by Karpathy: https://github.com/karpathy/minbpe/blob/master/minbpe/base.py
"""

from .bpe_base import Tokenizer, get_stats, merge, get_stats_single


class BasicTokenizer(Tokenizer):

    def __init__(self):
        super().__init__()

    def train(self, msgs, vocab_size, vocab, verbose=False):
        assert vocab_size >= len(vocab)
        num_merges = vocab_size - len(vocab)

        # input preprocessing
        ids = [msg for msg in msgs] # list of integers in range 0..len(vocab)
        bpe_vocab = {} # new byte-pair encoding vocab

        # iteratively merge the most common pairs to create new tokens
        merges = {} # (int, int) -> int
        for i in range(num_merges):
            # count up the number of times every consecutive pair appears
            stats = get_stats(ids)
            # find the pair with the highest count
            pair = max(stats, key=stats.get)
            # mint a new token: assign it the next available id
            idx = len(vocab) + i
            # replace all occurrences of pair in ids with idx
            new_msgs = []
            for msg in ids:
                new_msg = merge(msg, pair, idx)
                new_msgs.append(new_msg)
            ids = new_msgs
            # save the merge
            merges[pair] = idx
            bpe_vocab[idx] = pair
            # prints
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({bpe_vocab[idx]}) had {stats[pair]} occurrences")

        # save class variables
        self.merges = merges # used in encode()
        self.bpe_vocab = bpe_vocab   # used in decode()

    def bpe_decode(self, tokens, vocab):
        # decode the tokens into a message
        msg = []
        for token in tokens:
            if token >= len(vocab):
                # get the corresponding byte-pair encoding
                bpe = self.bpe_vocab[token]
                # append both elements
                msg.extend(bpe)
            else:
                msg.append(token)

        # check that all tokens in message are valid
        valid = all(token < len(vocab) for token in msg)
        while not valid:
            new_msg = []
            for token in msg:
                if token >= len(vocab):
                    # get the corresponding byte-pair encoding
                    bpe = self.bpe_vocab[token]
                    # append both elements
                    new_msg.extend(bpe)
                else:
                    new_msg.append(token)
            msg = new_msg
            valid = all(token < len(vocab) for token in msg)
        return msg

    def bpe_encode(self, msg):
        # given a message, return the token ids
        ids = list(msg)
        while len(ids) >= 2:
            # find the pair with the lowest merge index
            stats = get_stats_single(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # subtle: if there are no more merges available, the key will
            # result in an inf for every single pair, and the min will be
            # just the first pair in the list, arbitrarily
            # we can detect this terminating case by a membership check
            if pair not in self.merges:
                break # nothing else can be merged anymore
            # otherwise let's merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids