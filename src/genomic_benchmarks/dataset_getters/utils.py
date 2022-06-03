import types
from collections import Counter

import numpy as np
import torch
from torch.nn import ConstantPad1d
from torchtext.vocab import build_vocab_from_iterator, vocab

VARIABLE_LENGTH_DATASETS = [
    "human_enhancers_ensembl",
    "dummy_mouse_enhancers_ensembl",
    "human_ocr_ensembl",
    "human_ensembl_regulatory"
]


def coll_factory(vocab, tokenizer, device="cpu", pad_to_length=None):
    def coll(batch):
        xs, ys = [], []

        for text, label in batch:
            ys.append(torch.tensor([label], dtype=torch.float32))
            x = torch.tensor([vocab[token] for token in tokenizer(text)], dtype=torch.long)
            if pad_to_length != None:
                PAD_IDX = vocab["<pad>"]
                pad = ConstantPad1d((0, pad_to_length - len(x)), PAD_IDX)
                x = torch.tensor(pad(x), dtype=torch.long)
            xs.append(x)

        xs = torch.stack(xs)
        ys = torch.stack(ys)
        return xs.to(device), ys.to(device)

    return coll


#             torch.nn.functional.pad(
#                 torch.tensor([mapper[ch] for ch in text], dtype=torch.float32),
#                 (0,4707),
#                 mode='constant',
#                 value=4
#             )


class LetterTokenizer:
    def __init__(self, **kwargs):
        pass

    def __call__(self, items):
        if isinstance(items, str):
            return self.__tokenize_str(items)
        else:
            return (self.__tokenize_str(t) for t in items)

    def __tokenize_str(self, t):
        tokenized = list(t.replace("\n", ""))
        tokenized.append("<eos>")
        tokenized.insert(0, "<bos>")
        return tokenized


def build_vocab(dataset, tokenizer, use_padding):
    counter = Counter()
    for i in range(len(dataset)):
        counter.update(tokenizer(dataset[i][0]))
    #     print(counter.most_common())
    builded_voc = vocab(counter)
    if use_padding:
        builded_voc.append_token("<pad>")
    builded_voc.insert_token("<unk>", 0)
    builded_voc.set_default_index(0)
    return builded_voc


# todo: why build fn does not work as expected (iterator argument)
#     return build_vocab_from_iterator(
#         iterator = counter,
#         specials = ['<unk>', '<pad>', '<bos>', '<eos>'],
#         special_first = True)


def check_seq_lengths(dataset, use_padding):
    # Compute length of the longest sequence
    max_seq_len = max([len(dataset[i][0]) for i in range(len(dataset))])
    print("max_seq_len ", max_seq_len)
    same_length = [len(dataset[i][0]) == max_seq_len for i in range(len(dataset))]
    if not all(same_length):
        print("not all sequences are of the same length")

    # Count in tokens added in tokenizer '<bos>' and '<eos>' and the padding token <pad>
    if use_padding:
        len_with_tokens = max_seq_len + 3
    else:
        len_with_tokens = max_seq_len + 2
    return max_seq_len, len_with_tokens


def check_config(config):
    control_config = {
        "use_padding": bool,
        "run_on_gpu": bool,
        "dataset": str,
        "number_of_classes": int,
        "dataset_version": int,
        "force_download": bool,
        "epochs": int,
        "embedding_dim": int,
        "batch_size": int,
    }

    for key in config.keys():
        assert isinstance(config[key], control_config[key]), '"{}" in config should be of type {} but is {}'.format(
            key, control_config[key], type(config[key])
        )
