import torch 
import numpy as np
from collections import Counter 
from torch.nn import ConstantPad1d
from torchtext.vocab import vocab, build_vocab_from_iterator


def coll_factory(vocab, tokenizer, device='cpu', pad_to_length=None):
    def coll(batch):
        xs, ys = [],[]      
            
        for text,label in batch:
            ys.append(torch.tensor([label], dtype=torch.float32))
            x = torch.tensor([vocab[token] for token in tokenizer(text)], dtype=torch.long)
            if(pad_to_length != None):
                PAD_IDX = vocab['<pad>']
                pad = ConstantPad1d((0, pad_to_length - len(x)), PAD_IDX)
                x = torch.tensor(pad(x), dtype=torch.long)
            xs.append(x)

        xs = torch.stack(xs)
        ys = torch.stack(ys)
        return xs.to(device),ys.to(device)
    return coll

#             torch.nn.functional.pad(
#                 torch.tensor([mapper[ch] for ch in text], dtype=torch.float32), 
#                 (0,4707), 
#                 mode='constant', 
#                 value=4
#             )


class LetterTokenizer():
    def __init__(self, **kwargs):
        pass
    def __call__(self, items):
        if isinstance(items, str):
            return self.__tokenize_str(items)
        else:
            return (self.__tokenize_str(t) for t in items)
    def __tokenize_str(self, t):
        tokenized = list(t.replace("\n",""))
        tokenized.append('<eos>')
        tokenized.insert(0,'<bos>')
        return tokenized
    

def build_vocab(dataset, tokenizer, use_padding):
    counter = Counter()
    for i in range(len(dataset)):
        counter.update(tokenizer(dataset[i][0]))
#     print(counter.most_common())
    builded_voc = vocab(counter)
    if(use_padding):
        builded_voc.append_token('<pad>')
    return builded_voc

# todo: why build fn does not work as expected (iterator argument)
#     return build_vocab_from_iterator(
#         iterator = counter, 
#         specials = ['<unk>', '<pad>', '<bos>', '<eos>'],
#         special_first = True)
