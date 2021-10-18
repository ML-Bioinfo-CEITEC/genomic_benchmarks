import torch 
import numpy as np
from collections import Counter 
from torch.nn import ConstantPad1d
from torchtext.vocab import vocab, build_vocab_from_iterator


def simple_coll_factory(vocab, tokenizer):
    def simple_coll(batch):
        xs, ys = [],[]      
        for text,label in batch:
            ys.append(torch.tensor([label], dtype=torch.float32))
            tmp = torch.tensor([vocab[token] for token in tokenizer(text)], dtype=torch.long)
            x = torch.tensor(tmp, dtype=torch.long)
            xs.append(x)
            
        xs = torch.stack(xs)
        ys = torch.stack(ys)
        return xs.to('cuda'),ys.to('cuda')
    return simple_coll


def padding_coll_factory(longest_length, vocab, tokenizer):
    def padding_coll(batch):
        PAD_IDX = vocab['<pad>']
        xs, ys = [],[]
        for text,label in batch:
            ys.append(torch.tensor([label], dtype=torch.float32))
            tmp = torch.tensor([vocab[token] for token in tokenizer(text)], dtype=torch.long)
            pad = ConstantPad1d((0, longest_length - len(tmp)), PAD_IDX)
            tmp = pad(tmp)
            
            x = torch.tensor(tmp, dtype=torch.long)
            xs.append(x)

        xs = torch.stack(xs)
        ys = torch.stack(ys)
        return xs.to('cuda'),ys.to('cuda')
    
    return padding_coll

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
