import torch 
import numpy as np
from torch.nn import ConstantPad1d

# TODO add one-hot encoding

def simple_coll(batch):
    mapper = {
        'A':0,
        'C':1,
        'T':2,
        'G':3,
        'N':4,
    }
    xs, ys = [],[]
    for text,label in batch:
        ys.append(torch.tensor([label], dtype=torch.float32))
        # xs.append(torch.tensor([mapper[ch] for ch in text]).to('cuda'))
        tmp = [mapper[ch] for ch in text]
        x = torch.tensor(tmp, dtype=torch.float32)
        xs.append(x) #TODO int doesnt work, why?
    
    xs = torch.stack(xs)
    ys = torch.stack(ys)
    return xs.to('cuda'),ys.to('cuda')


def padding_coll_factory(longest_length, vocab, tokenizer):
    def padding_coll(batch):
        PAD_IDX = vocab['<pad>']
        
#         mapper = {
#             'A':0,
#             'C':1,
#             'T':2,
#             'G':3,
#             'N':4,
#         }
        xs, ys = [],[]
        for text,label in batch:
            ys.append(torch.tensor([label], dtype=torch.float32))
            tmp = torch.tensor([vocab[token] for token in tokenizer(text)], dtype=torch.long)
            
#             tmp = [mapper[ch] for ch in text]
    #         maybe not necessary
#             tmp = torch.tensor(tmp, dtype=torch.float32)
    #       Todo: padding
            pad = ConstantPad1d((0, longest_length - len(tmp)), PAD_IDX)
            tmp = pad(tmp)
            
            x = torch.tensor(tmp, dtype=torch.long)
            xs.append(x)

        xs = torch.stack(xs)
        ys = torch.stack(ys)
#         print(xs.size())
#         print(ys.size())
        return xs.to('cuda'),ys.to('cuda')
    
    return padding_coll

#             torch.nn.functional.pad(
#                 torch.tensor([mapper[ch] for ch in text], dtype=torch.float32), 
#                 (0,4707), 
#                 mode='constant', 
#                 value=4
#             )


