import torch 
import numpy as np

# TODO add one-hot encoding
def coll(batch):
    mapper = {
        'A':0,
        'C':1,
        'T':2,
        'G':3,
    }
    xs, ys = [],[]
    for text,label in batch:
        ys.append(torch.tensor([label], dtype=torch.float32))
        # xs.append(torch.tensor([mapper[ch] for ch in text]).to('cuda'))
        xs.append(torch.tensor([mapper[ch] for ch in text], dtype=torch.float32)) #TODO int doesnt work, why?
        
    
    xs = torch.stack(xs)
    ys = torch.stack(ys)
    return xs.to('cuda'),ys.to('cuda')