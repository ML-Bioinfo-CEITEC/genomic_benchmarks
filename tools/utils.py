import torch 
import numpy as np

# TODO add one-hot encoding
def coll(batch):
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
        
#         Todo: padding
        if(False):
#             for i in range(4707 - len(tmp)):
#                 tmp.append(4)
            pad = nn.ConstantPad2d(0, (0,4707 - len(tmp),0,0))
            tmp = pad(tmp)
        x = torch.tensor(tmp, dtype=torch.float32)
        xs.append(x) #TODO int doesnt work, why?
    
    xs = torch.stack(xs)
    ys = torch.stack(ys)
    return xs.to('cuda'),ys.to('cuda')


#             torch.nn.functional.pad(
#                 torch.tensor([mapper[ch] for ch in text], dtype=torch.float32), 
#                 (0,4707), 
#                 mode='constant', 
#                 value=4
#             )


