from Bio import SeqIO
import requests
from io import StringIO
from tqdm.auto import tqdm


def fasta2loc(fasta_url, ref_dict):

    data = requests.get(fasta_url).text

    tree = {}
    nseqs = 0

    # building tree for seq searching
    for seq in SeqIO.parse(StringIO(data), "fasta"):
        s = str(seq.seq)
        rev = str(seq.seq.reverse_complement())
        sname = seq.name
        Nseqs += 1
    
        _update_tree(tree, s, sname, '+')
        _update_tree(tree, rev, sname, '-')

    print(f"{nseqs} sequences read and parsed.")

    results = {}
    

    for chrom in tqdm(ref_dict):
        curr_positions = []

        for i, c in tqdm(enumerate(ref_dict[chrom]), total=len(ref_dict[chrom]), leave=False):
    
            prev_positions = curr_positions + [tree]
            curr_positions = []
    
            for pos in prev_positions:
                if c in pos:
                    pos = pos[c]
                    curr_positions.append(pos)
                    
                    if 'terminal' in pos:
                        results[pos['terminal'][0]] = (chrom, i-pos['terminal'][2]+1, i+1, pos['terminal'][1])

    return results


def _update_tree(root, seq_str, seq_name, direction):
    # updates tree in `root` with a sequence `seq_str`
    position = root

    for c in seq_str:
        if c in position:
            position = position[c]
        else:
            position[c] = {}
            position = position[c]
    position['terminal'] = (seq_name, direction, len(seq_str))