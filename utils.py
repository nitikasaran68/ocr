import numpy as np
from param import alphabet, rev_alphabet

def encode(label):
    len_label = len(label)
    ret = np.zeros(len_label)
    for idx, char in zip(range(len_label), label):
        ret[idx] = alphabet[char]
    return ret.astype(int)

# Reverse translation of numerical classes back to characters
def decode(encoded_label):
    len_encoded_label = len(encoded_label)
    ret = []
    for encoded_char in encoded_label:
        if encoded_char == len_encoded_label:  # CTC Blank
            ret.append("")
        else:
            ret.append(rev_alphabet[encoded_char])
    return "".join(ret)