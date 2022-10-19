import numpy as np
import torch

import mmengine

def compute_adjustions(ann_file, tro):
    """compute the base probabilities"""
    lines = mmengine.list_from_file(ann_file)
    samples = [x.strip().rsplit(' ', 1) for x in lines]

    label_freq = {}
    for filename, target in samples:
        label_freq[target] = label_freq.get(target, 0) + 1
    label_freq = dict(sorted(label_freq.items()))
    label_freq_array = np.array(list(label_freq.values()))
    label_freq_array = label_freq_array / label_freq_array.sum()
    adjustments = np.log(label_freq_array ** tro + 1e-12)
    adjustments = torch.from_numpy(adjustments)
    return adjustments

if __name__ == "__main__":
    adjustions = compute_adjustions("./data/ACCV_workshop/meta/all.txt", 1)
    print(len(adjustions))
    print(adjustions)
