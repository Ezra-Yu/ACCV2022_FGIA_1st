import argparse
import os
import pickle

import torch
from pathlib import Path
import mmengine.dist as dist
from mmengine.device import get_device
from mmengine.model import MMDistributedDataParallel
from unittest.mock import patch

from mmengine.config import Config, DictAction
from mmcls.utils import register_all_modules
from mmcls.apis import init_model
from mmengine.runner import Runner
from mmcls.datasets import CustomDataset
from mmcls.utils import track_on_main_process


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMCLS test (and eval) a model')
    parser.add_argument('query', help='checkpoint file')
    parser.add_argument('index', help='checkpoint file')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def main():
    args = parse_args()

    # register all modules in mmcls into the registries
    register_all_modules()

    query_images, query_feats, query_labels = loda_pkl(args.query)
    index_images, index_feats, index_labels = loda_pkl(args.index)
    query_feats = query_feats.to("cuda")
    index_feats = index_feats.to("cuda")

    result = []
    for image, feat, label in zip(query_images, query_feats, query_labels):
        prediction = get_pred(feat, index_feats)
        score = prediction['score'].cpu().numpy.tolist()
        label = index_labels[prediction['label']].cpu().numpy.tolist()
        result.append( (image, label, score) )

        print(result[-1])


def get_pred(feat, index_feats, topk=10):
    similarity_fn = lambda a, b: torch.cosine_similarity(a.unsqueeze(1), b.unsqueeze(0), dim=-1)
    sim = similarity_fn(feat, index_feats)
    sorted_sim, indices = torch.sort(sim, descending=True, dim=-1)
    prediction = dict(score=sorted_sim[:topk], pred_label=indices[:topk])
    return prediction

def loda_pkl(pkl_path):
    with open(pkl_path, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    images = data['images']
    feats = data['feats']
    labels = data['labels']
    return images, feats, labels

if __name__ == '__main__':
    main()
