import argparse
import imp
import os
import pickle
import csv
from pathlib import Path

import torch
from mmengine.utils import ProgressBar
from mmengine.config import  DictAction
from mmcls.utils import register_all_modules


CLASSES = [f"{i:0>4d}" for i in range(5000)]

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMCLS test (and eval) a model')
    parser.add_argument('query', help='checkpoint file')
    parser.add_argument('index', help='checkpoint file')
    parser.add_argument('--out', default="pred_results.csv", help='output path')
    parser.add_argument('--batch', type=int, default=16, help='output path')
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
    print(f"Query size: {query_feats.size()}; Index Size: {index_feats.size()}")

    result = []
    progressbar = ProgressBar(len(query_images))
    for i, (image, feat, label) in enumerate(zip(query_images, query_feats, query_labels)):
        feat = feat.unsqueeze(0)
        prediction = get_pred(feat, index_feats)
        score = prediction['score']
        label = index_labels[prediction['pred_label']]
        result.append( (Path(image), label, score) )
        progressbar.update()

    result_list = post_process(result)

    assert args.out and args.out.endswith(".csv")
    with open(args.out, "w") as csvfile:
        writer = csv.writer(csvfile)
        for result in result_list:
            writer.writerow(result)

def post_process(result):
    result_list = []
    for filename, label, scores in result:
        pred_label = label[0][0].item()
        pred_class = CLASSES[pred_label]
        result_list.append( (filename, pred_class) )
    return result_list

def get_pred(feat, index_feats, topk=100):
    similarity_fn = lambda a, b: torch.cosine_similarity(a.unsqueeze(1), b.unsqueeze(0), dim=-1)
    sim = similarity_fn(feat, index_feats)
    sorted_sim, indices = torch.sort(sim, descending=True, dim=-1)
    prediction = dict(score=sorted_sim[:topk].cpu().numpy(), pred_label=indices[:topk].cpu().numpy())
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
