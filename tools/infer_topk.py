import argparse
import os
import pickle
import csv
import random

import torch
from mmengine.utils import ProgressBar
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

CLASSES = [f"{i:0>4d}" for i in range(5000)]

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMCLS test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        'folder',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument('index', help='checkpoint file')
    parser.add_argument('--out', default="pred_results.csv", help='the file to save results.')
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

def get_select_index_list(index_labels, n=10):
    labels_dict = {c:0 for  c in CLASSES}
    labels2index_dict = {c:[] for  c in CLASSES}
    for i, label in enumerate(index_labels):
        labels_dict[label] += 1
        labels2index_dict[label].append(i)
    
    sel_index = []
    for  c in CLASSES:
        p_indexs = labels2index_dict[c]
        if len(p_indexs) > n:
            random.sample(p_indexs, n)
        elif len(p_indexs) == n:
            sel_index += p_indexs
        elif len(p_indexs) < n:
            while len(p_indexs) < n:
                p_indexs += p_indexs
            random.sample(p_indexs, n)
    
    return sel_index

def main():
    args = parse_args()

    # register all modules in mmcls into the registries
    register_all_modules()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.env_cfg.dist_cfg = dict(backend='gloo')
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if args.launcher != 'none' and not dist.is_distributed():
        dist_cfg: dict = cfg.env_cfg.get('dist_cfg', {})
        dist.init_dist(args.launcher, **dist_cfg)

    index_images, index_feats, index_labels = loda_pkl(args.index)
    select_index = get_select_index_list(index_labels)
    print(f"You have select {len(select_index)} samplers....")
    index_feats = index_feats[select_index]
    index_labels = index_labels[select_index]
    index_feats = index_feats.cuda()

    folder = Path(args.folder)
    if folder.is_file():
        image_path_list = [folder]
    elif folder.is_dir():
        # image_path_list = [p for p in folder.iterdir() if p.is_file()]
        image_path_list = os.listdir(folder)
    data_list = [
        {'img_path': os.path.join(folder, img_path), 'gt_label': int(-1)}
        for img_path in image_path_list
    ]
    print(f"Total images number : {len(image_path_list)}")
    model = init_model(cfg, args.checkpoint, device=get_device())

    sim_dataloader = cfg.test_dataloader
    print(args.folder)
    sim_dataloader.dataset=dict(
            type='CustomDataset',
            data_prefix=args.folder,
            pipeline=cfg.test_dataloader.dataset.pipeline)
    sim_dataloader.batch_size = 16

    if args.launcher != 'none' and dist.is_distributed:
        model = MMDistributedDataParallel(
            module=model,
            device_ids=[int(os.environ['LOCAL_RANK'])],)
    model = model.module
    with patch.object(CustomDataset, 'load_data_list', return_value=data_list):
        sim_loader = Runner.build_dataloader(sim_dataloader)

    result_list = []
    if dist.is_main_process():
        progressbar = ProgressBar(len(sim_loader) * sim_loader.batch_size)

    topk = 10
    with torch.no_grad():
        for data_batch in sim_loader:

            data = model.data_preprocessor(data_batch, False)
            feats = model.extract_feat(data["inputs"])
            data_samples = model.head.predict(feats, data['data_samples'])
            if isinstance(feats, tuple):
                feats = feats[-1]

            batch_sim_prediction = get_pred(feats, index_feats, topk)
            for data_sample, sim_pred_label, sim_pred_score  in zip(data_samples, batch_sim_prediction['pred_label'], batch_sim_prediction['score']):
                sample_idx = data_sample.get('sample_idx')
                filename = Path(image_path_list[sample_idx]).name
                score_pred = data_sample.pred_label.score
                label, s = get_single_res(score_pred, sim_pred_label, sim_pred_score ,index_labels, topk)
                result_list.append( (filename, label, s) )

            if dist.is_main_process():
                progressbar.update(sim_loader.batch_size)
    result = post_process(result_list)

    assert args.out and args.out.endswith(".csv")
    with open(args.out, "w") as csvfile:
        writer = csv.writer(csvfile)
        for r in result:
            writer.writerow(r)

def get_single_res(score_pred, pred_labels, scores, index_labels, topk=5):
    pred = torch.zeros( (5000) )
    for i in range(topk):
        pli = index_labels[pred_labels[i]]
        score = scores[i]
        pred[pli] += float(score) ** 1 *  float(score_pred[pli]) ** 3

    p_label = torch.argmax(pred).item()
    s = pred[p_label].item()

    return p_label, s

def post_process(result):
    result_list = []
    for filename, label, scores in result:
        pred_label = label
        pred_class = CLASSES[pred_label]
        result_list.append( (filename, pred_class) )
    return result_list

def get_pred(feat, index_feats, topk=100):
    similarity_fn = lambda a, b: torch.cosine_similarity(a.unsqueeze(1), b.unsqueeze(0), dim=-1)
    sim = similarity_fn(feat, index_feats)
    sorted_sim, indices = torch.sort(sim, descending=True, dim=-1)
    prediction = dict(score=sorted_sim[:, :topk].cpu().numpy(), pred_label=indices[:, :topk].cpu().numpy())
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
