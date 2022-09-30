# Copyright (c) OpenMMLab. All rights reserved.
import math
from pathlib import Path
import mmengine.dist as dist
from mmengine.device import get_device
from mmcls.datasets import CustomDataset
from mmengine.model import MMDistributedDataParallel
from unittest.mock import patch
from mmengine.logging import print_log
import torch

from mmcls.apis import  init_model
from mmcls.utils import register_all_modules

import argparse
import os
from mmengine.config import Config, DictAction
from mmengine.runner import Runner

from mmcls.utils import register_all_modules


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMCLS test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        'folder',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument('--out', help='the file to save results.')
    parser.add_argument(
        '--out-keys', 
        nargs='+',
        default=['filename', 'pred_class'], 
        help='output path')
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
    # do not init the default scope here because it will be init in the runner
    register_all_modules(init_default_scope=False)

    # load config
    cfg = Config.fromfile(args.config)
    cfg.env_cfg.dist_cfg = dict(type='gloo')
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if args.launcher != 'none' and not dist.is_distributed():
        dist_cfg: dict = cfg.env_cfg.get('dist_cfg', {})
        dist.init_dist(args.launcher, **dist_cfg)

    folder = Path(args.folder)
    if folder.is_file():
        image_path_list = [folder]
    elif folder.is_dir():
        image_path_list = [
            p for p in folder.iterdir() if p.is_file()]
    data_list = [
        {'img_path': str(img_path), 'gt_label': int(-1)}
        for img_path in image_path_list
    ]
    print(f"Total images number : {len(image_path_list)}")
    model = init_model(cfg, args.checkpoint, device=get_device())
    CLASSES = [f"{i:0>4d}" for i in range(model.head.num_classes)]

    if args.launcher != 'none' and dist.is_distributed:
        model = MMDistributedDataParallel(
            module=model,
            device_ids=[int(os.environ['LOCAL_RANK'])],)

    sim_dataloader = cfg.test_dataloader
    sim_dataloader.dataset=dict(
            type='CustomDataset',
            data_prefix=args.folder,
            pipeline=cfg.test_dataloader.dataset.pipeline,
            _scope_='mmcls')

    with patch.object(CustomDataset, 'load_data_list', return_value=data_list):
        sim_loader = Runner.build_dataloader(sim_dataloader)

        result_list = []
        world_size = dist.get_world_size()
        iters = math.ceil(len(sim_loader) // world_size)
        count = 0
        with torch.no_grad():
            for data_batch in sim_loader:
                batch_prediction = model.val_step(data_batch)
                count +=1 
                if count % 50 == 0:
                    print_log(f"[{count} / {iters}].....")
                # forward the model
                for cls_data_sample in batch_prediction:
                    cls_pred_label = cls_data_sample.pred_label
                    pred_score = torch.max(cls_pred_label.score).item()
                    pred_label = cls_pred_label.label.item()
                    result = dict(
                        filename = Path(cls_data_sample.img_path).name,
                        pred_score = pred_score,
                        pred_label = pred_label,
                        pred_class = CLASSES[pred_label])
                result_list.append(result)
    
    all_results = dist.all_gather(result_list)
    if dist.is_main_process():
        for result in all_results:
            print(result)
    output_result(args, all_results)


def output_result(args, result_list):
    if args.out and args.out.endswith(".json"):
        import json
        json.dump(result_list, open(args.out, 'w'))
    elif args.out and args.out.endswith(".csv"):
        import csv
        with open(args.out, "w") as csvfile: 
            writer = csv.writer(csvfile)
            # writer.writerow(args.out_keys)
            for result in result_list:
                writer.writerow([result[k] for k in args.out_keys])
    return result_list


if __name__ == '__main__':
    main()
