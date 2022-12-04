# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
from collections import OrderedDict

import mmengine.dist as dist
from mmengine.device import get_device
from mmengine.model import MMDistributedDataParallel
import torch
from mmengine.utils import ProgressBar

from mmcls.apis import  init_model
from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mmcls.utils import register_all_modules
import src

import mmengine.dist as dist
import rich.progress as progress


def track_on_main_process(sequence, *args, **kwargs):
    if not dist.is_main_process():
        return sequence

    yield from progress.track(sequence, *args, **kwargs)

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMCLS test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--train', action='store_true', help='enable tta')
    parser.add_argument('--dump', default="feats.pkl", help='dump to results.')
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

    # load config
    cfg = Config.fromfile(args.config)
    cfg.env_cfg.dist_cfg = dict(backend='gloo')
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # init didt env
    if args.launcher != 'none' and not dist.is_distributed():
        dist_cfg: dict = cfg.env_cfg.get('dist_cfg', {})
        dist.init_dist(args.launcher, **dist_cfg)

    # init model
    model = init_model(cfg, args.checkpoint, device=get_device())

    # DDP model in dist
    if args.launcher != 'none' and dist.is_distributed:
        model = MMDistributedDataParallel(
            module=model,
            device_ids=[int(os.environ['LOCAL_RANK'])],)

    # init dataloader
    if args.train:
        dataloader = Runner.build_dataloader(cfg.train_dataloader)
        print(f"Exract Train Dataloader {len(dataloader)}......")
    else:
        dataloader = Runner.build_dataloader(cfg.test_dataloader)
        print(f"Exract Test Dataloader {len(dataloader)}......")
    dataset = dataloader.dataset
    images = [dataset.get_data_info(i)['img_path'] for i in range(len(dataset))]
    # images = [data['img_path'] for data in dataloader.dataset.data_list]
    print(f"There are {len(images)} in the dataloader")

    # get feats and labels
    feats = get_feat_from_dataloader(model, dataloader)
    labels = dataloader.dataset.get_gt_labels()

    # load result
    result = OrderedDict(feats=feats, labels=labels, images=images)
    # result = OrderedDict(feats=feats, labels=labels)
    print(f"{len(feats)} images' features have been extracted and load into {args.dump}")
    assert len(feats) == len(labels) == len(images), f"{len(feats)}  {len(labels)}  {len(images)}"
    if args.dump:
        assert args.dump.endswith(".pkl")
        with open(args.dump, "wb") as dumpfile:
            import pickle
            pickle.dump(result, dumpfile)


@torch.no_grad()
def get_feat_from_dataloader(model, data_loader):
    """get prototype_vecs from dataloader."""
    num = len(data_loader.dataset)
    if dist.is_main_process():
        progressbar = ProgressBar(len(data_loader) * data_loader.batch_size)

    prototype_vecs = None
    for data_batch in data_loader:
        if isinstance(model, MMDistributedDataParallel):
            model = model.module

        data = model.data_preprocessor(data_batch, False)
        feat = model.extract_feat(data["inputs"])
        if isinstance(feat, tuple):
            feat = feat[-1]

        if prototype_vecs is None:
            dim = feat.shape[-1]
            prototype_vecs = torch.zeros(num, dim)
        for i, data_sample in enumerate(data_batch['data_samples']):
            sample_idx = data_sample.get('sample_idx')
            prototype_vecs[sample_idx] = feat[i]
        if dist.is_main_process():
            progressbar.update(data_loader.batch_size)

    assert prototype_vecs is not None
    dist.all_reduce(prototype_vecs)
    return prototype_vecs


if __name__ == '__main__':
    main()
