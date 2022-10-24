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
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        'folder',
        help='the directory to save the file containing evaluation metrics')
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
    CLASSES = [f"{i:0>4d}" for i in range(model.head.num_classes)]

    sim_dataloader = cfg.test_dataloader
    print(args.folder)
    sim_dataloader.dataset=dict(
            type='CustomDataset',
            data_prefix=args.folder,
            pipeline=cfg.test_dataloader.dataset.pipeline)

    if args.ttaug:
        from src.models.classifier_tta import ClassifierTTA
        from mmcls.models.classifiers import ImageClassifier

        assert isinstance(model, ImageClassifier)
        model = ClassifierTTA(model)
        sim_dataloader.dataset['pipeline'] = cfg.tta_pipeline

    if args.launcher != 'none' and dist.is_distributed:
        model = MMDistributedDataParallel(
            module=model,
            device_ids=[int(os.environ['LOCAL_RANK'])],)

    with patch.object(CustomDataset, 'load_data_list', return_value=data_list):
        sim_loader = Runner.build_dataloader(sim_dataloader) 

    result_list = []
    with torch.no_grad():
        for data_batch in track_on_main_process(sim_loader):
            batch_prediction = model.test_step(data_batch)

    


def get_pred(feats, index_feats):
    similarity_fn = lambda a, b: torch.cosine_similarity(a.unsqueeze(1), b.unsqueeze(0), dim=-1)
    sim = similarity_fn(feats, index_feats)
    sorted_sim, indices = torch.sort(sim, descending=True, dim=-1)
    predictions = dict(
            score=sim, pred_label=indices, pred_score=sorted_sim)
    return predictions

def loda_pkl(pkl_path):
    with open(pkl_path, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    images = data['images']
    feats = data['feats']
    labels = data['labels']
    return images, feats, labels

if __name__ == '__main__':
    main()
