# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Model Soup')
    parser.add_argument('--models', nargs='+', default=[], help='Ensemble results')
    parser.add_argument('--model-folder', default=None, help='Ensemble results')
    parser.add_argument('--out', default="pred_results.csv", help='output path')
    args = parser.parse_args()
    assert len(args.models) != 0 or args.model_folder is not None
    return args

def get_models(args):

    return []

def main():
    args = parse_args()
    models = get_models(args)

if __name__ == '__main__':
    main()
