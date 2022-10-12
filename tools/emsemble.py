# Copyright (c) OpenMMLab. All rights reserved.
import pickle
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Ensemble Learning')
    parser.add_argument(
        '--pkls',
        nargs='+',
        default=[],
        help='output path')
    parser.add_argument('--out', help='the file to save results.')
    args = parser.parse_args()
    return args

def loda_pkl(pkl_path):
    with open(pkl_path, "wb") as pkl_file:
        data = pickle.load(pkl_file)
    return data


def main():
    args = parse_args()
    data_list = []
    for pkl in args.pkls:
        data_list.append(loda_pkl(pkl))

if __name__ == '__main__':
    main()
