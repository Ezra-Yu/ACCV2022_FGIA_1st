# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmengine.fileio import dump
from rich import print_json
from pathlib import Path
from mmengine.utils import ProgressBar

from mmcls.apis import inference_model, init_model
from mmcls.utils import register_all_modules


def main():
    parser = ArgumentParser() 
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('path', help='Image file path or folder')
    parser.add_argument('--out', help='output path')
    parser.add_argument(
        '--out-keys', 
        nargs='+',
        default=['filename', 'pred_label'], 
        help='output path')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    # register all modules and set mmcls as the default scope.
    register_all_modules()
    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    # test a single image
    path = Path(args.path)
    if path.is_file():
        result = inference_model(model, str(path))
        print_json(dump(result, file_format='json', indent=4))
        return
    elif path.is_dir():
        image_path_list = [
            p for p in path.iterdir() if p.is_file()
        ]
        result_list = []
        pbar = ProgressBar(len(image_path_list))
        for i, image_path in enumerate(image_path_list):
            result = inference_model(model, str(image_path))
            result['filename'] = image_path.name
            result_list.append(result)
            pbar.update()

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


if __name__ == '__main__':
    main()
