import argparse
from plot_utils import plot_logs
from pathlib import Path, PurePath

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=True)
    parser.add_argument('--path', type=str)
    parser.add_argument('--name', type=str)
    parser.add_argument('--save', type=int, default=1)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args_parser()
    plot_logs([Path(args.path)], name=args.name, save=args.save)

    #plot_logs([PurePath('detr/sctrach_6/'),
    #           PurePath('detr/sctrach_1/'),
    #           PurePath('detr/finetuning_6/')])

