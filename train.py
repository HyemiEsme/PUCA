import argparse, os
from importlib import import_module

import torch

from src.util.config_parse import ConfigParser
from src.trainer import get_trainer_class


def main():
    # parsing configuration
    args = argparse.ArgumentParser()
    args.add_argument('-s', '--session_name', default='debug',  type=str)
    args.add_argument('-c', '--config',       default='PUCA_SIDD',  type=str)
    args.add_argument('-r', '--resume',       action='store_true')
    args.add_argument('-g', '--gpu',          default='3',  type=str)
    args.add_argument(      '--thread',       default=8,     type=int)

    args = args.parse_args()

    assert args.config is not None, 'config file path is needed'
    if args.session_name is None:
        args.session_name = args.config # set session name to config file name

    cfg = ConfigParser(args)

    # device setting
    if cfg['gpu'] is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg['gpu']

    # intialize trainer
    trainer = get_trainer_class(cfg['trainer'])(cfg)

    # train
    trainer.train()


if __name__ == '__main__':
    main()
