import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import _init_paths
from config.default import config as cfg
from config.default import update_config
import pprint
from models.network import MissNet
from dataset.dataset import WtalDataset
from core.train_eval import train, evaluate
from core.functions import prepare_env

from utils.utils import load_weights


def args_parser():
    parser = argparse.ArgumentParser(description='weakly supervised action localization baseline')
    parser.add_argument('-cfg', help='Experiment config file', default='../experiments/MissNet.yaml')
    args = parser.parse_args()
    return args


def main():
    args = args_parser()
    update_config(args.cfg)
    if cfg.BASIC.SHOW_CFG:
        pprint.pprint(cfg)
    # prepare running environment for the whole project
    prepare_env(cfg)

    # dataloader
    val_dset = WtalDataset(cfg, cfg.DATASET.VAL_SPLIT)
    val_loader = DataLoader(val_dset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False,
                            num_workers=cfg.BASIC.WORKERS, pin_memory=cfg.BASIC.PIN_MEMORY)

    # network
    model = MissNet(cfg)
    model.cuda()

    weight_file = '/disk3/yangle/diagnose/code/output/implementation/checkpoint_best_.pth'
    model = load_weights(model, weight_file)

    test_acc = evaluate(cfg, val_loader, model)
    print(test_acc)


if __name__ == '__main__':
    main()
