import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import argparse
import pprint
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import _init_paths
from config.default import config as cfg
from config.default import update_config

from models.network_2layers import IENet
from dataset.dataset import ClinicalDataset
from dataset.devide_dataset import devide_multiple_sets
from core.train_eval import train, evaluate
from core.functions import prepare_env, fix_random_seed_all

from utils.utils import decay_lr, save_best_model, save_best_record_txt


def args_parser():
    parser = argparse.ArgumentParser(description='weakly supervised action localization baseline')
    parser.add_argument('-cfg', help='Experiment config file', default='../experiments/IENet.yaml')
    parser.add_argument('-seed_name', default='IENet_10fold_seed0')
    args = parser.parse_args()
    return args


def main():
    # update parameters
    args = args_parser()
    update_config(args.cfg)
    fix_random_seed_all(cfg)

    device = torch.device("cuda")
    assert torch.cuda.is_available(), "CUDA is not available"
   
    num_fold = cfg.BASIC.NUM_FOLD
    exp_names = [str(i) for i in range(num_fold)]
    exp_names = ['exp_' + s for s in exp_names]

    # checkpoint directory
    cfg.DATASET.CKPT_DIR = os.path.join('../ckpt', '{}_{}'.format(args.seed_name, cfg.BASIC.TIME))

    # prepare data for all devison
    devide_multiple_sets(num_fold, exp_names, cfg)

    # record the complete results
    score_array = np.zeros((5, num_fold))

    for iexp, name in enumerate(exp_names):

        # data, output, logs, results
        cfg.DATASET.DATA_DIR = os.path.join(cfg.DATASET.CKPT_DIR, 'data', name)
        cfg.BASIC.BACKUP_DIR = os.path.join(cfg.DATASET.CKPT_DIR, name, 'backup')
        cfg.BASIC.LOG_DIR = os.path.join(cfg.DATASET.CKPT_DIR, name, 'log')
        cfg.TRAIN.OUTPUT_DIR = os.path.join(cfg.DATASET.CKPT_DIR, name, 'output')
        cfg.TEST.RESULT_DIR = os.path.join(cfg.DATASET.CKPT_DIR, name, 'result')

        if cfg.BASIC.SHOW_CFG:
            pprint.pprint(cfg)
        # prepare running environment for the whole project
        prepare_env(cfg)

        # log
        writer = SummaryWriter(log_dir=os.path.join(cfg.BASIC.LOG_DIR))

        # dataloader
        train_dset = ClinicalDataset(cfg, cfg.DATASET.TRAIN_SPLIT)
        train_loader = DataLoader(train_dset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True,
                                  num_workers=cfg.BASIC.WORKERS, pin_memory=cfg.BASIC.PIN_MEMORY)
        val_dset = ClinicalDataset(cfg, cfg.DATASET.VAL_SPLIT)
        val_loader = DataLoader(val_dset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False,
                                num_workers=cfg.BASIC.WORKERS, pin_memory=cfg.BASIC.PIN_MEMORY)

        # network
        model = IENet(cfg)
        model.to(device)

        # optimizer
        optimizer = optim.Adam(model.parameters(), lr=cfg.TRAIN.LR, betas=cfg.TRAIN.BETAS, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
        # criterion
        criterion = torch.nn.BCELoss()

        best_metric = 0
        best_record = np.zeros(5)
        for epoch in range(1, cfg.TRAIN.EPOCH_NUM+1):
            print('Epoch: %d:' % epoch)
            loss_average = train(cfg, train_loader, model, optimizer, criterion)

            writer.add_scalar('train_loss', loss_average, epoch)
            if cfg.BASIC.VERBOSE:
                print('training loss: %f' % loss_average)

            # decay learning rate
            if epoch in cfg.TRAIN.LR_DECAY_EPOCHS:
                decay_lr(optimizer, factor=cfg.TRAIN.LR_DECAY_FACTOR)

            if epoch % cfg.TEST.EVAL_INTERVAL == 0:
                acc, recall, auc, precision, f1 = evaluate(cfg, val_loader, model)
                # value_metric = (precision + recall) / 2
                value_metric = f1

                # save model
                if best_metric < value_metric:
                    info = [epoch, acc, recall, auc, precision, f1]
                    save_best_record_txt(info, os.path.join(cfg.TEST.RESULT_DIR, "best_record.txt"))
                    save_best_model(cfg, epoch=epoch, model=model, optimizer=optimizer)
                    best_metric = value_metric
                    # record the socre
                    best_record[0] = acc
                    best_record[1] = recall
                    best_record[2] = auc
                    best_record[3] = precision
                    best_record[4] = f1

        writer.close()
        # record the best score
        score_array[:, iexp] = best_record

    # save score array to txt
    txt_file = os.path.join(cfg.DATASET.CKPT_DIR, 'score.txt')
    np.savetxt(txt_file, score_array, fmt='%1.4f')


if __name__ == '__main__':
    main()
