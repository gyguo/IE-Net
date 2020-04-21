import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import _init_paths
from config.default import config as cfg
from config.default import update_config
import pprint
from models.network_2layers import MissNet
from dataset.dataset import WtalDataset
from dataset.devide_dataset import devide_multiple_sets
from core.train_eval import train, evaluate
from core.functions import prepare_env

from utils.utils import decay_lr, save_best_model, save_best_record_txt


def args_parser():
    parser = argparse.ArgumentParser(description='weakly supervised action localization baseline')
    parser.add_argument('-cfg', help='Experiment config file', default='../experiments/MissNet.yaml')
    parser.add_argument('-num_fold', default=10)
    parser.add_argument('-seed_name', default='exps_10fold_seed5')
    args = parser.parse_args()
    return args


def main():
    args = args_parser()
    num_fold = args.num_fold
    exp_names = [str(i) for i in range(num_fold)]
    exp_names = ['exp_' + s for s in exp_names]

    # prepare data for all devison
    devide_multiple_sets(num_fold, exp_names, args.seed_name)

    # record the complete results
    score_array = np.zeros((2, num_fold))

    for iexp, name in enumerate(exp_names):
        update_config(args.cfg)
        # update parameters
        new_str = cfg.BASIC.EXP_NAME + '/' + name
        update_config(args.cfg)
        # dataset related
        cfg.DATASET.DATA_DIR = 'data/' + new_str
        # output, logs related
        cfg.BASIC.BACKUP_DIR = cfg.BASIC.BACKUP_DIR.replace('name_tobe_replaced', new_str)
        cfg.BASIC.LOG_DIR = cfg.BASIC.LOG_DIR.replace('name_tobe_replaced', new_str)
        cfg.TRAIN.OUTPUT_DIR = cfg.TRAIN.OUTPUT_DIR.replace('name_tobe_replaced', new_str)
        cfg.TEST.RESULT_DIR = cfg.TEST.RESULT_DIR.replace('name_tobe_replaced', new_str)

        if cfg.BASIC.SHOW_CFG:
            pprint.pprint(cfg)
        # prepare running environment for the whole project
        prepare_env(cfg)

        # log
        writer = SummaryWriter(log_dir=os.path.join(cfg.BASIC.ROOT_DIR, cfg.BASIC.LOG_DIR))

        # dataloader
        train_dset = WtalDataset(cfg, cfg.DATASET.TRAIN_SPLIT)
        train_loader = DataLoader(train_dset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True,
                                  num_workers=cfg.BASIC.WORKERS, pin_memory=cfg.BASIC.PIN_MEMORY)
        val_dset = WtalDataset(cfg, cfg.DATASET.VAL_SPLIT)
        val_loader = DataLoader(val_dset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False,
                                num_workers=cfg.BASIC.WORKERS, pin_memory=cfg.BASIC.PIN_MEMORY)

        # network
        model = MissNet(cfg)
        model.cuda()

        # optimizer
        optimizer = optim.Adam(model.parameters(), lr=cfg.TRAIN.LR, betas=cfg.TRAIN.BETAS, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
        # criterion
        criterion = torch.nn.BCELoss()

        best_metric = 0
        best_record = np.zeros(2)
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
                test_acc, recay = evaluate(cfg, val_loader, model)
                value_metric = (test_acc + recay) / 2
                if cfg.BASIC.VERBOSE:
                    print('test_acc: %f, recay: %f' % (test_acc, recay))
                writer.add_scalar('test_acc', test_acc, epoch)
                writer.add_scalar('recay', recay, epoch)
                writer.add_scalar('mean', value_metric, epoch)

                # save model
                if best_metric < value_metric:
                    info = [epoch, test_acc, recay, value_metric]
                    save_best_record_txt(info, os.path.join(cfg.BASIC.ROOT_DIR, cfg.TEST.RESULT_DIR, "best_record.txt"))
                    save_best_model(cfg, epoch=epoch, model=model, optimizer=optimizer)
                    best_metric = value_metric
                    # record the socre
                    best_record[0] = test_acc
                    best_record[1] = recay

        writer.close()
        # record the best score
        score_array[:, iexp] = best_record

    # save score array to txt
    txt_file = os.path.join(cfg.BASIC.ROOT_DIR, 'output', cfg.BASIC.EXP_NAME, 'score.txt')
    np.savetxt(txt_file, score_array)


if __name__ == '__main__':
    main()
