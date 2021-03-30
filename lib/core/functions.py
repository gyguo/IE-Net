import os
import torch.backends.cudnn as cudnn
import numpy as np

from utils.utils import fix_random_seed, backup_codes, save_best_record_txt, save_best_model


def fix_random_seed_all(cfg):
    # fix random seed
    fix_random_seed(cfg.BASIC.SEED)
    # cudnn
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    cudnn.enabled = cfg.CUDNN.ENABLE


def prepare_env(cfg):
    # create output directory
    if cfg.BASIC.CREATE_OUTPUT_DIR:
        out_dir = cfg.TRAIN.OUTPUT_DIR
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        # create directory for prediction
        out_dir = cfg.TEST.RESULT_DIR
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    # backup codes
    if cfg.BASIC.BACKUP_CODES:
        backup_dir = cfg.BASIC.BACKUP_DIR
        backup_codes(cfg, cfg.BASIC.ROOT_DIR, backup_dir, cfg.BASIC.BACKUP_LIST)
