from easydict import EasyDict as edict
import yaml
import datetime


config = edict()

# basic
config.BASIC = edict()
config.BASIC.TIME = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
config.BASIC.ROOT_DIR = ''
config.BASIC.NUM_FOLD = 1
config.BASIC.SEED = 0
config.BASIC.LOG_DIR = ''
config.BASIC.CKPT_DIR = ''
config.BASIC.WORKERS = 1
config.BASIC.CREATE_OUTPUT_DIR = False
config.BASIC.PIN_MEMORY = True
config.BASIC.SHOW_CFG = False
config.BASIC.BACKUP_CODES = True
config.BASIC.BACKUP_LIST = ['lib', 'experiments', 'tools']
config.BASIC.BACKUP_DIR = ''
config.BASIC.VERBOSE = False
# digits for saved checkpoint, e.g, 01.pth, 001.pth, etc.
config.BASIC.CHECKPOINT_DIGITS = 1

# CUDNN
config.CUDNN = edict()
config.CUDNN.BENCHMARK = True
config.CUDNN.DETERMINISTIC = False
config.CUDNN.ENABLE = True

# dataset
config.DATASET = edict()
config.DATASET.DATASET_DIR = ''
config.DATASET.DATA_DIR = ''
config.DATASET.TRAIN_SPLIT = ''
config.DATASET.VAL_SPLIT = ''
config.DATASET.CLS_NUM = 1

# network
config.NETWORK = edict()
config.NETWORK.DATA_DIM = 1
config.NETWORK.FEATURE_DIM = 1
config.NETWORK.PRED_DIM = 1
config.NETWORK.DROPOUT = 1

# train
config.TRAIN = edict()
config.TRAIN.LR = 0.0001
config.TRAIN.BETAS = []
config.TRAIN.WEIGHT_DECAY = 0
config.TRAIN.EPOCH_NUM = 1
config.TRAIN.OUTPUT_DIR = ''
config.TRAIN.BATCH_SIZE = 1
config.TRAIN.LR_DECAY_EPOCHS = []
config.TRAIN.LR_DECAY_FACTOR = 1

# test
config.TEST = edict()
config.TEST.BATCH_SIZE = 1
config.TEST.EVAL_INTERVAL = 1
config.TEST.CLS_SCORE_TH = 0.5
config.TEST.RESULT_DIR = ''


def _update_dict(k, v):
    for dk, dv in v.items():
        if dk in config[k]:
            config[k][dk] = dv
        else:
            raise ValueError('{}.{} not exists in config.py'.format(k, dk))


def update_config(cfg_file):
    with open(cfg_file) as f:
        exp_config = edict(yaml.load(f, Loader=yaml.FullLoader))
    for k, v in exp_config.items():
        if k in config:
            if isinstance(v, dict):
                _update_dict(k, v)
            else:
                config[k] = v
        else:
            raise ValueError('{} not exists in config.py'.format(k))


if __name__ == '__main__':
    cfg_file = '../../experiments/IENet.yaml'
    update_config(cfg_file)
    print(config)


