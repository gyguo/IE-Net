import torch
import numpy as np
import random
import os
import shutil
import math
import torch.nn.init as init
import torch.nn as nn


#
# def weight_init(m):
#     if isinstance(m, nn.Conv1d):
#         init.kaiming_uniform_(m.weight, a=math.sqrt(5))
#         if m.bias is not None:
#             fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
#             bound = 1 / math.sqrt(fan_in)
#             init.uniform_(m.bias, -bound, bound)

def fix_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)


def backup_codes(cfg, root_dir, res_dir, backup_list):
    if os.path.exists(res_dir):
        shutil.rmtree(res_dir)
    os.makedirs(res_dir)
    for name in backup_list:
        shutil.copytree(os.path.join(root_dir, name), os.path.join(res_dir, name))
    if cfg.BASIC.VERBOSE:
        print('codes backup at {}'.format(os.path.join(res_dir, name)))


def decay_lr(optimizer, factor):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= factor


def save_best_model(cfg, epoch, model, optimizer):
    state = {'epoch': epoch,
             'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict()}
    save_name = 'checkpoint_'+'best'+'.pth'
    save_file = os.path.join(cfg.BASIC.ROOT_DIR, cfg.TRAIN.OUTPUT_DIR, save_name)
    torch.save(state, save_file)
    if cfg.BASIC.VERBOSE:
        print('save model: %s' % save_file)
    return save_file


def load_weights(model, weight_file):
    checkpoint = torch.load(weight_file)
    model.load_state_dict(checkpoint['state_dict'])

    return model


def save_best_record_txt(info, file_path):
    epoch, test_acc, recay, value_metric = info

    with open(file_path, "w") as f:
        f.write("Epoch: {}\n".format(epoch))
        f.write("test_acc: {:.4f}\n".format(test_acc))
        f.write("recay: {:.4f}\n".format(recay))
        f.write("value_metric: {:.4f}\n".format(value_metric))
