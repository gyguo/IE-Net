import torch
import numpy as np
import random
import os
import shutil


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
    save_file = os.path.join(cfg.TRAIN.OUTPUT_DIR, save_name)
    torch.save(state, save_file)
    if cfg.BASIC.VERBOSE:
        print('save model: %s' % save_file)
    return save_file


def load_weights(model, weight_file):
    checkpoint = torch.load(weight_file)
    model.load_state_dict(checkpoint['state_dict'])

    return model


def save_best_record_txt(info, file_path):
    epoch, acc, recall, auc, precision, f1 = info

    with open(file_path, "w") as f:
        f.write("Epoch: {}\n".format(epoch))
        f.write("acc: {:.4f}\n".format(acc))
        f.write("recall: {:.4f}\n".format(recall))
        f.write("precision: {:.4f}\n".format(precision))
        f.write("auc: {:.4f}\n".format(auc))
        f.write("f1: {:.4f}\n".format(f1))
