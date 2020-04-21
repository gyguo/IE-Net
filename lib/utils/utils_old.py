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


def save_best_model(cfg, epoch, model, optimizer, name):
    state = {'epoch': epoch,
             'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict()}
    save_name = 'checkpoint_'+'best_'+name+'.pth'
    save_file = os.path.join(cfg.BASIC.ROOT_DIR, cfg.TRAIN.OUTPUT_DIR, save_name)
    torch.save(state, save_file)
    if cfg.BASIC.VERBOSE:
        print('save model: %s' % save_file)
    return save_file


def load_weights(model, weight_file):
    checkpoint = torch.load(weight_file)
    model.load_state_dict(checkpoint['state_dict'])

    return model


def save_best_record_txt(cfg, info, file_path):
    epoch, test_acc, average_mAP, mAP = info
    tIoU_thresh = cfg.TEST.IOU_TH

    with open(file_path, "w") as f:
        f.write("Epoch: {}\n".format(epoch))
        f.write("Test_acc: {:.4f}\n".format(test_acc))
        f.write("average_mAP: {:.4f}\n".format(average_mAP))

        for i in range(len(tIoU_thresh)):
            f.write("mAP@{:.1f}: {:.4f}\n".format(tIoU_thresh[i], mAP[i]))
