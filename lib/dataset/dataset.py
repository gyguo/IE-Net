# ------------------------------------------------------------------------------
# Author: Le Yang
# Descriptions: dataloader for the project
#     1.  valuable fields and its meaning
#         cls_label: shape [C], video classification labels.
#             If the video contains action $i$, the $i-1$ place is set to $1$.
#         feat_spa / feat_tem: spatial / temporal feature, scaled to fixed temporal length via linear interpolation
#         vid_name: video name
#         frame_num: total frame number in the video.
#             As for THUMOS14, frames are extracted via original frame rate.
#             As for ActivityNet, frames are extracted via 30 fps.
#         fps: fps for this video
#     2.  We first load all data into memory, so as to speed up the data load process
# ------------------------------------------------------------------------------
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import random


class WtalDataset(Dataset):
    def __init__(self, cfg, split):
        self.split = split
        self.train_split = cfg.DATASET.TRAIN_SPLIT
        self.base_dir = os.path.join(cfg.BASIC.ROOT_DIR, cfg.DATASET.DATA_DIR, split)
        self.datas = self._load_dataset()

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, item):
        file_name = self.datas[item]
        data = np.load(os.path.join(self.base_dir, file_name))
        feature = data['feature']
        num_feat = data['num_feat']
        cls_label = data['cls_label']

        # data augmentation in training: shuffle the input examination
        if self.train_split == 'train':
            num_feat_int = int(num_feat)
            idx = list(range(num_feat_int))
            random.shuffle(idx)
            feature_sel = feature[:num_feat_int, :]
            feature[:num_feat_int, :] = feature_sel[idx, :]

        return feature, num_feat, cls_label

    def _load_dataset(self):
        data_set = os.listdir(self.base_dir)
        data_set.sort()
        datas = [i for i in data_set if i.endswith('.npz')]
        return datas


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '/disk3/yangle/diagnose/code/lib')
    from config.default import config as cfg
    from config.default import update_config
    cfg_file = '/disk3/yangle/diagnose/code/experiments/MissNet.yaml'
    update_config(cfg_file)
    train_dset = WtalDataset(cfg, cfg.DATASET.TRAIN_SPLIT)
    train_loader = DataLoader(train_dset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True,
                              num_workers=cfg.BASIC.WORKERS, pin_memory=cfg.BASIC.PIN_MEMORY)

    for feature, num_feature, cls_label in train_loader:
        print(type(feature), num_feature.size(), feature.size(), cls_label)
