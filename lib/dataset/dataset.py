from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import random


class ClinicalDataset(Dataset):
    def __init__(self, cfg, split):
        self.split = split
        self.train_split = cfg.DATASET.TRAIN_SPLIT
        self.base_dir = os.path.join(cfg.DATASET.DATA_DIR, split)
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
