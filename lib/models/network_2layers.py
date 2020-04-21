import torch
import torch.nn as nn


class MissNet(nn.Module):
    def __init__(self, cfg):
        super(MissNet, self).__init__()
        self.softmax = nn.Softmax(dim=0)
        self.sigmoid = nn.Sigmoid()
        self.fc_feat1 = nn.Linear(in_features=cfg.NETWORK.DATA_DIM, out_features=cfg.NETWORK.FEATURE_DIM)
        self.fc_feat2 = nn.Linear(in_features=cfg.NETWORK.FEATURE_DIM, out_features=cfg.NETWORK.FEATURE_DIM)
        self.fc_cls1 = nn.Linear(in_features=cfg.NETWORK.FEATURE_DIM, out_features=cfg.NETWORK.FEATURE_DIM)
        self.fc_cls2 = nn.Linear(in_features=cfg.NETWORK.FEATURE_DIM, out_features=cfg.NETWORK.PRED_DIM)
        self.embedding_vector = nn.Parameter(torch.randn((cfg.NETWORK.FEATURE_DIM, 1))).float().cuda()
        self.dropout = nn.Dropout(p=cfg.NETWORK.DROPOUT)
        self.lrelu = nn.LeakyReLU()
        
    def forward(self, x):
        feature1 = self.lrelu(self.fc_feat1(x))  # [N, D]
        feature = self.fc_feat2(feature1)
        # unsqueeze
        feature_uns = torch.unsqueeze(feature, dim=0)
        embedding = torch.unsqueeze(self.embedding_vector, dim=0)
        weight = torch.matmul(feature_uns, embedding)  # [1, N, 1]
        weight = torch.squeeze(weight, dim=0)  # [N, 1]
        # weight_norm = self.softmax(weight)

        feature_wei = feature * weight  # [N, D]
        feature_agg = torch.sum(feature_wei, dim=0, keepdim=True)  # [1, D]
        feature_cls = self.lrelu(self.fc_cls1(feature_agg))
        # dropout
        feature_per = self.dropout(feature_cls)

        score = self.fc_cls2(feature_per)  # [1, 1]
        score = torch.squeeze(score, dim=0)
        score = self.sigmoid(score)  # we calculate BCEloss, thus use Sigmoid activation
        return score
        
        
if __name__ == '__main__':
    import sys
    sys.path.insert(0, '/disk3/yangle/diagnose/code/lib')
    from config.default import config as cfg
    from config.default import update_config

    cfg_file = '/disk3/yangle/diagnose/code/experiments/MissNet.yaml'
    update_config(cfg_file)

    data = torch.randn((10, 97)).cuda()
    network = MissNet(cfg).cuda()

    score = network(data)
    print(score.size(), score)
