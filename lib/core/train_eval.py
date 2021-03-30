import torch

dtype = torch.cuda.FloatTensor() if torch.cuda.is_available() else torch.FloatTensor()
dtypel = torch.cuda.LongTensor() if torch.cuda.is_available() else torch.LongTensor()

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score


def train(cfg, data_loader, model, optimizer, criterion):
    model.train()

    loss_record = 0

    for features, num_feat, cls_label in data_loader:
        # use batch size = 1
        assert features.size(0) == 1
        features = torch.squeeze(features, dim=0)  # [90, 97]
        num_feat = num_feat.item()
        features = features[:num_feat, :]  # [N, 97]
        features = features.type_as(dtype)
        cls_label = cls_label.type_as(dtype)
        cls_label = torch.unsqueeze(cls_label, dim=0)  # [1]

        score = model(features)
        loss = criterion(score, cls_label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_record += loss.item()

    loss_average = loss_record / len(data_loader)

    return loss_average


def evaluate(cfg, data_loader, model):
    model.eval()

    score_all = []
    result_all = []
    label_all = []
    for features, num_feat, cls_label in data_loader:
        # use batch size = 1
        assert features.size(0) == 1
        features = torch.squeeze(features, dim=0)  # [90, 97]
        num_feat = num_feat.item()
        features = features[:num_feat, :]  # [N, 97]
        features = features.type_as(dtype)
        cls_label = torch.unsqueeze(cls_label, dim=0)  # [1]

        score = model(features)

        score_np = score.data.cpu().numpy()
        cls_label_np = cls_label.numpy()

        # record
        label_all.append(cls_label_np[0][0])
        score_all.append(score_np)
        if score_np >= cfg.TEST.CLS_SCORE_TH:
            result_all.append(1)
        else:
            result_all.append(0)

    acc_test = round(accuracy_score(result_all, label_all) * 100, 2)
    print('\tacc: ', acc_test)
    recall_test = round(recall_score(result_all, label_all) * 100, 2)
    print('\trecall: ', recall_test)
    try:
        auc_test = round(roc_auc_score(result_all, label_all) * 100, 2)
    except BaseException:
        auc_test = 0
    print('\tauc: ', auc_test)
    precision_test = round(precision_score(result_all, label_all) * 100, 2)
    print('\tprecision: ', precision_test)
    f1_test = round(f1_score(result_all, label_all) * 100, 2)
    print('\tf1: ', f1_test)

    return acc_test, recall_test, auc_test, precision_test, f1_test
