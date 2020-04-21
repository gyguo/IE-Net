import torch
import numpy as np


dtype = torch.cuda.FloatTensor() if torch.cuda.is_available() else torch.FloatTensor()
dtypel = torch.cuda.LongTensor() if torch.cuda.is_available() else torch.LongTensor()


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

    num_correct = 0
    num_positive_correct = 0
    num_total = 0
    num_positive = 0
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
        if cls_label_np == 1:
            num_positive += 1

        # True positive
        if (score_np >= cfg.TEST.CLS_SCORE_TH) and (cls_label_np == 1):
            num_correct += 1
            num_positive_correct += 1
        # True negative
        elif (score_np < cfg.TEST.CLS_SCORE_TH) and (cls_label_np == 0):
            num_correct += 1

        num_total += 1

    # accuracy
    test_acc = num_correct / num_total
    print('correct: %f, total: %f, test_acc: %f' % (num_correct, num_total, test_acc))
    # recay
    recay = num_positive_correct / num_positive
    print('correct: %f, total positive: %f, recay: %f' % (num_positive_correct, num_positive, recay))

    return test_acc, recay

