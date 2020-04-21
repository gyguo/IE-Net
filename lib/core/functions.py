import os
import torch.backends.cudnn as cudnn
import numpy as np

from utils.utils import fix_random_seed, backup_codes, save_best_record_txt, save_best_model


def prepare_env(cfg):
    # fix random seed
    fix_random_seed(cfg.BASIC.SEED)
    # create output directory
    if cfg.BASIC.CREATE_OUTPUT_DIR:
        out_dir = os.path.join(cfg.BASIC.ROOT_DIR, cfg.TRAIN.OUTPUT_DIR)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        # create directory for prediction
        out_dir = os.path.join(cfg.BASIC.ROOT_DIR, cfg.TEST.RESULT_DIR)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    # backup codes
    if cfg.BASIC.BACKUP_CODES:
        backup_dir = os.path.join(cfg.BASIC.ROOT_DIR, cfg.BASIC.BACKUP_DIR)
        backup_codes(cfg, cfg.BASIC.ROOT_DIR, backup_dir, cfg.BASIC.BACKUP_LIST)
    # cudnn
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    cudnn.enabled = cfg.CUDNN.ENABLE


def evaluate_mAP(cfg, json_path, gt_path, verbose):
    tIoU_thresh = np.array(cfg.TEST.IOU_TH)
    anet_detection = ANETdetection(gt_path, json_path,
                                   subset='test', tiou_thresholds=tIoU_thresh,
                                   verbose=verbose, check_status=False)
    mAP, average_mAP = anet_detection.evaluate()

    if verbose:
        for i in range(tIoU_thresh.shape[0]):
            print(tIoU_thresh[i], mAP[i])
    return mAP, average_mAP


def post_process(cfg, actions_json_file, test_acc, writer, model, optimizer, best_mAP, epoch, name):
    mAP, average_mAP = evaluate_mAP(cfg, actions_json_file, os.path.join(cfg.BASIC.ROOT_DIR, cfg.DATASET.GT_FILE), cfg.BASIC.VERBOSE)
    for i in range(len(cfg.TEST.IOU_TH)):
        writer.add_scalar('z_mAP@{}/{}'.format(cfg.TEST.IOU_TH[i], name), mAP[i], epoch)
    writer.add_scalar('Average mAP/{}'.format(name), average_mAP, epoch)

    # use mAP@0.5 as the metric
    mAP_5 = mAP[4]
    if mAP_5 > best_mAP:
        best_mAP = mAP_5
        info = [epoch, test_acc, average_mAP, mAP]
        save_best_record_txt(cfg, info, os.path.join(cfg.BASIC.ROOT_DIR, cfg.TEST.RESULT_DIR, "best_record_{}.txt".format(name)))
        save_best_model(cfg, epoch=epoch, model=model, optimizer=optimizer, name=name)

    if cfg.BASIC.VERBOSE:
        print('test_acc %f' % test_acc)
    writer.add_scalar('test_acc/{}'.format(name), test_acc, epoch)

    return writer, best_mAP
