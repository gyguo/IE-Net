import argparse
import os
import random
import shutil
import copy


def parser_args():
    parser = argparse.ArgumentParser(description='devide dataset into train and validation')
    parser.add_argument('-train_perc', default=0.8)
    parser.add_argument('-pos_dir', default='/disk3/yangle/diagnose/code/data/feature_original/positive')
    parser.add_argument('-neg_dir', default='/disk3/yangle/diagnose/code/data/feature_original/negative')
    parser.add_argument('-train_dir', default='/disk3/yangle/diagnose/code/data')
    parser.add_argument('-val_dir', default='/disk3/yangle/diagnose/code/data')
    args = parser.parse_args()
    return args


def prepare_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir


def copy_file(name_set, ori_dir, res_dir):
    for name in name_set:
        shutil.copyfile(os.path.join(ori_dir, name), os.path.join(res_dir, name))
    return


def devide_data(args, file_dir):
    file_set = os.listdir(file_dir)
    num_sel = int(round(args.train_perc * len(file_set)))
    train_set = random.sample(file_set, num_sel)
    copy_file(train_set, file_dir, args.train_dir)
    val_set = [s for s in file_set if s not in train_set]
    copy_file(val_set, file_dir, args.val_dir)
    return


def separate_names(name_set, num_set):
    name_set_ori = copy.deepcopy(name_set)
    val_sepas = list()

    num_per_set = int(round(len(name_set) / num_set))
    for i in range(num_set - 1):
        sel_names = random.sample(name_set, num_per_set)
        val_sepas.append(sel_names)
        name_set = [n for n in name_set if n not in sel_names]
    # dispose the last set
    val_sepas.append(name_set)

    all_n_dev = list()
    for i in range(num_set):
        val_names = val_sepas[i]
        train_names = [n for n in name_set_ori if n not in val_names]
        all_n_dev.append([train_names, val_names])

    return all_n_dev


def devide_multiple_sets(num_set, exp_names, seed_name):
    args = parser_args()
    train_dir_ori = copy.deepcopy(args.train_dir)
    val_dir_ori = copy.deepcopy(args.val_dir)
    # update args
    args.train_perc = 1 - 1 / num_set
    # exp_names = [str(i) for i in range(num_set)]
    # exp_names = ['exp_'+s for s in exp_names]

    # dispose positive
    file_set = os.listdir(args.pos_dir)
    all_pos_dev = separate_names(file_set, num_set)
    # dispose negative
    file_set = os.listdir(args.neg_dir)
    all_neg_dev = separate_names(file_set, num_set)

    for exp_name, pos_dev, neg_dev in zip(exp_names, all_pos_dev, all_neg_dev):
        print(exp_name)
        # update and make directory
        args.train_dir = os.path.join(train_dir_ori, seed_name, exp_name, 'train')
        args.val_dir = os.path.join(val_dir_ori, seed_name, exp_name, 'val')
        prepare_dir(args.train_dir)
        prepare_dir(args.val_dir)

        # dispose positive, train
        copy_file(pos_dev[0], args.pos_dir, args.train_dir)
        # dispose positive, val
        copy_file(pos_dev[1], args.pos_dir, args.val_dir)
        # dispose negative, train
        copy_file(neg_dev[0], args.neg_dir, args.train_dir)
        # dispose negative, val
        copy_file(neg_dev[1], args.neg_dir, args.val_dir)

    return


if __name__ == '__main__':
    # args = parser_args()
    # prepare_dir(args.train_dir)
    # prepare_dir(args.val_dir)
    # devide_data(args, args.pos_dir)
    # devide_data(args, args.neg_dir)
    num_set = 10
    devide_multiple_sets(num_set)
