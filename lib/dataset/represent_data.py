import argparse
import pandas as pd
import numpy as np
import os


def args_parser():
    parser = argparse.ArgumentParser(description='dispose raw excell data')
    parser.add_argument('-ori_file', default='/disk3/yangle/diagnose/code/data/dataset.xlsx')
    parser.add_argument('-num_feat', default=98)
    parser.add_argument('-max_item', default=90)
    parser.add_argument('-res_dir', default='/disk3/yangle/diagnose/code/data/feature')
    args = parser.parse_args()
    return args


def name_encoder(args, df):
    '''
    create one-hot encode for each feature name
    this is used to concatenate with the value
    '''
    col_names = df.columns.to_list()
    col_names = col_names[2:]

    dict_label_encode = dict()
    for idx, item in enumerate(col_names):
        label = np.zeros(args.num_feat)
        label[idx] = 1
        dict_label_encode[item] = label

    return col_names, dict_label_encode


def main():
    args = args_parser()
    df = pd.read_excel(args.ori_file)
    col_names, dict_label_encode = name_encoder(args, df)

    # iter over row
    for idx, row in df.iterrows():
        datas = row.to_list()
        patient_id = datas[0]
        cls_label = int(datas[1])
        # feature is: [name_encode, value]
        feature = np.zeros((args.max_item, args.num_feat+1))
        feat_count = 0
        for place, value in enumerate(datas[2:]):
            # not nan
            if value == value:
                name_encode = dict_label_encode[col_names[place]]
                feature[feat_count, :-1] = name_encode
                feature[feat_count, -1] = value
                feat_count += 1
                print(col_names[place], name_encode, value)
        res_file = os.path.join(args.res_dir, str(patient_id)+'.npz')
        np.savez(res_file, cls_label=cls_label, feature=feature, num_feat=feat_count)


if __name__ == '__main__':
    main()




