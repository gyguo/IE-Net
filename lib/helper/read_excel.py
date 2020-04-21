import argparse
import pandas as pd
import numpy as np


def args_parser():
    parser = argparse.ArgumentParser(description='dispose raw excell data')
    parser.add_argument('-ori_file', default='/disk3/yangle/diagnose/code/data/dataset.xlsx')
    parser.add_argument('-res_file', default='/disk3/yangle/diagnose/code/data/dataset_filter.xlsx')
    parser.add_argument('-num_ele', default=13)
    parser.add_argument('-num_pos', default=4)
    args = parser.parse_args()
    return args


def check_data_float(df):
    for col in df.columns:
        if col == 'Patient ID':
            continue
        df_col = df[col]
        df_col_list = df_col.to_list()
        data_sel = [i for i in df_col_list if i == i]

        # statistic information
        data_float = [float(i) for i in data_sel]
        data_np = np.array(data_float)
        print('%s,%f,%f,%f,%f,%f' % (col, data_np.mean(), data_np.std(), data_np.min(), data_np.max(), np.median(data_np)))
    return


def adjust_column_order(args, df):
    info = list()
    for col in df.columns:
        if col == 'Patient ID':
            continue
        df_col = df[col]
        df_col_list = df_col.to_list()
        data_sel = [i for i in df_col_list if i == i]
        info.append([col, len(data_sel)])

    info_sort = sorted(info, key=lambda x: x[1])

    exam_num_set = list()
    for item in info_sort:
        num = item[1]
        if num not in exam_num_set:
            exam_num_set.append(num)

    exam_num_set = exam_num_set[::-1]
    exam_name = list()
    for num in exam_num_set:
        sel_names = [item[0] for item in info_sort if item[1] == num]
        sel_names.sort()
        exam_name.extend(sel_names)
    exam_name.insert(0, 'Patient ID')

    df_new = df[exam_name]
    df_new.to_excel(args.res_file_path, index='Patient ID')
    return


def filter_item(args, df):
    row_list = list()
    for idx, row in df.iterrows():
        data = row.to_list()
        data_sel = [i for i in data if i == i]
        data_bin = [float(bool(i)) for i in data_sel]
        data_bin_sum = np.sum(np.array(data_bin))
        if (len(data_sel) >= args.num_ele) and (data_bin_sum >= args.num_pos):
            # print(idx, len(data_sel))
            row_list.append(idx)
    # df_sel = df.iloc[row_list, :]
    # df_sel.to_excel(args.res_file)

    return


if __name__ == '__main__':
    args = args_parser()
    df = pd.read_excel(args.ori_file)
    # check_data_float(df)
    filter_item(args, df)
