import argparse
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, recall_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


def parser_args():
    parser = argparse.ArgumentParser(description='comparison experiments')
    parser.add_argument('-excel_file', default='./dataset_fill0.xlsx')
    parser.add_argument('-num_fold', default=10)
    parser.add_argument('-seeds', default=[0, 6])
    parser.add_argument('-res_dir', default='./result_score')
    args = parser.parse_args()
    return args


def run_experiments(args):
    df_to_select = pd.read_excel(args.excel_file)

    Y = df_to_select['target']
    X = df_to_select.drop(columns='target', axis=1)

    for seed in range(args.seeds[0], args.seeds[1]):
        print('Experiment with seed %d' % seed)
        num_fold = args.num_fold
        record_scores = np.zeros((6, num_fold))

        np.random.seed(seed)

        count = 0
        kf = KFold(n_splits=num_fold, shuffle=True, random_state=seed)
        for train_index, test_index in kf.split(X):
            print('fold %d' % count)
            x_train = X.loc[train_index, :]
            y_train = Y.loc[train_index]
            x_test = X.loc[test_index, :]
            y_test = Y.loc[test_index]

            # GradientBoost
            gbc = GradientBoostingClassifier(loss='exponential', learning_rate=0.05, n_estimators=500, subsample=1.0, random_state=seed)
            gbc.fit(x_train, y_train)

            # Test
            y_pred_gbc_test = gbc.predict(x_test)

            #Test
            acc_gbc_test = round(accuracy_score(y_pred_gbc_test, y_test) * 100, 2)
            print('GradientBoosting: ', acc_gbc_test)
            recall_gbc_test = round(recall_score(y_pred_gbc_test, y_test) * 100, 2)
            print('GradientBoosting: ', recall_gbc_test)
            record_scores[0, count] = acc_gbc_test
            record_scores[1, count] = recall_gbc_test

            # Random Forest
            rndforest = RandomForestClassifier(n_estimators=30, criterion='gini', max_depth=10, min_samples_split=2, min_samples_leaf=1, n_jobs=100, random_state=seed)
            rndforest.fit(x_train, y_train)

            # Test
            y_pred_rndforest_test = rndforest.predict(x_test)

            #Test
            acc_rndforest_test = round(accuracy_score(y_pred_rndforest_test, y_test) * 100, 2)
            print('RandomForest: ', acc_rndforest_test)
            recall_rndforest_test = round(recall_score(y_pred_rndforest_test, y_test) * 100, 2)
            print('RandomForest: ', recall_rndforest_test)
            record_scores[2, count] = acc_rndforest_test
            record_scores[3, count] = recall_rndforest_test


            # MLP
            clf1 = MLPClassifier(solver='adam', activation='logistic', alpha=1e-3, hidden_layer_sizes=(40, 4), random_state=seed)
            clf1.fit(x_train, y_train)

            # Test
            y_pred_nn1_test = clf1.predict(x_test)

            # Test
            acc_nn1_test = round(accuracy_score(y_pred_nn1_test, y_test) * 100, 2)
            print('MLP: ', acc_nn1_test)
            recall_nn1_test = round(recall_score(y_pred_nn1_test, y_test) * 100, 2)
            print('MLP: ', recall_nn1_test)
            record_scores[4, count] = acc_nn1_test
            record_scores[5, count] = recall_nn1_test

            count += 1

        print(record_scores)
        record_file = os.path.join(args.res_dir, 'score_seed'+str(seed)+'.txt')
        np.savetxt(record_file, record_scores)


if __name__ == '__main__':
    args = parser_args()
    run_experiments(args)
