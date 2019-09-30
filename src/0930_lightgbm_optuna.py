import argparse
import glob
import os
import re
import sys

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import pandas_profiling as pdp
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.svm import SVR
from tqdm import tqdm_notebook as tqdm


def main(args):

    df = pd.read_csv(args.train)

    use_col = ['age', 'area', 'floor', 'structure', 'max_floor', 'room_num', 'L', 'D',
                'K', 'S', 'north', 'min_time', 'avg_time', 'bicycle_parking','car_parking','bike_parking','23ku_mean_std',
                'direction_countall']

    df = df.loc[:,use_col + ['log_y', 'y']]

    X, y = df.loc[:, use_col], df['y']
    lgb_train = lgb.Dataset(X, y)

    print('loading csv is completed!')

    def objective(trial):

        drop_rate = trial.suggest_uniform('drop_rate', 0, 1.0)
        learning_rate = trial.suggest_uniform('learning_rate', 0, 1.0)
        subsample = trial.suggest_uniform('subsample', 0.8, 1.0)
        num_leaves = trial.suggest_int('num_leaves', 10, 1000)
        max_depth = trial.suggest_int('max_depth', 1, 50)
        min_data_in_leaf = trial.suggest_int('min_data_in_leaf', 2, 100)
        # min_child_samples = trial.suggest_int('min_child_samples', 5, 500)
        # min_child_weight = trial.suggest_int('min_child_weight', 5, 500)

        lgbm_params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'regression',
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            # "max_bin": 256,
            "metrics": 'rmse',
            "drop_rate": drop_rate,
            "max_depth": max_depth,
            # "min_split_gain": 0,
            "min_data_in_leaf": min_data_in_leaf,
            "n_jobs": 1,
            'verbose': -1,
            "seed": 0
        }


        cv_results = lgb.cv(lgbm_params, lgb_train, nfold=args.k_fold, stratified=False)

        score = np.array(cv_results['rmse-mean']).mean()

        return score

    study = optuna.create_study()

    print('begin training')
    # optuna.logging.enable_default_handler()
    study.optimize(objective, n_trials=args.n_trials)
    print()

    print('---------------------------------')
    print('best trial: ', study.best_trial)
    print('---------------------------------')
    print('best_params: ', study.best_params)
    print('---------------------------------')
    print()

    print('---------------------------------')
    print('predict test data')
    print('---------------------------------')
    print()

    mdl = lgb.train(study.best_params, lgb_train)
    

    test_df = pd.read_csv(args.test)
    input_df = test_df.loc[:, use_col]

    y_pred = mdl.predict(input_df, num_iteration=mdl.best_iteration)

    submit_file = pd.concat([test_df['id'], pd.Series(y_pred)], axis=1)
    submit_file.columns = [0, 1]
    submit_file[1] = submit_file[1].astype(np.int64)
    submit_file.to_csv(args.submit, header=False, index=False)

    print('---------------------------------')
    print('saving model')
    print('---------------------------------')
    print()

    mdl.save_model('mdl/0930_lgbm.txt',num_iteration=mdl.best_iteration)

    print('*********Done!*********')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="training LightGBM by Optuna")

    parser.add_argument(
        '--n_trials',
        '-n',
        help='training iteration',
        type=int,
        default=100
    )

    parser.add_argument(
        '--k_fold',
        '-k',
        help='number of k-fold split',
        type=int,
        default=5
    )

    parser.add_argument(
        '--train',
        help='file path for train data',
        type=str,
        required=True
    )

    parser.add_argument(
        '--test',
        help='file path for test data',
        type=str,
        required=True
    )

    parser.add_argument(
        '--submit',
        help='path for outputing submit file',
        type=str,
        required=True
    )

    parser.add_argument

    args = parser.parse_args()

    print('start loading and training...')
    print()
    main(args)
