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
from tqdm import tqdm_notebook as tqdm


def main(args):

    df = pd.read_csv(args.train)

    use_col = df.columns

    un_use_col = ['id','y','log_y','location', 'access', 'layout', 'age', 'direction', 'area','floor', 'bath_toilet', 'kitchen',
                 'broadcast_com', 'facilities','parking', 'enviroment', 'structure', 'contract_period',
                 'walk_time','23ku',
                #  'area_num_countall','floor_countall','room_num_countall','facilities_countall',
                ]

    use_col = [c for c in use_col if c not in un_use_col]

    print(use_col)

    X, y = df.loc[:, use_col], df['log_y']
    lgb_train = lgb.Dataset(X, y)

    print('loading csv is completed!')

    def objective(trial):

        drop_rate = trial.suggest_uniform('drop_rate', 0, 1.0)
        learning_rate = trial.suggest_uniform('learning_rate', 0, 1.0)
        subsample = trial.suggest_uniform('subsample', 0.8, 1.0)
        num_leaves = trial.suggest_int('num_leaves', 10, 1000)
        max_depth = trial.suggest_int('max_depth', 3, 15)
        min_data_in_leaf = trial.suggest_int('min_data_in_leaf', 2, 100)
        reg_lambda = trial.suggest_loguniform('reg_lambda', 1e-3, 1e4)
        reg_alpha = trial.suggest_loguniform('reg_alpha', 1e-3, 1e4)
        colsample_bytree = trial.suggest_discrete_uniform('colsample_bytree', 0.5, 0.9, .1)
        min_child_weight = trial.suggest_int('min_child_weight',5,40)


        lgbm_params = {
            'task': 'train',
            "metrics": 'rmse',
            'boosting_type': 'gbdt',
            'objective': 'regression',
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "reg_lambda": reg_lambda,
            "reg_alpha": reg_alpha,
            "colsample_bytree": colsample_bytree,
            "min_child_weight": min_child_weight,
            "subsample": subsample,
            "max_bin": 256,
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

    submit_file = pd.concat([test_df['id'], pd.Series(np.exp(y_pred))], axis=1)
    submit_file.columns = [0, 1]
    submit_file[1] = submit_file[1].astype(np.int64)
    submit_file.to_csv(args.submit, header=False, index=False)

    print('---------------------------------')
    print('saving model')
    print('---------------------------------')
    print()

    mdl.save_model(args.mdl_save,num_iteration=mdl.best_iteration)

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

    parser.add_argument(
        '--mdl_save',
        help='path for mdl saving',
        type=str,
        required=True
    )


    parser.add_argument

    args = parser.parse_args()

    print('start loading and training...')
    print()
    main(args)
