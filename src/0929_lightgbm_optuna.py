import sys
import os
import glob
import re

import pandas as pd
import pandas_profiling as pdp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.svm import SVR
from sklearn import metrics
import optuna
import argparse

def main(args):

    df = pd.read_csv('~/Workspace/learning/signate/mynavi_2019/processed_data/train_v2.csv')
    df = df.loc[:,['id', 'age','area','floor', 'structure', 'max_floor',
        'room_num', 'L', 'D', 'K', 'S', 'north', 'y']]

    X, y = df.loc[:,'age':'north'], df['y']
    lgb_train = lgb.Dataset(X, y)

    print('loading csv is completed!')

    def objective(trial):  

        drop_rate = trial.suggest_uniform('drop_rate', 0, 1.0)
        learning_rate = trial.suggest_uniform('learning_rate', 0, 1.0)
        subsample = trial.suggest_uniform('subsample', 0.8, 1.0)
        num_leaves = trial.suggest_int('num_leaves', 5, 10000)
        num_boost_round = trial.suggest_int('num_boost_round', 10, 10000)
        min_data_in_leaf = trial.suggest_int('min_data_in_leaf', 1, 1000)
        min_child_samples = trial.suggest_int('min_child_samples', 5, 500)
        min_child_weight = trial.suggest_int('min_child_weight', 5, 500)

        lgbm_params = {
            'task' : 'train',
            'boosting_type' : 'gbdt',
            'objective' : 'regression',
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            # "max_bin": 256,
            "metrics": 'rmse',
            "drop_rate": drop_rate,
            "num_boost_round": num_boost_round,
            # "is_unbalance": False,
            # "min_child_samples": min_child_samples,
            # "min_child_weight": min_child_weight,
            "min_split_gain": 0,
            "min_data_in_leaf": min_data_in_leaf,
            # "subsample": subsample
            "n_jobs": 1,
            # "device_type": 'gpu',
            'verbose': -1,
            "seed":0
        }
                
        # print('1')
        # mdl = lgb.train(params, lgb_train,num_boost_round=num_boost_round)
        cv_results = lgb.cv(lgbm_params, lgb_train, nfold=args.k_fold,stratified=False)

        # scores = cv_results['rmse']
        # print(cv_results)
        
        # print('4')
        # score = np.sqrt(abs(scores)).mean()
        # print(len(cv_results['rmse-mean']))
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

    params = study.best_params

    mdl = lgb.train(params,lgb_train)

    test_df = pd.read_csv('~/Workspace/learning/signate/mynavi_2019/processed_data/test_v2.csv')
    input_df = test_df.loc[:,['age','area','floor', 'structure', 'max_floor','room_num', 'L', 'D', 'K', 'S', 'north']]

    y_pred = mdl.predict(input_df,num_iteration=mdl.best_iteration)

    submit_file = pd.concat([test_df['id'],pd.Series(y_pred)],axis=1)
    submit_file.to_csv('~/Workspace/learning/signate/mynavi_2019/submit/0929_submit.csv',header=False,index=False)

    print('Done!')
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="training LightGBM by Optuna")

    parser.add_argument(
        '--n_trials',
        '-n',
        help ='training iteration',
        type=int,
        default=100
    )

    parser.add_argument(
        '--k_fold',
        '-k',
        help ='number of k-fold split',
        type=int,
        default=5
    )

    args = parser.parse_args()

    print('start loading and training...')
    print()
    main(args)