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

    df = pd.read_csv('~/Workspace/learning/signate/mynavi_2019/processed_data/train_v3.csv')
    df = df.loc[:,['id', 'age','area','floor', 'structure', 'max_floor',
        'room_num', 'L', 'D', 'K', 'S', 'north', 'min_time','avg_time','23ku_mean_std','log_y','y']]

    X, y = df.loc[:,'age':'23ku_mean_std'], df['y']
    lgb_train = lgb.Dataset(X, y)

    print('loading csv is completed!')

    def objective(trial):  

        drop_rate = trial.suggest_uniform('drop_rate', 0, 1.0)
        learning_rate = trial.suggest_uniform('learning_rate', 0, 1.0)
        subsample = trial.suggest_uniform('subsample', 0.8, 1.0)
        num_leaves = trial.suggest_int('num_leaves', 10, 1000)
        max_depth = trial.suggest_int('max_depth',1,20)
        min_data_in_leaf = trial.suggest_int('min_data_in_leaf', 2, 100)
        # min_child_samples = trial.suggest_int('min_child_samples', 5, 500)
        # min_child_weight = trial.suggest_int('min_child_weight', 5, 500)

        lgbm_params = {
            'task' : 'train',
            'boosting_type' : 'gbdt',
            'objective' : 'regression',
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
            "seed":0
        }

        # init_params = {
        #             'drop_rate': 0.06454829515920846, 
        #             'learning_rate': 0.2880705152639411, 
        #             'subsample': 0.9587507216075405, 
        #             'num_leaves': 168, 
        #             'max_depth': 19, 
        #             'min_data_in_leaf': 7
        #             }
    
        cv_results = lgb.cv(lgbm_params, lgb_train, nfold=args.k_fold,stratified=False)

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

    mdl = lgb.train(study.best_params,lgb_train)

    test_df = pd.read_csv('~/Workspace/learning/signate/mynavi_2019/processed_data/test_v3.csv')
    input_df = test_df.loc[:,['age','area','floor', 'structure', 'max_floor',
        'room_num', 'L', 'D', 'K', 'S', 'north', 'min_time','avg_time','23ku_mean_std']]

    y_pred = mdl.predict(input_df,num_iteration=mdl.best_iteration)

    submit_file = pd.concat([test_df['id'],pd.Series(y_pred)],axis=1)
    submit_file.to_csv('~/Workspace/learning/signate/mynavi_2019/submit/0930_submit.csv',header=False,index=False)

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

    # parser.add_argument(
    #     '--output_file',
    #     '-output',
    #     help ='csv path for submition',
    #     type=str,
    #     required=True
    # )

    args = parser.parse_args()

    print('start loading and training...')
    print()
    main(args)