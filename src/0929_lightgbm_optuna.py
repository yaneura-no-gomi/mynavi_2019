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
        num_leaves = trial.suggest_int('num_leaves', 5, 1000)
        # num_boost_round = trial.suggest_int('num_boost_round', 10, 100000)
        num_boost_round = trial.suggest_int('num_boost_round', 10, 100000)
        min_data_in_leaf = trial.suggest_int('min_data_in_leaf', 10, 100000)
        min_child_samples = trial.suggest_int('min_child_samples', 5, 500)
        min_child_weight = trial.suggest_int('min_child_weight', 5, 500)

        lgbm_params = {
            'task' : 'train',
            'boosting_type' : 'gbdt',
            'objective' : 'regression',
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "max_bin": 256,
            "metrics": 'rmse',
            # "drop_rate": drop_rate,
            # "is_unbalance": False,
            # "max_drop": 50,
            # "min_data": 1,
            # "min_child_samples": min_child_samples,
            # "min_child_weight": min_child_weight,
            # "min_split_gain": 0,
            # "min_data_in_leaf": min_data_in_leaf,
            # "subsample": subsample
        }
                
        # print('1')
        # mdl = lgb.train(params, lgb_train,num_boost_round=num_boost_round)
        cv_results = lgb.cv(lgbm_params, lgb_train, nfold=10, num_boost_round=100)

        # scores = cv_results['rmse']
        # print(cv_results)
        
        # print('4')
        # score = np.sqrt(abs(scores)).mean()
        score = cv_results['rmse-mean']


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


    # mdl = SVR(**study.best_params)
    # X_train, X_test,y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    # mdl.fit(X_train, y_train)

    # pred_train = mdl.predict(X_train)
    # rmse_train = np.sqrt(mean_squared_error(y_train, pred_train))
    # pred_test = mdl.predict(X_test)
    # rmse_test = np.sqrt(mean_squared_error(y_test, pred_test))

    # print('---------------------------------')
    # print('train score :', rmse_train)
    # print('test score :', rmse_test)
    # print('---------------------------------')
    # print()

    # print('---------------------------------')    
    # print('predict test data')
    # print('---------------------------------')
    # print()    

    # mdl = SVR(**study.best_params)
    # mdl.fit(X, y)

    # test_df = pd.read_csv('~/Workspace/learning/signate/mynavi_2019/processed_data/test_v2.csv')
    # input_df = test_df.loc[:,['age','area','floor', 'structure', 'max_floor','room_num', 'L', 'D', 'K', 'S', 'north']]

    # y_pred = mdl.predict(input_df)

    # submit_file = pd.concat([test_df['id'],pd.Series(y_pred)],axis=1)
    # submit_file.to_csv('~/Workspace/learning/signate/mynavi_2019/submit/0928_submit_2.csv',header=False,index=False)

    # print('Done!')
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="training SVR by Optuna")

    parser.add_argument(
        '--n_trials',
        '-n',
        help ='training iteration',
        type=int,
        default=10
    )

    # parser.add_argument(
    #     '--k_fold',
    #     '-k',
    #     help ='number of k-fold split',
    #     type=int,
    #     default=5
    # )

    args = parser.parse_args()

    print('start loading and training...')
    print()
    main(args)