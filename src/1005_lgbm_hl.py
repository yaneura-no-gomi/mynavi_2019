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

from termcolor import cprint


def train_and_pred_low(train,test,use_col,args):

    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    cprint("-" * 50, "blue")
    cprint(("Low: Train and Predict").center(50), "blue")
    cprint("-" * 50, "blue")

    X, y = train.loc[:, use_col], train['log_y']
    X_train, X_test,y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    lgb_train = lgb.Dataset(X_train, y_train)

    print('loading csv is completed!')

    def objective(trial):

        drop_rate = trial.suggest_uniform('drop_rate', 0, 1.0)
        learning_rate = trial.suggest_uniform('learning_rate', 0, 1.0)
        # subsample = trial.suggest_uniform('subsample', 0.6, 1.0)
        num_leaves = trial.suggest_int('num_leaves', 10, 2**8)
        max_depth = trial.suggest_int('max_depth', 3, 8)
        min_data_in_leaf = trial.suggest_int('min_data_in_leaf', 2, 1000)
        reg_lambda = trial.suggest_loguniform('reg_lambda', 1e-4, 1e3)
        reg_alpha = trial.suggest_loguniform('reg_alpha', 1e-4, 1e3)
        min_split_gain = trial.suggest_loguniform('min_split_gain', 1e-4, 1e3)
        colsample_bytree = trial.suggest_uniform('colsample_bytree', 0.3, 1.0)
        min_child_weight = trial.suggest_int('min_child_weight',5,50)


        lgbm_params = {
            'task': 'train',
            "metrics": 'rmse',
            'boosting_type': 'gbdt',
            'objective': 'regression',
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "reg_lambda": reg_lambda,
            "reg_alpha": reg_alpha,
            "min_split_gain": min_split_gain,
            "colsample_bytree": colsample_bytree,
            "min_child_weight": min_child_weight,
            # "subsample": subsample,
            # "max_bin": 1024,
            "drop_rate": drop_rate,
            "max_depth": max_depth,
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
    print('best_params: ', study.best_params)
    print('---------------------------------')
    print()

    print('---------------------------------')
    print('reference score')
    
    mdl = lgb.train(study.best_params, lgb_train)
    pred_train = mdl.predict(X_train,num_iteration=mdl.best_iteration)
    rmse_train = np.sqrt(mean_squared_error(np.exp(y_train), np.exp(pred_train)))
    pred_test = mdl.predict(X_test,num_iteration=mdl.best_iteration)
    rmse_test = np.sqrt(mean_squared_error(np.exp(y_test), np.exp(pred_test)))

    print('train score: ',rmse_train)
    print('test score: ',rmse_test)
    print('---------------------------------')
    print()

    print('---------------------------------')
    print('predict test data...')

    X, y = train.loc[:, use_col], train['log_y']
    lgb_train = lgb.Dataset(X, y)

    mdl = lgb.train(study.best_params, lgb_train)
    
    input_df = test.loc[:, use_col]
    y_pred = mdl.predict(input_df, num_iteration=mdl.best_iteration)
    
    submit_file = pd.concat([test['id'], pd.Series(np.exp(y_pred))], axis=1)
    submit_file.columns = [0, 1]

    print(submit_file)
    submit_file[1] = submit_file[1].astype(np.int64)

    print('saving model...')
    print()

    d,e = os.path.splitext(args.mdl_save)
    p = d+'_high'+e
    mdl.save_model(p,num_iteration=mdl.best_iteration)

    print('*********Done!*********')

    return submit_file

def train_and_pred_high(train,test,use_col,args):

    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    cprint("-" * 50, "red")
    cprint(("High: Train and Predict").center(50), "red")
    cprint("-" * 50, "red")

    X, y = train.loc[:, use_col], train['log_y']
    X_train, X_test,y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    lgb_train = lgb.Dataset(X_train, y_train)

    print('loading csv is completed!')

    def objective(trial):

        drop_rate = trial.suggest_uniform('drop_rate', 0, 1.0)
        learning_rate = trial.suggest_uniform('learning_rate', 0, 0.5)
        # subsample = trial.suggest_uniform('subsample', 0.6, 1.0)
        num_leaves = trial.suggest_int('num_leaves', 10, 2**5)
        max_depth = trial.suggest_int('max_depth', 3, 5)
        min_data_in_leaf = trial.suggest_int('min_data_in_leaf', 2, 100)
        reg_lambda = trial.suggest_loguniform('reg_lambda', 1e-4, 1e3)
        reg_alpha = trial.suggest_loguniform('reg_alpha', 1e-4, 1e3)
        min_split_gain = trial.suggest_loguniform('min_split_gain', 1e-4, 1e3)
        colsample_bytree = trial.suggest_uniform('colsample_bytree', 0.3, 1.0)
        min_child_weight = trial.suggest_int('min_child_weight',5,50)

        lgbm_params = {
            'task': 'train',
            "metrics": 'rmse',
            'boosting_type': 'gbdt',
            'objective': 'regression',
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "reg_lambda": reg_lambda,
            "reg_alpha": reg_alpha,
            "min_split_gain": min_split_gain,
            "colsample_bytree": colsample_bytree,
            "min_child_weight": min_child_weight,
            # "subsample": subsample,
            # "max_bin": 1024,
            "drop_rate": drop_rate,
            "max_depth": max_depth,
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
    study.optimize(objective, n_trials=args.n_trials*3)
    print()

    print('---------------------------------')
    print('best trial: ', study.best_trial)
    print('best_params: ', study.best_params)
    print('---------------------------------')
    print()

    print('---------------------------------')
    print('reference score')
    
    mdl = lgb.train(study.best_params, lgb_train)
    pred_train = mdl.predict(X_train,num_iteration=mdl.best_iteration)
    rmse_train = np.sqrt(mean_squared_error(np.exp(y_train), np.exp(pred_train)))
    pred_test = mdl.predict(X_test,num_iteration=mdl.best_iteration)
    rmse_test = np.sqrt(mean_squared_error(np.exp(y_test), np.exp(pred_test)))

    print('train score: ',rmse_train)
    print('test score: ',rmse_test)
    print('---------------------------------')
    print()

    print('---------------------------------')
    print('predict test data...')

    X, y = train.loc[:, use_col], train['log_y']
    lgb_train = lgb.Dataset(X, y)

    mdl = lgb.train(study.best_params, lgb_train)
    
    input_df = test.loc[:, use_col]
    y_pred = mdl.predict(input_df, num_iteration=mdl.best_iteration)

    submit_file = pd.concat([test['id'], pd.Series(np.exp(y_pred))], axis=1)
    print(submit_file)
    submit_file.columns = [0, 1]
    submit_file[1] = submit_file[1].astype(np.int64)

    print('saving model...')
    print()

    d,e = os.path.splitext(args.mdl_save)
    p = d+'_low'+e
    mdl.save_model(p,num_iteration=mdl.best_iteration)

    print('*********Done!*********')

    return submit_file



def main(args):

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)

    use_col = train.columns

    un_use_col = ['id','y','log_y','high_price_flag','prob','label',
                    'location', 'access', 'layout', 'age', 'direction', 'area','floor', 'bath_toilet', 'kitchen',
                    'broadcast_com', 'facilities','parking', 'enviroment', 'structure', 'contract_period',
                    'walk_time','23ku',
                    #  'area_num_countall','floor_countall','room_num_countall','facilities_countall','age_countall','area_num_countall',
                ]

    use_col = [c for c in use_col if c not in un_use_col]

    print(use_col)

    # low_train = train[train['high_price_flag']==0]
    low_train = train
    high_train = train[train['high_price_flag']==1]
    
    low_test = test[test['label']==0]
    high_test = test[test['label']==1]

    print('Low: ',len(low_train),len(low_test))
    print('High: ',len(high_train),len(high_test))

    low_pred = train_and_pred_low(low_train,test[test['label']==0],use_col,args)
    # high_pred = train_and_pred_high(high_train,test,use_col+['prob'],args)
    high_pred = train_and_pred_high(high_train,test[test['label']==1],use_col,args)

    # print(test.loc[:,'prob'])
    # submit_df = low_pred*(1 - test['prob']) + high_pred*test['prob']

    submit_df = pd.concat([low_pred,high_pred]).sort_values(0)
    print(submit_df)
    submit_df.to_csv(args.submit,index=False,header=False)


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
