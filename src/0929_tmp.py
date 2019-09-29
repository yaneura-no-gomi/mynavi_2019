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

    
    print('---------------------------------')    
    print('predict test data')
    print('---------------------------------')
    print()    

    params = {'drop_rate': 0.7898608505827744, 'learning_rate': 0.38819893077272155, 'subsample': 0.9407875772392491, 
            'num_leaves': 19, 'num_boost_round': 5477, 'min_data_in_leaf': 1, 'min_child_samples': 377, 
            'min_child_weight': 185}
    
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