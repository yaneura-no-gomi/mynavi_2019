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
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.svm import SVR
from sklearn import metrics
import optuna
import argparse

import warnings
warnings.filterwarnings('ignore')

def main(args):

    df = pd.read_csv('~/Workspace/learning/signate/mynavi_2019/processed_data/train_v2.csv')
    df = df.loc[:,['id', 'age','area','floor', 'structure', 'max_floor',
        'room_num', 'L', 'D', 'K', 'S', 'north', 'y']]

    X, y = df.loc[:,'age':'north'], df['y']


    print('loading csv is completed!')

    def objective(trial):  
        params = {
            'kernel': trial.suggest_categorical('kernel', ['linear','rbf','sigmoid']),
            'C': trial.suggest_loguniform('C', 1e0, 1e+3),
            'gamma': trial.suggest_loguniform('gamma', 1e-3, 1e+2),
            'epsilon':trial.suggest_loguniform('epsilon', 1e-1, 1e1)
        }
        
        # print('1')
        mdl = SVR(**params)

        # print('2')
        kfold = KFold(n_splits=args.k_fold, random_state=0,shuffle=True)

        # print('3')
        scores = cross_val_score(mdl, X, y, cv=kfold,scoring='neg_mean_squared_error',n_jobs=-1)

        # print('4')
        score = np.sqrt(abs(scores)).mean()

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


    mdl = SVR(**study.best_params)
    X_train, X_test,y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    mdl.fit(X_train, y_train)

    pred_train = mdl.predict(X_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, pred_train))
    pred_test = mdl.predict(X_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, pred_test))

    print('---------------------------------')
    print('train score :', rmse_train)
    print('test score :', rmse_test)
    print('---------------------------------')
    print()

    print('---------------------------------')    
    print('predict test data')
    print('---------------------------------')
    print()    

    mdl = SVR(**study.best_params)
    mdl.fit(X, y)

    test_df = pd.read_csv('~/Workspace/learning/signate/mynavi_2019/processed_data/test_v2.csv')
    input_df = test_df.loc[:,['age','area','floor', 'structure', 'max_floor','room_num', 'L', 'D', 'K', 'S', 'north']]

    y_pred = mdl.predict(input_df)

    submit_file = pd.concat([test_df['id'],pd.Series(y_pred)],axis=1)
    submit_file.to_csv('~/Workspace/learning/signate/mynavi_2019/submit/0928_submit_2.csv',header=False,index=False)

    print('Done!')
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="training SVR by Optuna")

    parser.add_argument(
        '--n_trials',
        '-n',
        help ='training iteration',
        type=int,
        default=10
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



