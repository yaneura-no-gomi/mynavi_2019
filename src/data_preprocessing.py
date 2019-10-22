import sys
import os
import glob
import re
import pickle

import pandas as pd
import pandas_profiling as pdp
import numpy as np
import lightgbm as lgb
import argparse
from sklearn.preprocessing import StandardScaler

from utils.prepro_func import preprocessing_train_test
from utils.high_and_low_clf import High_and_Low_Classifier


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="preprocessing for train and test data")

    parser.add_argument(
        '--output_train',
        help ='output file path for train data',
        type=str,
        required=True
    )

    parser.add_argument(
        '--output_test',
        help ='output file path for test data',
        type=str,
        required=True
    )

    args = parser.parse_args()

    processed_train,processed_test = preprocessing_train_test()

    use_col = processed_train.columns
    un_use_col = ['id','y','log_y','high_price_flag','location', 'access', 'layout', 'age', 'direction', 'area','floor', 'bath_toilet', 'kitchen',
                 'broadcast_com', 'facilities','parking', 'enviroment', 'structure', 'contract_period',
                 'walk_time','23ku','23ku_mean_2',
                #  'area_num_countall','floor_countall','room_num_countall','facilities_countall','age_countall','area_num_countall',
                ]

    mdl = lgb.Booster(model_file='mdl/1011_lgbm.txt')
    feature_importances = pd.DataFrame()
    feature_importances['feature'] = mdl.feature_name()
    feature_importances['importance'] = mdl.feature_importance()
    feature_importances = feature_importances.sort_values(by='importance', ascending=False)

    un_use_col += list(feature_importances[feature_importances['importance']==0]['feature'])

    use_col = [c for c in use_col if c not in un_use_col]

    print(use_col)
    print(len(use_col))

    # # high and low classifier
    # clf = High_and_Low_Classifier(processed_train,use_col)
    # clf.train()
    # clf.pred(processed_test)
    # processed_test['prob'],processed_test['label'] = clf.labeling()

    with open('mdl/1011_hl0.pkl', mode='rb') as f:
        r1 = pickle.load(f)

    with open('mdl/1011_hl1.pkl', mode='rb') as f:
        r2 = pickle.load(f)

    with open('mdl/1011_hl2.pkl', mode='rb') as f:
        r3 = pickle.load(f)

    with open('mdl/1011_hl3.pkl', mode='rb') as f:
        r4 = pickle.load(f)

    pred1 = r1.predict_proba(processed_test.loc[:,use_col])[:,1]
    # pred1 = [1 if i>0.5 else 0 for i in pred1]
    pred2 = r2.predict_proba(processed_test.loc[:,use_col])[:,1]
    # pred2 = [1 if i>0.5 else 0 for i in pred2]
    pred3 = r3.predict_proba(processed_test.loc[:,use_col])[:,1]
    # pred3 = [1 if i>0.5 else 0 for i in pred3]
    pred4 = r4.predict_proba(processed_test.loc[:,use_col])[:,1]
    # pred4 = [1 if i>0.5 else 0 for i in pred4]


    label = []

    for p1,p2,p3,p4 in zip(pred1,pred2,pred3,pred4):
        if p1+p2+p3+p4>2:
            label.append(1)
        else:
            label.append(0)
    
    processed_test['high_price_flag'] = label

    processed_train.to_csv(args.output_train,index=False)
    processed_test.to_csv(args.output_test,index=False)


    