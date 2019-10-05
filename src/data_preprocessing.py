import sys
import os
import glob
import re

import pandas as pd
import pandas_profiling as pdp
import numpy as np
import argparse
from tqdm import tqdm_notebook as tqdm
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
                 'walk_time','23ku',
                #  'area_num_countall','floor_countall','room_num_countall','facilities_countall','age_countall','area_num_countall',
                ]

    use_col = [c for c in use_col if c not in un_use_col]

    # print(use_col)

    # high and low classifier
    clf = High_and_Low_Classifier(processed_train,use_col)
    clf.train()
    clf.pred(processed_test)
    processed_test['prob'],processed_test['label'] = clf.labeling()

    processed_train.to_csv(args.output_train,index=False)
    processed_test.to_csv(args.output_test,index=False)


    