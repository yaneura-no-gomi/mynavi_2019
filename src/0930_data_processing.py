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

from prepro_func.prepro_func import preprocessing_train_test
# from prepro_func.tmp import preprocessing_train_test


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

    processed_train.to_csv(args.output_train,index=False)
    processed_test.to_csv(args.output_test,index=False)


    