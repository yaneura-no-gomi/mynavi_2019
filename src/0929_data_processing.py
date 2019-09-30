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



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="data processing")

    parser.add_argument(
        '--mode',
        '-m',
        help ='choose train or test for the input csv file',
        type=str,
        required=True
    )

    parser.add_argument(
        '--input_file',
        '-input',
        help ='csv path for processing',
        type=str,
        required=True
    )

    parser.add_argument(
        '--output_file',
        '-output',
        help ='csv path for processing',
        type=str,
        required=True
    )

    args = parser.parse_args()

    in_f = args.input_file
    out_f = args.output_file

    df = pd.read_csv(in_f)

    res = processing_data(df,args.mode)

    res.to_csv(out_f,index=False)

    