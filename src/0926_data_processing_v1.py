import sys
import os
import glob
import re

import pandas as pd
import pandas_profiling as pdp
import numpy as np
import argparse
from tqdm import tqdm_notebook as tqdm

def processing_data(df,mode):

    if mode=='train':
        df.columns = ['id','y','location','access','layout','age','direction','area','floor',
            'bath_toilet','kitchen','broadcast_com','facilities','parking','enviroment',
            'structure','contact_period']

        df_tar = df.loc[:,['id','age','area','floor','y']]

    else:
        df.columns = ['id','location','access','layout','age','direction','area','floor',
            'bath_toilet','kitchen','broadcast_com','facilities','parking','enviroment',
            'structure','contact_period']

        df_tar = df.loc[:,['id','age','area','floor']]
    

    # area
    df_tar['area'] = df_tar['area'].apply(del_m2)

    # age
    df_tar['age'] = df_tar['age'].apply(year_to_month)

    # floor
    df_tar['max_floor'] = max_floor_col(df_tar['floor'])
    df_tar['floor'] = df_tar['floor'].apply(extract_floor)

    if mode=='train':
        df_tar = df_tar.loc[:,['id','age','area','floor','max_floor','y']]

    else:
        df_tar = df_tar.loc[:,['id','age','area','floor','max_floor']]
        

    return df_tar


def del_m2(a):
    return float(a[:-2])

def year_to_month(age):
    y,m = 0,0
    for i,s in enumerate(age):
        if s=="年":
            y = age[:i]
            tmp = i+1
        if s=="ヶ":
            m = age[tmp:i] 
    
    return int(y)*12 + int(m) 

def extract_floor(f):
    try:
        floor = re.search(r'\d+階／', f)
        
        try:
            floor = int(floor.group()[:-2])
        
        except:
            '''
            最上階が書かれていない場合の表記ゆれに対応
            '''
            floor = re.search(r'\d+階', f)
            floor = int(floor.group()[:-1])

    except:
        '''
        欠損の場合1階とする
        '''
        floor = 1
        
    return floor

def max_floor_col(floor_col):
    res = []
    
    for f in floor_col:

        try:
            m = re.search(r'\d+階建', f)
            try:
                res.append(int(m.group()[:-2]))
                
            except:
                '''
                最上階についての記述がない場合は所在階を最上階と断定
                '''
                m = re.search(r'\d+階', f)
                res.append(int(m.group()[:-1]))

        except:
            '''
            欠損の場合1階とする
            '''
            res.append(1)

    return res

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

    