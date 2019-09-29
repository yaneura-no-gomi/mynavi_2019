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


    else:
        df.columns = ['id','location','access','layout','age','direction','area','floor',
            'bath_toilet','kitchen','broadcast_com','facilities','parking','enviroment',
            'structure','contact_period']

    

    # area
    df['area'] = df['area'].apply(del_m2)

    # age
    df['age'] = df['age'].apply(year_to_month)

    # floor
    df['max_floor'] = max_floor_col(df['floor'])
    df['floor'] = df['floor'].apply(extract_floor)

    # layout
    room_num,l,d,k,s = layout_split(df['layout'])
    df['room_num'] = room_num
    df['L'] = pd.Series(l)
    df['D'] = pd.Series(d)
    df['K'] = pd.Series(k)
    df['S'] = pd.Series(s)

    # direction
    df['north'] = pd.Series(north_flag(df['direction']))

    # structure
    df['structure'] = structure_order(df['structure'])

    # access
    df['walk_time'],df['min_time'],df['avg_time'] = processing_walk_time(df['access'])

    return df


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

def layout_split(layout):
    room_num = []
    l,d,k,s = [],[],[],[]
    
    for v in layout:
        room_num.append(int(v[0]))
        
        for x, r_type in zip([l,d,k,s],['L','D','K','S']):
            
            if r_type in v:
                x.append(1)
                
            else :
                x.append(0)
        
    return room_num,l,d,k,s

def structure_order(structure):
    res = structure.map({
        'SRC（鉄骨鉄筋コンクリート）': 8,
        'RC（鉄筋コンクリート）': 7,
        'ALC（軽量気泡コンクリート）': 6,
        'PC（プレキャスト・コンクリート（鉄筋コンクリート））': 5,
        '軽量鉄骨': 4,
        'HPC（プレキャスト・コンクリート（重量鉄骨））': 3,
        '鉄骨造':  2,
        'その他': 1,
        '木造': 1,
        '鉄筋ブロック':0,
        'ブロック': 0
    })

    return res
    
def north_flag(direction):
    res = []
    
    direction = direction.fillna('')
    
    for d in direction:
        if  '北' in d:
            res.append(1)
        
        else:
            res.append(0)
            
    return res

def processing_walk_time(access):
    walk_time = []
    min_time = []
    avg_time = []
    for a in access:
        if '徒歩' in a:
            tmp_l = []

            for t in re.findall(r'徒歩\d+分',a):
                tmp = int(re.search(r'\d+',t).group())
                tmp_l.append(tmp)

        walk_time.append(tmp_l)
        min_time.append(np.array(tmp_l).min())
        avg_time.append(np.array(tmp_l).mean())
    
    return walk_time, min_time, avg_time

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

    