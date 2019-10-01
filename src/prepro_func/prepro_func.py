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

def preprocessing_train_test():

    train = pd.read_csv('~/Workspace/learning/signate/mynavi_2019/data/train.csv')
    test = pd.read_csv('~/Workspace/learning/signate/mynavi_2019/data/test.csv')

    
    train.columns = ['id','y','location','access','layout','age','direction','area','floor',
        'bath_toilet','kitchen','broadcast_com','facilities','parking','enviroment',
        'structure','contact_period']

    train['log_y'] = np.log(train['y'])


    test.columns = ['id','location','access','layout','age','direction','area','floor',
        'bath_toilet','kitchen','broadcast_com','facilities','parking','enviroment',
        'structure','contact_period']

    
    n_df = []

    for df in [train,test]:
        # remove outlier row
        df = remove_outlier(df)

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

        # parking 
        parking = df['parking']
        df['bicycle_parking'] = processing_bicycle_parking(parking)
        df['car_parking'] = processing_car_parking(parking)
        df['bike_parking'] = processing_bike_parking(parking)

        # bath_toilet
        bt = df['bath_toilet']
        df['toilet'] = processing_toilet(bt)
        df['bath'] = processing_bath(bt)
        df['sm_doku'],df['kanso'],df['onsui'],df['oidaki'],df['b_t_split'] = bath_toilet_option(bt)        
 
        # location
        df['23ku'],ku_mean_std = preprocessing_location(df['location'])
        n_df.append(pd.merge(df,ku_mean_std,on='23ku').sort_values(by='id'))


    train,test = n_df[0],n_df[1]

    # count encoding for direction
    train, test = makeCountFull(train, test, ['direction'])

    return train, test


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

def preprocessing_location(location):
    '''
    あらかじめtrainから求めておいた区ごとの家賃平均を標準化したものを特徴量として追加
    '''

    ku23 = ['千代田区','中央区','港区','新宿区','文京区','台東区','墨田区','江東区','品川区','目黒区',
        '大田区','世田谷区','渋谷区','中野区','杉並区','豊島区','北区','荒川区','板橋区','練馬区',
        '足立区','葛飾区','江戸川区']

    l = []
    for loc in location:
        for ku in ku23:
            if ku in loc:
                l.append(ku)

    # df['23ku'] = pd.Series(l)

    ku_mean_std = pd.DataFrame({
        '23ku':['葛飾区', '足立区', '江戸川区', '板橋区', '練馬区', '杉並区', '北区', '中野区', '大田区', '豊島区',
            '世田谷区', '墨田区', '荒川区', '品川区', '江東区', '台東区', '文京区', '新宿区', '目黒区', '渋谷区',
            '中央区', '千代田区', '港区'],
        
        '23ku_mean_std':[-1.2807574098213284, -1.0838383537489147, -1.0628650998041507, -0.963454612411768,
            -0.8841650819596225,-0.7389579005058835,-0.6865532096858878, -0.6261828349388672, -0.468735870806376, -0.35406716765268814,
            -0.2706013619101045,-0.20008236695852455,-0.1821170416794129,0.044923401400534836,0.09176861038849397,0.1328208371225872,
            0.16009877672317527,0.2187234739900172,0.6587373980981039,1.4211751659457696,1.684828598882115,1.7378144005920697,2.651487648740651]

    })

    return l, ku_mean_std

def makeCountFull(train, test, categorical_features=None, report=False):
    add_cols = categorical_features
    if report:
        print('add_cols: ', add_cols)
    for add_col in add_cols:
        train[add_col + '_countall'] = train[add_col].map(pd.concat([train[add_col], test[add_col]], ignore_index=True).value_counts(dropna=False))
        test[add_col + '_countall'] = test[add_col].map(pd.concat([train[add_col], test[add_col]], ignore_index=True).value_counts(dropna=False))
    return train, test

def processing_bicycle_parking(parking):
    '''
    駐輪場
    無・空無 -> 0
    有かつ無料 -> 1
    有かつ有料 -> 2
    '''
    bicycle = []
    for e in parking.fillna(''):
        split_e = e.split('\t')
        if '駐輪場' in e:
            try:
                state = split_e[split_e.index('駐輪場')+1]
                if '無' in state:
                    bicycle.append(0)

                elif '空有' in state or '近隣' in state:
                    target = split_e[split_e.index('駐輪場')+2]
                    try:
                        '''
                        値段の記載がある場合
                        '''
                        price = re.search(r'\d*,*\d+円',target).group()
                        price =  int(price[:-1].replace(',',''))

                        if price==0:
                            bicycle.append(1)

                        else:
                            bicycle.append(2)

                    except:
                        try:
                            free = re.search(r'無料',target).group()
                            bicycle.append(1)

                        except:
                            try:
                                not_free = re.search(r'有料',target).group()
                                bicycle.append(2)

                            except:
                                bicycle.append(1)

            except:
                # '詳細をお問い合わせください'などの余計な文字列が入っている場合
                # この場合は駐輪場が有る場合なのでstate=1とする
                bicycle.append(1)

        else:
            bicycle.append(0)
    
    return bicycle

def processing_car_parking(parking):
    '''
    駐車場
    無・空無 -> 0
    有かつ無料 -> 1
    有かつ有料 -> 2
    '''
    car = []
    for e in parking.fillna(''):
        split_e = e.split('\t')
        if '駐車場' in e:
            state = split_e[split_e.index('駐車場')+1]

            if '無' in state:
                car.append(0)
            elif '空有' in state or '近隣' in state:
                try:
                    target = split_e[split_e.index('駐車場')+2]
                    '''
                    値段の記載がある場合
                    '''
                    price = re.search(r'\d*,*\d+円',target).group()
                    price =  int(price[:-1].replace(',',''))

                    if price==0:
                        car.append(1)

                    else:
                        car.append(2)
                except:
                    '''
                    index('駐車場')+2が存在しない場合(stateの記述で終わっている場合)
                    '''
                    car.append(1)

        else:
            car.append(0)
    
    return car

def processing_bike_parking(parking):
    '''
    バイク置き場
    無・空無 -> 0
    有かつ無料 -> 1
    有かつ有料 -> 2
    '''
    bike = []
    for e in parking.fillna(''):
        split_e = e.split('\t')
        if 'バイク置き場' in e:
            try:
                state = split_e[split_e.index('バイク置き場')+1]
                hoge.append(state)

                if '無' in state:
                    bike.append(0)

                elif '空有' in state or '近隣' in state:
                    try:
                        target = split_e[split_e.index('バイク置き場')+2]
                        '''
                        値段の記載がある場合
                        '''
                        price = re.search(r'\d*,*\d+円', target).group()
                        price = int(price[:-1].replace(',', ''))

                        if price == 0:
                            bike.append(1)

                        else:
                            bike.append(2)

                    except:
                        bike.append(1)
            except:
                '''
                 '(大型バイク置き場有)', '(バイク置き場有)'
                 よって1とする
                '''
                bike.append(1)
        else:
            bike.append(0)
            
    return bike

def remove_outlier(df):
    remove_ids = [20927,20232,20428]
    
    for _id in remove_ids:
        res = df.drop(df[df['id']==_id].index)
    
    return res

def processing_toilet(bt):
    '''
    トイレ
    なし：０
    共同：１
    専用：２
    '''
    toilet = []
    for t in bt.fillna(''):
        if 'トイレ' in t or '温水洗浄便座' in t:
            if '共同トイレ' in t:
                toilet.append(1)
            elif '専用トイレ' in t:
                toilet.append(2)
            else:
                toilet.append(2)
        else:
            if t=='':
                toilet.append(2)
            else:
                toilet.append(0)
    
    return toilet

def processing_bath(bt):
    '''
    バス
    なし：０
    シャワー：１
    共同：２
    専用：３
    '''
    bath = []
    for b in bt.fillna(''):
        if 'バス' in b or '浴室乾燥' in b or '追焚機能' in b:
            if '共同バス' in b:
                bath.append(2)
            elif '専用バス' in b:
                bath.append(3)
            else:
                bath.append(3)
        else:
            if 'シャワー' in b or '脱衣所' in b:
                bath.append(1)
            elif b=='':
                bath.append(3)
            else:
                bath.append(0)
    
    return bath

def bath_toilet_option(bt):
    '''
    洗面台独立、浴室乾燥機、追焚機能、温水洗浄便座、バス・トイレ別があるところにフラグ
    '''
    sm_doku = []
    kanso = []
    oidaki = []
    onsui = []
    b_t_split = []

    for b in bt.fillna(''):
        if '洗面台独立' in b:
            sm_doku.append(1)
        else:
            sm_doku.append(0)

        if '浴室乾燥機' in b:
            kanso.append(1)
        else:
            kanso.append(0)

        if '温水洗浄便座' in b:
            onsui.append(1)
        else:
            onsui.append(0)

        if '追焚機能' in b:
            oidaki.append(1)
        else:
            oidaki.append(0)

        if 'バス・トイレ別' in b:
            b_t_split.append(1)
        else:
            b_t_split.append(0)

    return sm_doku, kanso, onsui, oidaki, b_t_split