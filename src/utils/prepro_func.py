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
        'structure','contract_period']

    train['log_y'] = np.log(train['y'])


    test.columns = ['id','location','access','layout','age','direction','area','floor',
        'bath_toilet','kitchen','broadcast_com','facilities','parking','enviroment',
        'structure','contract_period']

    
    n_df = []

    for df in [train,test]:
        # remove outlier row
        df = remove_outlier(df)

        # area
        df['area_num'] = df['area'].apply(del_m2)

        # age
        df['age_month'] = df['age'].apply(year_to_month)

        # floor
        df['max_floor'] = max_floor_col(df['floor'])
        df['floor'] = df['floor'].apply(extract_floor)

        # layout
        room_num,l,d,k,r,s = layout_split(df['layout'])
        df['room_num'] = room_num
        df['L'] = l
        df['D'] = d
        df['K'] = k
        df['R'] = r
        df['S'] = s

        # direction
        df['north'] = north_flag(df['direction'])

        # structure
        df['structure_orderd'] = structure_order(df['structure'])

        # access
        df['walk_time'],df['min_time'],df['avg_time'] = processing_walk_time(df['access'])

        # parking 
        parking = df['parking']
        df['bicycle_parking'] = processing_bicycle_parking(parking)
        df['car_parking'] = processing_car_parking(parking)
        df['bike_parking'] = processing_bike_parking(parking)
        # df['parking'] = df['parking'].fillna('駐車場\t無')

        # bath_toilet
        bt = df['bath_toilet']
        df['toilet'] = processing_toilet(bt)
        df['bath'] = processing_bath(bt)
        df['sm_doku'],df['kanso'],df['onsui'],df['oidaki'],df['b_t_split'] = bath_toilet_option(bt)

        # contract_period
        df['teiki_syakuya'] = teiki_syakuya(df['contract_period'])
        df['contract_period'] = df['contract_period'].fillna('2年間')
 
        # enviroment
        df['e_num'],df['dis_ave'],df['dis_min'] = processing_env(df['enviroment'])
        df['school'],df['univ'] = processing_school(df['enviroment'])

        # area_par_room
        df['area_par_room'] = area_par_room(df)

        # location
        df['23ku'],ku_mean_std = preprocessing_location(df['location'])
        n_df.append(pd.merge(df,ku_mean_std,on='23ku').sort_values(by='id'))


    train,test = n_df[0],n_df[1]

    # high price labeling
    train['high_price_flag'] = high_price_flag(train,400000)

    # facilities
    train, test = processing_facilities(train,test)
    # broadcast_com
    train, test = processing_broadcast_com(train,test)
    # kitchen
    train, test = processing_kitchen(train,test)

    # count encoding for direction
    train, test = makeCountFull(train, test, ['23ku','area_num','age','floor','max_floor','layout',
            'direction','facilities','contract_period'])

    # too large y is supposed to outliers
    train = train[train['y'] < 2000000]

    train_idokdo = pd.read_csv('processed_data/place_train.csv')
    train = pd.merge(train,train_idokdo,on='id')

    test_idokdo = pd.read_csv('processed_data/place_test.csv')
    test = pd.merge(test,test_idokdo,on='id')

    km2 = pd.read_csv('processed_data/23ku_mean_2.csv')

    train = pd.merge(train,km2,on=['23ku','room_num']).sort_values('id').reset_index(drop=True)
    test = pd.merge(test,km2,on=['23ku','room_num']).sort_values('id').reset_index(drop=True)
    
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

def high_price_flag(train,th_h):
    def labeling(row):
        if row['y']<th_h:
            return 0
        else:
            return 1
    
    return train.apply(labeling,axis=1)

def layout_split(layout):
    room_num = []
    l,d,k,r,s = [],[],[],[],[]
    
    # layout = layout.fillna('')
    for v in layout:
        room_n = re.search(r'\d+',v).group()
        room_num.append(int(room_n))

        for x, r_type in zip([l,d,k,r,s],['L','D','K','R','S']):
            
            if r_type in v:
                x.append(1)
                
            else :
                x.append(0)
        
    return room_num,l,d,k,r,s

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
    times = []
    min_time = []
    ave_time = []

    for a in access:
        ways = a.split('\t\t')
        time = []
        for way in ways:
            hun = way.count('分')
            if hun == 0:
                continue

            elif hun == 1:
                t = re.search(r'\d+分', way).group()[:-1]
                time.append(int(t))

            elif hun == 2:
                if 'バス' not in way:
                    # 9行のみ
#                     print(way)
                    tmp = re.findall(r'徒歩\d+分', way)
                    l = []
                    for s in tmp:
                        t = int(s[2:-1])
                        l.append(int(t))
                    time.append(min(l))

                else:
                    try:
                        b_t = re.search(r'バス\(\d+分\)', way).group()
                    except:
                        b_t = re.search(r'バス\d+分', way).group()
                    b_t = int(re.search(r'\d+', b_t).group())
                    w_t = re.search(r'徒歩\d+分', way).group()
                    w_t = int(re.search(r'\d+', w_t).group())
                    time.append(b_t*5+w_t)

            elif hun == 3:
                # 3行のみ
                if 'バス' in way:
                    t = re.search(r'下車徒歩\d+分', way).group()
                    t = re.search(r'\d+', t).group()
                    time.append(int(t))
                else:
                    tmp = re.findall(r'徒歩\d+分', way)
                    tmp = [int(re.search(r'\d+', t).group()) for t in tmp]
                    time += [int(t) for t in tmp]

            
        times.append(time)
        min_time.append(min(time))
        ave_time.append(np.array(time).mean())

    return times,min_time,ave_time

def preprocessing_location(location):
    '''
    あらかじめtrainから求めておいた区ごとの家賃平均を標準化したものを特徴量として追加
    '''

    ku23 = location.str.extract('(.+)都(.*区)(.*)',expand=True)[1]

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

    return ku23, ku_mean_std

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
                        # try:
                            # _f = re.search(r'無料',target).group()
                        if '無料' in target:   
                            bicycle.append(1)

                        else:
                            # try:
                            #     _f = re.search(r'有料',target).group()
                            if '有料' in target:
                                bicycle.append(2)

                            else:
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
    '''
    id = 20232,20428 築年数がおかしい
    id = 20927 面積と家賃のバランスがおかしい
    '''
    remove_ids = [5776,20927,20232,20428]
    
    for _id in remove_ids:
        res = df.drop(df[df['id']==_id].index)
        df = res

    res = df

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

def teiki_syakuya(c_period):
    res = []
    for c in c_period.fillna(''):
        if '定期借家' in c:
            res.append(1)
        else:
            res.append(0)
    return res

def processing_kitchen(train,test):
    kitchen = pd.concat([train['kitchen'],test['kitchen']])
    
    k_uni = []
    for k in kitchen.fillna(''):
        k_split = k.split('\t')
        k_uni += k_split
        
    k_uni = sorted(list(set(k_uni)))
    
    rm_idx = []
    for i,u in enumerate(k_uni):
        if '／' in u:
            rm_idx.append(i)
            
    k_uni = [k_uni[i] for i in range(len(k_uni)) if i not in rm_idx]    
    n_df = []
    for df in [train,test]:
        for i,v in enumerate(k_uni[1:]):
            l = []

            for k in df['kitchen'].fillna(' '):
                if v in k:
                    l.append(1)
                else:
                    l.append(0)
            df['kitchen_'+str(i)] = l
            
        n_df.append(df)
        
    return n_df[0],n_df[1]

def processing_broadcast_com(train,test):
    broadcast_com = pd.concat([train['broadcast_com'],test['broadcast_com']])
    
    b_uni = []
    for b in broadcast_com.fillna(''):
        b_split = b.split('\t')
        b_uni += b_split
        
    b_uni = sorted(list(set(b_uni)))
    
    rm_idx = []
    for i,u in enumerate(b_uni):
        if '／' in u:
            rm_idx.append(i)
            
    b_uni = [b_uni[i] for i in range(len(b_uni)) if i not in rm_idx]
#     print(b_uni)
    
    n_df = []
    for df in [train,test]:
        for i,v in enumerate(b_uni[1:]):
            l = []

            for k in df['broadcast_com'].fillna(' '):
                if v in k:
                    l.append(1)
                else:
                    l.append(0)
            df['bc_'+str(i)] = l
            
        n_df.append(df)
        
    return n_df[0],n_df[1]

def processing_facilities(train,test):
    facilities = pd.concat([train['facilities'],test['facilities']])
    
    f_uni = []
    for f in facilities.fillna(''):
        f_split = f.split('\t')
        f_uni += f_split
        
    f_uni = sorted(list(set(f_uni)))
    
    rm_idx = []
    for i,u in enumerate(f_uni):
        if '／' in u:
            rm_idx.append(i)
            
    f_uni = [f_uni[i] for i in range(len(f_uni)) if i not in rm_idx]
    # print(f_uni)
    
    n_df = []
    for df in [train,test]:
        for i,v in enumerate(f_uni[1:]):
            l = []

            for k in df['facilities'].fillna(' '):
                if v in k:
                    l.append(1)
                else:
                    l.append(0)
            df['facilities_'+str(i)] = l
            
        n_df.append(df)
        
    return n_df[0],n_df[1]

def processing_env(env):
    e_num = []
    dis_ave = []
    dis_min = []

    # nan_idx = env[env.isnull()].index
    # print(nan_idx)
    for e in env.fillna(''):
        if e=='':
            e_num.append(0)
            dis_ave.append(-1)
            dis_min.append(-1)
        else:
            # print(idx)
            e_split = e.split('\t')
            e_num.append(len(e_split))

            distance = []
            for s in e_split:
                distance.append(int(re.search(r'\d+',s).group()))
                
            dis_ave.append(np.array(distance).mean())
            dis_min.append(np.array(distance).min())

    # df['env_num'] = e_num
    # df['dis_ave'] = dis_ave
    # df['dis_min'] = dis_min

    # df[df['dis_ave']==-1].loc[:,'dis_ave'] = np.array(dis_ave).mean()
    # df[df['dis_min']==-1].loc[:,'dis_min'] = np.array(dis_min).mean()

    dis_ave = [i if i!=-1 else np.array(dis_ave).mean() for i in dis_ave]
    dis_min = [i if i!=-1 else np.array(dis_min).mean() for i in dis_min]
    
    return e_num,dis_ave,dis_min

def processing_school(env):
    school = []
    univ = []
    
    for e in env.fillna(''):
        if '大学' in e:
            univ.append(1)
        else:
            univ.append(0)
            
        if '学校' in e or '小学校' in e or  '幼稚園・保育園' in e:
            school.append(1)
        else:
            school.append(0)
    
    return school, univ

def area_par_room(df):
    def calc(row):
        return row['area_num']/row['room_num']
    
    res = df.apply(calc,axis=1)
    
    return pd.Series(res)