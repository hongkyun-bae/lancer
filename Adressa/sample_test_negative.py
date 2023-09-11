
from config import model_name
import pandas as pd
from tqdm import tqdm
import os
import random
import numpy as np
import csv
import importlib
import copy
from datetime import timedelta
try:
    config = getattr(importlib.import_module('config'), f"{model_name}Config")
except AttributeError:
    print(f"{model_name} not included!")
    exit()

import warnings
warnings.filterwarnings('ignore')
from datetime import datetime

def str_to_timestamp(string):
    return datetime.timestamp(datetime.strptime(string,'%Y-%m-%d %H:%M:%S'))

model_name = 'NRMS'
preprocess_data_folder = './data/preprocessed_data'


print(os.listdir(preprocess_data_folder))


ratio = 20
lifetime = 36
data_list = ['Adressa_5w(type1)'
]
for data_type in os.listdir(preprocess_data_folder):
    if data_type in data_list:
        print()
        print(data_type)
        news_dir = os.path.join(preprocess_data_folder,data_type)
        test_dir = os.path.join(news_dir,'test')


        news = pd.read_table(os.path.join(news_dir, 'total_news(raw).tsv'),
                            quoting=csv.QUOTE_NONE,
                            header=None,
                            names=['id','category', 'subcategory','title','body','raw_id','publish_time','clicks'])  # TODO try to avoid csv.QUOTE_NONE
        news['publish_time'] = pd.to_datetime(news['publish_time'])
        news = news.set_index(['id'])
        news.fillna(' ', inplace=True)
        behaviors = pd.read_table(
            os.path.join(test_dir,'behaviors.tsv'),
            header=None,
            names=['user', 'time', 'history', 'click'])
        behaviors.history.fillna(' ', inplace=True)
        behaviors.click = behaviors.click.str.split('-').str[0]
        ##
        behaviors['time'] = pd.to_datetime(behaviors['time'])

        user_poslist = behaviors.groupby(by='user')['click'].apply(lambda x: ','.join(x))

        del_row_index = []
        dead_test_click_cnt = 0
        last_cnt = len(behaviors)
        for row in tqdm(behaviors.itertuples(), desc=f"{data_type} / Generate test impression candidate(lifetime)"):
            endtime = row.time - timedelta(hours=lifetime)
            if news.loc[row.click].publish_time <= endtime:
                dead_test_click_cnt += 1 
                del_row_index.append(row.Index)
                continue
            poslist = user_poslist.loc[row.user].split(',')
            mask2 = (news['publish_time'] <= row.time) & (news['publish_time'] >= endtime)
            news_negative_candidate = news.loc[mask2]
            news_negative_list = news_negative_candidate.index.to_list()

            # news_negative_list = news.index.to_list()

            for del_news in set(news_negative_list).intersection(poslist):
                del_index = news_negative_list.index(del_news)

                del news_negative_list[del_index]

            news_negative_list2 = []
            try:
                news_negative_list2 += random.sample(news_negative_list,ratio)
            except:
                news_negative_list2 += random.sample(news_negative_list,len(news_negative_list))
                if len(news_negative_list) == 0:
                    print(row.click)
                    break
            behaviors.at[row.Index,'click'] = row.click+'-1 '+'-0 '.join(news_negative_list2) +'-0'
        
        behaviors.drop(del_row_index, axis=0, inplace=True)
        behaviors.to_csv(
                os.path.join(test_dir,f'behaviors_{ratio}_ltlive.tsv'),
                sep='\t',
                header=None)
