
from config import model_name
import pandas as pd
from tqdm import tqdm
import os
import random
import numpy as np
import csv
import importlib
import copy
from collections import Counter
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
ratio = config.negative_sampling_ratio
lifetime = config.lifetime
preprocess_data_folder = './data/preprocessed_data'
preprocess_type = '(type1)'         # '(type1)'/ '(type2)' / ''
data_list = ['Adressa_5w',
# 'Adressa_6w','Adressa_7w'
]
data_list = [s+preprocess_type for s in data_list]

for data_type in data_list:
    print(f'\n{data_type} / {model_name} \nmake_behaviors_parsed')
    news_dir = f'./data/preprocessed_data/{data_type}'
    train_dir = f'./data/preprocessed_data/{data_type}/train'
    # behaviors = pd.read_table(
    #     os.path.join(train_dir,f'behaviors_parsed_ns{ratio}_lt{lifetime}.tsv'))
    # behaviors['click'] = behaviors['candidate_news_impre'].str.split(' ')[0]
    
    behaviors = pd.read_table(
        os.path.join(train_dir,'behaviors.tsv'),
        header=None,
        names=['user', 'time', 'history', 'click'])
    behaviors.click = behaviors.click.str.split('-').str[0]
    behaviors.history.fillna(' ', inplace=True)
    
    ##
    behaviors['time'] = pd.to_datetime(behaviors['time'])
    ##
    added_columns_list = []

    news = pd.read_table(os.path.join(news_dir, 'total_news(raw).tsv'),
                        quoting=csv.QUOTE_NONE,
                        header=None,
                        names=['id','category', 'subcategory','title','body','raw_id','publish_time','clicks'])  # TODO try to avoid csv.QUOTE_NONE

    news['publish_time'] = pd.to_datetime(news['publish_time'])
    news = news.set_index(['id'])
    news.fillna(' ', inplace=True)
    news['train_click'] = 0
    news['his_click'] = 0
    news['lifetime_click'] = 0
    news['current_lifetime_click'] = 0

    train_click_cnt = behaviors['click'].value_counts()
    history_click = behaviors.groupby(by='user')['history'].last()
    history_click_list = history_click.str.cat(sep=' ').split()
    history_click_cnt = dict(Counter(history_click_list))
    for i,row in tqdm(news.iterrows(),desc="News training click count"):
        try:
            news.at[i,'train_click'] = train_click_cnt[i]
        except:
            pass

        try:
            news.at[i,'his_click'] = history_click_cnt[i]
        except:
            pass
        end_time = row.publish_time + timedelta(hours=lifetime)
        behavior_mask = (behaviors['time']<= end_time) & (behaviors['time'] >= row.publish_time)
        behaviors_tmp = behaviors.loc[behavior_mask]
        click_cnt = dict(Counter(behaviors_tmp['click'].str.cat(sep=' ').split()))
        try:
            news.at[i,'lifetime_click'] = history_click_cnt[i]
        except:
            pass
    news['all_click'] = news['train_click'] + news['his_click']
        
    news = news.sort_values(by=['lifetime_click'], ascending=False)

    user_poslist = behaviors.groupby(by='user')['click'].apply(lambda x: ','.join(x))
    # negative_candidate_len = []


    ### (lifetime pop)current_log_pop weight + rev_current_log_pop 

    new_behaviors = copy.copy(behaviors)

    new_column_name = 'candidate_news_current_log_pop'
    new_column_name2 = 'candidate_news_rev_current_log_pop'
    for row in tqdm(new_behaviors.itertuples(), desc=f"Generate {new_column_name} and {new_column_name2}"):
        endtime = row.time - timedelta(hours=lifetime)
        poslist = user_poslist.loc[row.user].split(',')

        mask = (news['publish_time'] <= row.time) & (news['publish_time'] >= endtime)
        news_negative_candidate = copy.copy(news.loc[mask])
        behavior_mask = (behaviors['time']<= row.time) & (behaviors['time'] >= endtime)
        behaviors_tmp = behaviors.loc[behavior_mask]
        click_cnt = dict(Counter(behaviors_tmp['click'].str.cat(sep=' ').split()))
        
        news_negative_candidate['current_lifetime_click'] = news_negative_candidate.index.map(click_cnt)
        news_negative_candidate.dropna(inplace=True)
        news_negative_candidate.dropna(subset='current_lifetime_click',inplace=True)

        news_negative_list = news_negative_candidate.index.to_list()
        news_weight = news_negative_candidate.current_lifetime_click.to_list()
        news_weight = [np.log(x+2) for x in news_weight]

        for del_news in set(news_negative_list).intersection(poslist):
            del_index = news_negative_list.index(del_news)

            del news_negative_list[del_index]
            del news_weight[del_index]

        candidate_news = [row.click]

        if ratio > len(news_negative_list):
            times = (ratio//len(news_negative_list)+1)
            news_negative_list = news_negative_list*times

            news_weight = news_weight*times

        news_negative_list2 = copy.copy(news_negative_list)
        news_weight2 = [1/x for x in news_weight]
        candidate_news2 = copy.copy(candidate_news)

        for i in range(ratio):
            ns = random.choices(news_negative_list,k=1,weights = news_weight)

            candidate_news.append(ns[0])
            del_index = news_negative_list.index(ns[0])

            del news_negative_list[del_index]
            del news_weight[del_index]

        for i in range(ratio):
            ns = random.choices(news_negative_list2,k=1,weights = news_weight2)

            candidate_news2.append(ns[0])
            del_index = news_negative_list2.index(ns[0])

            del news_negative_list2[del_index]
            del news_weight2[del_index]
        
        new_behaviors.at[row.Index,new_column_name] = ' '.join(candidate_news)
        new_behaviors.at[row.Index,new_column_name2] = ' '.join(candidate_news2)
        # tmp_cnt += 1
        # if tmp_cnt == 1000:
        #     break
    added_columns_list.append(new_column_name.replace('candidate_news_','',1))
    added_columns_list.append(new_column_name2.replace('candidate_news_','',1))

   
    for row in tqdm(behaviors.itertuples(), desc="Generate impression candidate"):
        endtime = row.time - timedelta(hours=lifetime)
        poslist = user_poslist.loc[row.user].split(',')

        ### impression
        mask = (news['publish_time'] <= row.time)
        news_negative_candidate = news.loc[mask]
        news_negative_list = news_negative_candidate.index.to_list()

        for del_news in set(news_negative_list).intersection(poslist):
            del_index = news_negative_list.index(del_news)

            del news_negative_list[del_index]

        candidate_news_impre = [row.click]

        if ratio > len(news_negative_list):
            times = (ratio//len(news_negative_list)+1)
            news_negative_list = news_negative_list*times

        candidate_news_impre += random.sample(news_negative_list,ratio)
        behaviors.at[row.Index,'candidate_news_impre'] = ' '.join(candidate_news_impre)
    added_columns_list.append('impre')

    
    new_behaviors['clicked'] = ' '.join(['1']+['0']*ratio)
    del new_behaviors['click']
    try:
        new_behaviors.rename(columns={'history': 'clicked_news'}, inplace=True)
    except:
        pass
    new_behaviors.to_csv(
            os.path.join(train_dir,f'behaviors_parsed_ns{ratio}_lt{lifetime}.tsv'),
            sep='\t',
            index=False)
    # print(f'\nTry {data_type} / add_behaviors_parsed')
    # try:
    #     behaviors_origin = pd.read_table(
    #         os.path.join(train_dir,f'behaviors_parsed_ns{ratio}_lt{lifetime}.tsv'))
    #     behaviors.to_csv(
    #         os.path.join(train_dir,f'behaviors_parsed_ns{ratio}_lt{lifetime}_tmp.tsv'),
    #         sep='\t',
    #         index=False)
    #     behaviors2 = pd.read_table(
    #         os.path.join(train_dir,f'behaviors_parsed_ns{ratio}_lt{lifetime}_tmp.tsv'))
    # 
    #     for col in list(behaviors2.columns):
    #         if col.startswith('candidate_news'):
    #             behaviors_origin[col] = behaviors2[col]
    # 
    #     behaviors_origin.to_csv(
    #         os.path.join(train_dir,f'behaviors_parsed_ns{ratio}_lt{lifetime}.tsv'),
    #         sep='\t',
    #         index=False)
    #     print("Add to Origin File Done")
    # except:
    #     behaviors.to_csv(
    #             os.path.join(train_dir,f'behaviors_parsed_ns{ratio}_lt{lifetime}.tsv'),
    #             sep='\t',
    #             index=False)
    #     print("No Origin File/ Made Origin File")

    # 
    # with open('train_batch.bat','w') as wf:
    #     for candidate in added_columns_list:
    #         wf.writelines(f'python write_new_config.py {candidate} CEL {model_name} {ratio} {lifetime} {data_type} {config.numbering}\n')
    #         wf.writelines(f'python train.py\n')


