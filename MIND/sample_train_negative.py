
from collections import Counter
import numpy as np
import copy
import os
import pandas as pd
from tqdm import tqdm
from datetime import timedelta
import random

import importlib
from config import model_name
try:
    config = getattr(importlib.import_module('config'), f"{model_name}Config")
except AttributeError:
    print(f"{model_name} not included!")
    exit()

dataset = config.dataset
train_dir = f'./data/{dataset}/train'
# test_dir = f'./data/{dataset}/test'
negative_sampling_ratio = 200
# lifetime = 36
behaviors = pd.read_table(os.path.join(train_dir,'behaviors.tsv'),header=None)
behaviors.rename(columns={1:'user',2:'time',4:'impression'}, inplace=True)

news_parsed = pd.read_table(os.path.join(train_dir,'news_parsed_pb_time.tsv'))

behaviors['time'] = pd.to_datetime(behaviors['time'],format= "%m/%d/%Y %I:%M:%S %p")

# pb_time_dic ={}
# for i,row in tqdm(behaviors.iterrows()):
#     impression_list = row.impression.replace('-1','').replace('-0','').split()
#     for news in impression_list:
#         if news in pb_time_dic.keys():
#             if pb_time_dic[news] > row.time:
#                 pb_time_dic[news] = row.time
#         else:
#             pb_time_dic[news] = row.time
# news_parsed['pb_time'] = news_parsed.id.map(pb_time_dic)
news_parsed.dropna(inplace=True)
news_parsed['pb_time'] = pd.to_datetime(news_parsed['pb_time'],format= "%Y-%m-%d %H:%M:%S")
print('negative_news_pool:',len(news_parsed))

news_parsed.set_index('id',drop=True,inplace=True)

behaviors_parsed = pd.read_table(os.path.join(train_dir,'behaviors_parsed.tsv'))
behaviors_parsed['time'] = pd.to_datetime(behaviors_parsed['time'],format= "%m/%d/%Y %I:%M:%S %p")

behaviors_parsed['click'] = behaviors_parsed['candidate_news'].str.split().str[0]

user_poslist = behaviors_parsed.groupby(by='user')['click'].apply(lambda x: ','.join(x))

new_column_name = 'candidate_news_rev_current_log_ltpop'
for row in tqdm(behaviors_parsed.itertuples(),total=len(behaviors_parsed), desc=f"Generate {new_column_name}"):
    endtime = row.time - timedelta(hours=config.lifetime)
    try:
        poslist = user_poslist.loc[row.user].split(',')
    except:
        poslist = []

    mask = (news_parsed['pb_time'] <= row.time) & (news_parsed['pb_time'] >= endtime)
    news_negative_candidate = copy.copy(news_parsed.loc[mask])
    behavior_mask = (behaviors_parsed['time']<= row.time) & (behaviors_parsed['time'] >= endtime)
    behaviors_tmp = behaviors_parsed.loc[behavior_mask]
    click_cnt = dict(Counter(behaviors_tmp['click'].str.cat(sep=' ').split()))
    
    news_negative_candidate['current_lifetime_click'] = news_negative_candidate.index.map(click_cnt)
    news_negative_candidate.fillna(0,inplace=True)

    # news_negative_candidate.dropna(subset='current_lifetime_click',inplace=True)

    news_negative_list = news_negative_candidate.index.to_list()
    news_weight = news_negative_candidate.current_lifetime_click.to_list()
    news_weight = [1/np.log(x+2) for x in news_weight]

    for del_news in set(news_negative_list).intersection(poslist):
        del_index = news_negative_list.index(del_news)

        del news_negative_list[del_index]
        del news_weight[del_index]

    candidate_news = [row.click]

    try:
        if negative_sampling_ratio > len(news_negative_list):
            times = (negative_sampling_ratio//len(news_negative_list)+1)
            news_negative_list = news_negative_list*times

            news_weight = news_weight*times
    except:
        continue
    for i in range(negative_sampling_ratio):
        ns = random.choices(news_negative_list,k=1,weights = news_weight)

        candidate_news.append(ns[0])
        del_index = news_negative_list.index(ns[0])

        del news_negative_list[del_index]
        del news_weight[del_index]
    
    behaviors_parsed.at[row.Index,new_column_name] = ' '.join(candidate_news)


for i,row in tqdm(behaviors_parsed.iterrows(),total=len(behaviors_parsed),desc = 'make candidate pb'):
    
    pb_time = news_parsed.loc[row.click].pb_time
    end_time = pb_time - timedelta(hours=config.lifetime)
    # mask = (news_parsed['pb_time'] <= pb_time)
    mask = (news_parsed['pb_time'] <= pb_time) & (news_parsed['pb_time'] >= end_time)
    news_negative_candidate = news_parsed.loc[mask]
    news_negative_list = news_negative_candidate.index.to_list()
    
    # news_negative_list = news_parsed.index.to_list()

    news_negative_list = list(set(news_negative_list)-set(user_poslist.loc[row.user].split(',')))
    if len(news_negative_list) == 0:
        continue
    elif negative_sampling_ratio > len(news_negative_list):
        times = (negative_sampling_ratio//len(news_negative_list)+1)
        news_negative_list = news_negative_list*times
    candidate_news_pb = [row.click]
    candidate_news_pb += random.sample(news_negative_list,negative_sampling_ratio)
    behaviors_parsed.at[i,'candidate_news_pb'] = ' '.join(candidate_news_pb)

del behaviors_parsed['click']

behaviors_parsed.to_csv(
        os.path.join(train_dir,f'behaviors_parsed.tsv'),
        sep='\t',
        index=False,
        columns=['user', 'clicked_news', 'candidate_news', 'clicked','candidate_news_pb','candidate_news_rev_current_log_ltpop'])