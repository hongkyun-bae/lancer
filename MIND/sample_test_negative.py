
import os
from tkinter import X
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
sampling_type = 'impre'
dataset = config.dataset
train_dir = f'./data/{dataset}/train'
test_dir = f'./data/{dataset}/test'
# val_dir = f'./data/{dataset}/dev'
val_dir = f'./data/{dataset}/val'

# lifetime_list = [24]


# for lifetime in lifetime_list:
train_behaviors = pd.read_table(os.path.join(train_dir,'behaviors.tsv'),header=None)
train_behaviors.rename(columns={1:'user', 2:'time',4:'impression'}, inplace=True)
train_behaviors['time'] = pd.to_datetime(train_behaviors['time'],format= "%m/%d/%Y %I:%M:%S %p")

test_behaviors = pd.read_table(os.path.join(test_dir,'behaviors.tsv'),header=None)
test_behaviors.rename(columns={1:'user', 2:'time',4:'impression'}, inplace=True)
test_behaviors['time'] = pd.to_datetime(test_behaviors['time'],format= "%m/%d/%Y %I:%M:%S %p")

val_behaviors = pd.read_table(os.path.join(val_dir,'behaviors.tsv'),header=None)
val_behaviors.rename(columns={1:'user', 2:'time',4:'impression'}, inplace=True)
val_behaviors['time'] = pd.to_datetime(val_behaviors['time'],format= "%m/%d/%Y %I:%M:%S %p")
behaviors = pd.concat([train_behaviors, val_behaviors, test_behaviors])

news_parsed_train = pd.read_table(os.path.join(train_dir,'news_parsed.tsv'))
news_parsed_test = pd.read_table(os.path.join(test_dir,'news_parsed.tsv'))
news_parsed = pd.concat([news_parsed_train,news_parsed_test]).drop_duplicates().reset_index(drop=True)
news_parsed.to_csv(os.path.join(test_dir,'news_parsed_total.tsv'), sep='\t', index=False)
news_parsed.to_csv(os.path.join(val_dir,'news_parsed_total.tsv'), sep='\t', index=False)
news_parsed.to_csv(os.path.join(val_dir,'news_parsed_total.tsv'), sep='\t', index=False)

print('\nTotal News Cnt:',len(news_parsed))
pb_time_dic ={}
for i,row in tqdm(behaviors.iterrows()):
    impression_list = row.impression.replace('-1','').replace('-0','').split()
    for news in impression_list:
        if news in pb_time_dic.keys():
            if pb_time_dic[news] > row.time:
                pb_time_dic[news] = row.time
        else:
            pb_time_dic[news] = row.time
news_parsed.dropna(inplace=True)
news_parsed['pb_time'] = news_parsed.id.map(pb_time_dic)
news_parsed.drop(['category','subcategory','title','abstract','title_entities','abstract_entities'],axis=1,inplace=True)
news_parsed.set_index('id',drop=True,inplace=True)
news_parsed.to_csv(os.path.join(test_dir,'news_parsed_pb_time.tsv'), sep='\t')
news_parsed.to_csv(os.path.join(val_dir,'news_parsed_pb_time.tsv'), sep='\t')


user_pos_df = behaviors.groupby(by='user')['impression'].apply(lambda x: ' '.join(x))
    
lifetime = 36
for i,row in tqdm(test_behaviors.iterrows(),desc='make_test_pb'):
    impression_list = row.impression.split()
    neg_list = [X for x in impression_list if x.endswith('-0')]
    pos_list = [x for x in impression_list if x.endswith('-1')]
    end_time = row.time - timedelta(hours=lifetime)
    # mask = (news_parsed['pb_time'] <= row.time) & (news_parsed['pb_time'] >= end_time)
    mask = (news_parsed['pb_time'] <= row.time)
    news_negative_candidate = news_parsed.loc[mask]
    news_negative_list = news_negative_candidate.index.to_list()
    user_pos_list = [x.replace('-1','') for x in user_pos_df.loc[row.user].split(' ') if x.endswith('-1')]
    news_negative_list = list(set(news_negative_list)-set(user_pos_list))
    if len(neg_list) > len(news_negative_list):
        times = (len(neg_list)//len(news_negative_list)+1)
        news_negative_list = news_negative_list*times
    candidate_news_pb = pos_list
    if sampling_type == 'impre':
        neg_list = random.sample(news_negative_list,len(neg_list))
        neg_list = [x+'-0' for x in neg_list]
    # neg_list = [x+'-0' for x in news_negative_list]

    candidate_news_pb.extend(neg_list)
    random.shuffle(candidate_news_pb)
    test_behaviors.at[i,'impression'] = ' '.join(candidate_news_pb)    



test_behaviors.set_index(0,drop=True,inplace=True)
test_behaviors.reset_index(drop=True,inplace=True)

test_behaviors.to_csv(
        os.path.join(test_dir,f'behaviors_lt{lifetime}_{sampling_type}.tsv'),
        header=None,
        sep='\t')
# 
