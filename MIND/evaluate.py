import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import torch
from config import model_name
from torch.utils.data import Dataset, DataLoader
from os import path
import sys
import pandas as pd
from ast import literal_eval
import importlib
import math
import os
from tkinter import X
from datetime import timedelta
import random
import time
import sys
from pathlib import Path

try:
    Model = getattr(importlib.import_module(f"model.{model_name}"), model_name)
    config = getattr(importlib.import_module('config'), f"{model_name}Config")
except AttributeError:
    print(f"{model_name} not included!")
    exit()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_sigmoid = False

dataset = config.test_type
lifetime = int(dataset.split('_')[1][2:])
k = dataset.split('_')[2][1:]


# k = '30'   #all, 30
dataset_type = config.dataset

def sigmoid(x,a=0.1):
    if x < -50/a:
        return 0
    elif x > 50/a:
        return 1
    return 1 / (1 +np.exp(-a*x))

def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2**y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


def value2rank(d):
    values = list(d.values())
    ranks = [sorted(values, reverse=True).index(x) for x in values]
    return {k: ranks[i] + 1 for i, k in enumerate(d.keys())}


class NewsDataset(Dataset):
    """
    Load news for evaluation.
    """
    def __init__(self, news_path):
        super(NewsDataset, self).__init__()
        self.news_parsed = pd.read_table(
            news_path,
            usecols=['id'] + config.dataset_attributes['news'],
            converters={
                attribute: literal_eval
                for attribute in set(config.dataset_attributes['news']) & set([
                    'title', 'abstract', 'title_entities', 'abstract_entities'
                ])
            })
        self.news2dict = self.news_parsed.to_dict('index')
        for key1 in self.news2dict.keys():
            for key2 in self.news2dict[key1].keys():
                if type(self.news2dict[key1][key2]) != str:
                    self.news2dict[key1][key2] = torch.tensor(
                        self.news2dict[key1][key2])

    def __len__(self):
        return len(self.news_parsed)

    def __getitem__(self, idx):
        item = self.news2dict[idx]
        return item


class UserDataset(Dataset):
    """
    Load users for evaluation, duplicated rows will be dropped
    """
    def __init__(self, behaviors_path, user2int_path):
        super(UserDataset, self).__init__()
        self.behaviors = pd.read_table(behaviors_path,
                                       header=None,
                                       usecols=[1, 3],
                                       names=['user', 'clicked_news'])
        self.behaviors.clicked_news.fillna(' ', inplace=True)
        self.behaviors.drop_duplicates(inplace=True)
        user2int = dict(pd.read_table(user2int_path).values.tolist())
        user_total = 0
        user_missed = 0
        for row in self.behaviors.itertuples():
            user_total += 1
            if row.user in user2int:
                self.behaviors.at[row.Index, 'user'] = user2int[row.user]
            else:
                user_missed += 1
                self.behaviors.at[row.Index, 'user'] = 0
        if model_name == 'LSTUR':
            print(f'User miss rate: {user_missed/user_total:.4f}')

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, idx):
        row = self.behaviors.iloc[idx]
        item = {
            "user":
            row.user,
            "clicked_news_string":
            row.clicked_news,
            "clicked_news":
            row.clicked_news.split()[:config.num_clicked_news_a_user]
        }
        item['clicked_news_length'] = len(item["clicked_news"])
        repeated_times = config.num_clicked_news_a_user - len(
            item["clicked_news"])
        assert repeated_times >= 0
        item["clicked_news"] = ['PADDED_NEWS'
                                ] * repeated_times + item["clicked_news"]

        return item


class BehaviorsDataset(Dataset):
    """
    Load behaviors for evaluation, (user, time) pair as session
    """
    def __init__(self, behaviors_path):
        super(BehaviorsDataset, self).__init__()
        self.behaviors = pd.read_table(behaviors_path,
                                       header=None,
                                       usecols=range(5),
                                       names=[
                                           'impression_id', 'user', 'time',
                                           'clicked_news', 'impressions'
                                       ])
        self.behaviors.clicked_news.fillna(' ', inplace=True)
        self.behaviors.impressions = self.behaviors.impressions.str.split()

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, idx):
        row = self.behaviors.iloc[idx]
        item = {
            "impression_id": row.impression_id,
            "user": row.user,
            "time": row.time,
            "clicked_news_string": row.clicked_news,
            "impressions": row.impressions
        }
        return item


def calculate_single_user_metric(pair):
    try:
        auc = roc_auc_score(*pair)
        mrr = mrr_score(*pair)
        ndcg5 = ndcg_score(*pair, 5)
        ndcg10 = ndcg_score(*pair, 10)
        return [auc, mrr, ndcg5, ndcg10]
    except ValueError:
        return [np.nan] * 4


@torch.no_grad()
def evaluate(model, directory, num_workers, max_count=sys.maxsize):
    """
    Evaluate model on target directory.
    Args:
        model: model to be evaluated
        directory: the directory that contains two files (behaviors.tsv, news_parsed.tsv)
        num_workers: processes number for calculating metrics
    Returns:
        AUC
        MRR
        nDCG@5
        nDCG@10
    """
    global news_encoder_time, user_encoder_time, click_predictor_time

    news_dataset = NewsDataset(path.join(directory, 'news_parsed.tsv'))
    news_dataloader = DataLoader(news_dataset,
                                 batch_size=config.batch_size * 16,
                                 shuffle=False,
                                 num_workers=config.num_workers,
                                 drop_last=False,
                                 pin_memory=True)

    news2vector = {}
    news_encoder_start = time.time()
    for minibatch in tqdm(news_dataloader,
                          desc="Calculating vectors for news"):
        news_ids = minibatch["id"]
        if any(id not in news2vector for id in news_ids):
            news_vector = model.get_news_vector(minibatch)
            for id, vector in zip(news_ids, news_vector):
                if id not in news2vector:
                    news2vector[id] = vector

    news2vector['PADDED_NEWS'] = torch.zeros(
        list(news2vector.values())[0].size())
    # print(news2vector)
    # print(news2vector)
    news_encoder_time = time.time() - news_encoder_start
    
    if dataset.startswith(f'_lt{lifetime}_K'):
        
        
        train_dir = f'./data/{dataset_type}/train'
        test_dir = f'./data/{dataset_type}/test'
        val_dir = f'./data/{dataset_type}/val'

        
        
        train_behaviors = pd.read_table(os.path.join(train_dir,'behaviors.tsv'),header=None)
        train_behaviors.rename(columns={1:'user', 2:'time',4:'impressions'}, inplace=True)
        train_behaviors['time'] = pd.to_datetime(train_behaviors['time'],format= "%m/%d/%Y %I:%M:%S %p")

        test_behaviors = pd.read_table(os.path.join(test_dir,'behaviors.tsv'),header=None)
        test_behaviors.rename(columns={1:'user', 2:'time',4:'impressions'}, inplace=True)
        test_behaviors['time'] = pd.to_datetime(test_behaviors['time'],format= "%m/%d/%Y %I:%M:%S %p")

        # val_behaviors = pd.read_table(os.path.join(val_dir,'behaviors.tsv'),header=None)
        # val_behaviors.rename(columns={1:'user', 2:'time',4:'impressions'}, inplace=True)
        # val_behaviors['time'] = pd.to_datetime(val_behaviors['time'],format= "%m/%d/%Y %I:%M:%S %p")
        # behaviors = pd.concat([train_behaviors, val_behaviors, test_behaviors])
        behaviors = pd.concat([train_behaviors, test_behaviors])

        news_parsed = pd.read_table(os.path.join(test_dir,'news_parsed_pb_time.tsv'),index_col = 'id')
        news_parsed['pb_time'] = pd.to_datetime(news_parsed['pb_time'])

        
        user_pos_df = behaviors.groupby(by='user')['impressions'].apply(lambda x: ' '.join(x))
        behaviors_dataset = BehaviorsDataset(path.join(directory, f'behaviors.tsv'))
        user_dataset = UserDataset(path.join(directory, f'behaviors.tsv'),
                        f'data/{dataset_type}/train/user2int.tsv')
        
    else:
        user_dataset = UserDataset(path.join(directory, f'behaviors.tsv'),
                                f'data/{dataset_type}/train/user2int.tsv')
        behaviors_dataset = BehaviorsDataset(path.join(directory, f'behaviors.tsv'))
    user_dataloader = DataLoader(user_dataset,
                                batch_size=config.batch_size * 16,
                                shuffle=False,
                                num_workers=config.num_workers,
                                drop_last=False,
                                pin_memory=True)
    user2vector = {}
    user_encoder_start = time.time()
    for minibatch in tqdm(user_dataloader,
                        desc="Calculating vectors for users"):
        user_strings = minibatch["clicked_news_string"]
        if any(user_string not in user2vector for user_string in user_strings):
            clicked_news_vector = torch.stack([
                torch.stack([news2vector[x].to(device) for x in news_list],
                            dim=0) for news_list in minibatch["clicked_news"]
            ],
                                            dim=0).transpose(0, 1)
            if model_name == 'LSTUR':
                user_vector = model.get_user_vector(
                    minibatch['user'], minibatch['clicked_news_length'],
                    clicked_news_vector)
            else:
                user_vector = model.get_user_vector(clicked_news_vector)
            for user, vector in zip(user_strings, user_vector):
                if user not in user2vector:
                    user2vector[user] = vector
    user_encoder_time = time.time() - user_encoder_start


    behaviors_dataloader = DataLoader(behaviors_dataset,
                                      batch_size=1,
                                      shuffle=False,
                                      num_workers=config.num_workers)

    count = 0
    aucs = []
    mrrs = []
    ndcg5s = []
    ndcg10s = []
    nan_count = 0
    news_df = pd.read_table(os.path.join(f'./data/{dataset_type}/test','news_parsed_pb_time.tsv'),index_col = 'id')
    news_df['pb_time'] = pd.to_datetime(news_df['pb_time'])
    click_predictor_time = 0
    if dataset.startswith(f'_lt{lifetime}_K'):
        for minibatch in tqdm(behaviors_dataloader,desc="Calculating all probabilities"):
            count += 1
            

            minibatch['time'][0] = pd.to_datetime(minibatch['time'][0],format= "%m/%d/%Y %I:%M:%S %p")

            pos_list = [news[0] for news in minibatch['impressions'] if news[0].endswith('-1')]
            end_time = minibatch['time'][0] - timedelta(hours=lifetime)
            mask = (news_parsed['pb_time'] <= minibatch['time'][0]) & (news_parsed['pb_time'] >= end_time)
            news_negative_candidate = news_parsed.loc[mask]
            news_negative_list = news_negative_candidate.index.to_list()
            user_pos_list = [x.replace('-1','') for x in user_pos_df.loc[minibatch['user'][0]].split(' ') if x.endswith('-1')]
            news_negative_list = list(set(news_negative_list)-set(user_pos_list))
            click_predictor_start = time.time()
            try:
                news_negative_list = random.sample(news_negative_list,int(k))
            except:
                pass
            candidate_news_pb = pos_list
            neg_list = [x+'-0' for x in news_negative_list]

            candidate_news_pb.extend(neg_list)
            random.shuffle(candidate_news_pb)

            # print(news2vector)

            candidate_news_vector = torch.stack([
                news2vector[news.split('-')[0]]
                for news in candidate_news_pb
            ], dim=0)
            user_vector = user2vector[minibatch['clicked_news_string'][0]]
            click_probability = model.get_prediction(candidate_news_vector, user_vector)

            y_pred = click_probability.tolist()

            if config.test_filter != False:
                impre_news_list = [news.split('-')[0] for news in candidate_news_pb]
                news_publish_time = news_df.loc[impre_news_list]['publish_time'].to_list()
                news_age = [(minibatch['time'].values[0]-publish_time).total_seconds()/3600.00 for publish_time in news_publish_time ]
                freshness_filter = [sigmoid(36-age,config.test_filter) if age >=0 else 0 for age in news_age]
                assert len(freshness_filter) == len(y_pred)
                y_pred = [a * b for a, b in zip(y_pred, freshness_filter)]

            y = [int(news.split('-')[1]) for news in candidate_news_pb]

            try:
                auc = roc_auc_score(y, y_pred)
                mrr = mrr_score(y, y_pred)
                ndcg5 = ndcg_score(y, y_pred, 5)
                ndcg10 = ndcg_score(y, y_pred, 10)
            except:
                try:
                    auc = roc_auc_score(y, y_pred[0])
                    mrr = mrr_score(y, y_pred[0])
                    ndcg5 = ndcg_score(y, y_pred[0], 5)
                    ndcg10 = ndcg_score(y, y_pred[0], 10)
                except ValueError:
                    continue
            if math.isnan(auc + mrr + ndcg5  + ndcg10):
                nan_count += 1
                print('nan_count',nan_count)
                continue
            else:
                aucs.append(auc)
                mrrs.append(mrr)
                ndcg5s.append(ndcg5)
                ndcg10s.append(ndcg10)
            click_predictor_time += time.time() - click_predictor_start

    else:
        for minibatch in tqdm(behaviors_dataloader,desc="Calculating probabilities"):
            # print(minibatch)
            count += 1
            click_predictor_start = time.time()
            
            candidate_news_vector = torch.stack([
                news2vector[news[0].split('-')[0]]
                for news in minibatch['impressions']
            ], dim=0)
            user_vector = user2vector[minibatch['clicked_news_string'][0]]
            click_probability = model.get_prediction(candidate_news_vector, user_vector)

            y_pred = click_probability.tolist()
            y = [int(news[0].split('-')[1]) for news in minibatch['impressions']]

            try:
                auc = roc_auc_score(y, y_pred)
                mrr = mrr_score(y, y_pred)
                ndcg5 = ndcg_score(y, y_pred, 5)
                ndcg10 = ndcg_score(y, y_pred, 10)
            except:
                try:
                    auc = roc_auc_score(y, y_pred[0])
                    mrr = mrr_score(y, y_pred[0])
                    ndcg5 = ndcg_score(y, y_pred[0], 5)
                    ndcg10 = ndcg_score(y, y_pred[0], 10)
                except ValueError:
                    continue
            if math.isnan(auc + mrr + ndcg5  + ndcg10):
                nan_count += 1
                print('nan_count',nan_count)
                continue
            else:
                aucs.append(auc)
                mrrs.append(mrr)
                ndcg5s.append(ndcg5)
                ndcg10s.append(ndcg10)
            click_predictor_time += time.time() - click_predictor_start

    print('nan_count:',nan_count,'nan ratio:',round(100*nan_count/count,4))


    return np.mean(aucs), np.mean(mrrs), np.mean(ndcg5s), np.mean(ndcg10s)


if __name__ == '__main__':
    print('Using device:', device)
    print(f'Evaluating model {model_name}')
    # Don't need to load pretrained word/entity/context embedding
    # since it will be loaded from checkpoint later
    model = Model(config).to(device)
    from train import latest_checkpoint  # Avoid circular imports
    if config.checkpoint_num == '':
        checkpoint_path = latest_checkpoint(path.join(f'./checkpoint/{dataset_type}', f'{model_name}{config.training_type}_{config.negative_sampling_ratio}'))
        # print('aaaaaaaaaaa')        
    else:
        checkpoint_path = latest_checkpoint(path.join(f'./checkpoint/{dataset_type}_{config.checkpoint_num}', f'{model_name}{config.training_type}_{config.negative_sampling_ratio}'))

    if checkpoint_path is None:
        print('No checkpoint file found!')
        exit()
    print(f"Load saved parameters in {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    auc, mrr, ndcg5, ndcg10 = evaluate(model, f'./data/{dataset_type}/test',
                                       config.num_workers)
    print(
        f'AUC: {auc:.6f}\nMRR: {mrr:.6f}\nnDCG@5: {ndcg5:.6f}\nnDCG@10: {ndcg10:.6f}'
    )
    result_timedir = f'./results/{dataset_type}/time/'
    Path(result_timedir).mkdir(parents=True, exist_ok=True)
    if config.checkpoint_num == '':
        result_file_ori = f'./results/{dataset_type}/{model_name}{config.training_type}_{config.negative_sampling_ratio}{dataset}'
    else:    
        result_file_ori = f'./results/{dataset_type}/{model_name}{config.training_type}_{config.negative_sampling_ratio}{dataset}-{config.checkpoint_num}'
    number = 1
    while True:
        result_file = f"{result_file_ori}-{str(number)}"
        if os.path.isfile(result_file+".txt"):
            number += 1
        else:
            break
    time_file = result_file.split('/')[-1]
    result_file = result_file +".txt"
    with open(result_file,'w') as wf:
        wf.writelines(f'{auc:.6f}, {mrr:.6f}, {ndcg5:.6f}, {ndcg10:.6f}')
    with open(f'./results/{dataset_type}/time/{time_file}.txt','w') as wf:
        wf.writelines(f'news_encoder: {news_encoder_time:.1f}\nuser_encoder: {user_encoder_time:.1f}\nclick_predictor: {click_predictor_time:.1f}')