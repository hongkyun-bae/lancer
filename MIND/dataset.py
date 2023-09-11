from torch.utils.data import Dataset
import pandas as pd
from ast import literal_eval
from os import path
import numpy as np
from config import model_name
import importlib
import torch
import random
try:
    config = getattr(importlib.import_module('config'), f"{model_name}Config")
except AttributeError:
    print(f"{model_name} not included!")
    exit()


class BaseDataset(Dataset):
    def __init__(self, behaviors_path, news_path):
        super(BaseDataset, self).__init__()
        assert all(attribute in [
            'category', 'subcategory', 'title', 'abstract', 'title_entities',
            'abstract_entities'
        ] for attribute in config.dataset_attributes['news'])
        assert all(attribute in ['user', 'clicked_news_length']
                   for attribute in config.dataset_attributes['record'])

        self.behaviors_parsed = pd.read_table(behaviors_path)
        drop_columns = [x for x in self.behaviors_parsed.columns.values.tolist() if x.startswith('candidate_news')]
        if config.training_type.startswith('_impre'):
                # print(row.candidate_news.split())
            drop_columns.remove('candidate_news')
        elif config.training_type.startswith('_recent'):
            self.behaviors_parsed.dropna(inplace=True)
            drop_columns.remove('candidate_news_pb')
        elif config.training_type.startswith('_random'):
            self.behaviors_parsed.dropna(inplace=True)
            drop_columns.remove('candidate_news_random')
        else:
            self.behaviors_parsed.dropna(inplace=True)
            drop_columns.remove(f'candidate_news{config.training_type}')
            
        self.behaviors_parsed.drop(columns= drop_columns)

        self.news_parsed = pd.read_table(
            news_path,
            index_col='id',
            usecols=['id'] + config.dataset_attributes['news'],
            converters={
                attribute: literal_eval
                for attribute in set(config.dataset_attributes['news']) & set([
                    'title', 'abstract', 'title_entities', 'abstract_entities'
                ])
            })
        self.news_id2int = {x: i for i, x in enumerate(self.news_parsed.index)}
        self.news2dict = self.news_parsed.to_dict('index')
        for key1 in self.news2dict.keys():
            for key2 in self.news2dict[key1].keys():
                self.news2dict[key1][key2] = torch.tensor(
                    self.news2dict[key1][key2])
        padding_all = {
            'category': 0,
            'subcategory': 0,
            'title': [0] * config.num_words_title,
            'abstract': [0] * config.num_words_abstract,
            'title_entities': [0] * config.num_words_title,
            'abstract_entities': [0] * config.num_words_abstract
        }
        for key in padding_all.keys():
            padding_all[key] = torch.tensor(padding_all[key])

        self.padding = {
            k: v
            for k, v in padding_all.items()
            if k in config.dataset_attributes['news']
        }

    def __len__(self):
        return len(self.behaviors_parsed)

    def __getitem__(self, idx):
        item = {}
        row = self.behaviors_parsed.iloc[idx]
        if 'user' in config.dataset_attributes['record']:
            item['user'] = row.user
        item["clicked"] = list(map(int, row.clicked.split()))
        try:
            if config.training_type.startswith('_impre'):
                # print(row.candidate_news.split())
                item["candidate_news"] = [
                    self.news2dict[x] for x in row.candidate_news.split()
                ]
            elif config.training_type.startswith('_recent'):
                news_list = random.sample(row.candidate_news_pb.split()[1:], config.negative_sampling_ratio)
                news_list.insert(0,row.candidate_news.split()[0])
                # print(news_list)
                item["candidate_news"] = [
                    self.news2dict[x] for x in news_list
                ]
            elif config.training_type.startswith('_random'):
                news_list = random.sample(row.candidate_news_random.split()[1:], config.negative_sampling_ratio)
                news_list.insert(0,row.candidate_news.split()[0])
                item["candidate_news"] = [
                    self.news2dict[x] for x in news_list
                ]
            elif config.training_type.startswith('_rev_current_log_ltpop'):
                news_list = random.sample(row.candidate_news_rev_current_log_ltpop.split()[1:], config.negative_sampling_ratio)
                news_list.insert(0,row.candidate_news.split()[0])
                item["candidate_news"] = [
                    self.news2dict[x] for x in news_list
                ]
        except:
            print(idx,row)
            assert 1 == 0
            # print(news_list) 
        item["clicked_news"] = [
            self.news2dict[x]
            for x in row.clicked_news.split()[:config.num_clicked_news_a_user]
        ]
        if 'clicked_news_length' in config.dataset_attributes['record']:
            item['clicked_news_length'] = len(item["clicked_news"])
        repeated_times = config.num_clicked_news_a_user - \
            len(item["clicked_news"])
        assert repeated_times >= 0
        item["clicked_news"] = [self.padding
                                ] * repeated_times + item["clicked_news"]

        return item
