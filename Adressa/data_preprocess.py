
from config import model_name
import pandas as pd
import json
import math
from tqdm import tqdm
from os import path
from pathlib import Path
import random
from nltk.tokenize import word_tokenize
import nltk
import numpy as np
import csv
import importlib
import os
import shutil
try:
    config = getattr(importlib.import_module('config'), f"{model_name}Config")
except AttributeError:
    print(f"{model_name} not included!")
    exit()


def parse_behaviors(source, target, user2int_path,mode):
    """
    Parse behaviors file in training set.
    Args:
        source: source behaviors file
        target: target behaviors file
        user2int_path: path for saving user2int file
    """
    print(f"Parse {source}")
    behaviors = pd.read_table(
            source,
            header=None,
            names=['user', 'time', 'clicked_news', 'impressions'])
    behaviors.clicked_news.fillna(' ', inplace=True)
    if mode == 'train':

        user2int = {}
        for row in behaviors.itertuples(index=False):
            if row.user not in user2int:
                user2int[row.user] = len(user2int) + 1
        
        pd.DataFrame(user2int.items(), columns=['user',
                                                'int']).to_csv(user2int_path,
                                                            sep='\t',
                                                            index=False)
        print(
            f'Please modify `num_users` in `src/config.py` into 1 + {len(user2int)}'
        )
    elif mode == 'val':
        user2int = dict(
            pd.read_table(user2int_path, na_filter=False).values.tolist())
    for row in behaviors.itertuples():
        behaviors.at[row.Index, 'user'] = user2int[row.user]
    behaviors.to_csv(
        target,
        sep='\t',
        index=False,
        header=None)
def parse_news(source, target, roberta_output_dir, category2int_path,
               word2int_path, entity2int_path, mode):
    """
    Parse news for training set and test set
    Args:
        source: source news file
        target: target news file
        if mode == 'train':
            category2int_path, word2int_path, entity2int_path: Path to save
        elif mode == 'test':
            category2int_path, word2int_path, entity2int_path: Path to load from
    """
    print(f"Parse {source}")
    news = pd.read_table(source,
                         header=None,
                         usecols=[0, 1, 2, 3, 4],
                         quoting=csv.QUOTE_NONE,
                         names=[
                             'id', 'category', 'subcategory', 'title',
                             'abstract'
                         ])  # TODO try to avoid csv.QUOTE_NONE
    news.fillna(' ', inplace=True)
    news['title'] = news['title'].replace('#TAB#','\t').replace('#ENTER#','\n')
    news['abstract'] = news['abstract'].replace('#TAB#','\t').replace('#ENTER#','\n')

    def parse_row(row):
        new_row = [
            row.id,
            category2int[row.category] if row.category in category2int else 0,
            category2int[row.subcategory] if row.subcategory in category2int else 0,
            [0] * config.num_words_title,
            [0] * config.num_words_abstract,
            [0] * config.num_words_cat
        ]
        # print(new_row)
        try:
            for i, w in enumerate(word_tokenize(row.title.lower(),language="norwegian")):
                if w in word2int.keys():
                    new_row[3][i] = word2int[w]
        except IndexError:
            pass
        try:
            for i, w in enumerate(word_tokenize(row.abstract.lower(),language="norwegian")):
                if w in word2int:
                    new_row[4][i] = word2int[w]
        except IndexError:
            pass
        try:
            for i, w in enumerate(word_tokenize((row.category + ' ' + row.subcategory).lower(),language="norwegian")):
                if w in word2int:
                    new_row[5][i] = word2int[w]
        except IndexError:
            pass

        return pd.Series(new_row,
                         index=[
                             'id', 'category', 'subcategory', 'title',
                             'abstract','category_word'
                         ])

    if mode == 'train':
        category2int = {}
        word2int = {}
        word2freq = {}
        for row in tqdm(news.itertuples(index=False)):

            if row.category not in category2int:
                category2int[row.category] = len(category2int) + 1
            if row.subcategory not in category2int:
                category2int[row.subcategory] = len(category2int) + 1

            for w in word_tokenize(row.title.lower(),language="norwegian"):
                if w not in word2freq.keys():
                    word2freq[w] = 1
                else:
                    word2freq[w] += 1
            for w in word_tokenize(row.abstract.lower(),language="norwegian"):
                if w not in word2freq:
                    word2freq[w] = 1
                else:
                    word2freq[w] += 1
            for w in word_tokenize(row.category.lower(),language="norwegian"):
                if w not in word2freq:
                    word2freq[w] = 1
                else:
                    word2freq[w] += 1
            for w in word_tokenize(row.subcategory.lower(),language="norwegian"):
                if w not in word2freq:
                    word2freq[w] = 1
                else:
                    word2freq[w] += 1

        for k, v in word2freq.items():
            if v >= config.word_freq_threshold:
                word2int[k] = len(word2int) + 1

        parsed_news = news.apply(parse_row, axis=1)
        parsed_news.to_csv(target, sep='\t', index=False)

        pd.DataFrame(category2int.items(),
                     columns=['category', 'int']).to_csv(category2int_path,
                                                         sep='\t',
                                                         index=False)
        print(
            f'Please modify `num_categories` in `src/config.py` into 1 + {len(category2int)}'
        )

        pd.DataFrame(word2int.items(), columns=['word',
                                                'int']).to_csv(word2int_path,
                                                               sep='\t',
                                                               index=False)
        print(
            f'Please modify `num_words` in `src/config.py` into 1 + {len(word2int)}'
        )


    elif mode == 'test':
        # na_filter=False is needed since nan is also a valid word
        category2int = dict(pd.read_table(category2int_path).values.tolist())        
        word2int = dict(
            pd.read_table(word2int_path, na_filter=False).values.tolist())


        parsed_news = news.apply(parse_row, axis=1)
        parsed_news.to_csv(target, sep='\t', index=False)

    else:
        print('Wrong mode!')


def generate_word_embedding(source, target, word2int_path):
    """
    Generate from pretrained word embedding file
    If a word not in embedding file, initial its embedding by N(0, 1)
    Args:
        source: path of pretrained word embedding file, e.g. glove.840B.300d.txt
        target: path for saving word embedding. Will be saved in numpy format
        word2int_path: vocabulary file when words in it will be searched in pretrained embedding file
    """
    # na_filter=False is needed since nan is also a valid word
    # word, int
    word2int = pd.read_table(word2int_path, na_filter=False, index_col=0)
    source_embedding = pd.read_table(source,
                                     index_col=0,
                                     sep=' ',
                                     header=None,
                                     quoting=csv.QUOTE_NONE,
                                     names=range(config.word_embedding_dim))
    # word, vector
    word2int.index.rename('word', inplace=True)
    source_embedding.index.rename('word', inplace=True)
    # word, int, vector
    merged = word2int.merge(source_embedding,
                            how='inner',
                            left_index=True,
                            right_index=True)
    merged.set_index('int', inplace=True)

    missed_index = np.setdiff1d(np.arange(len(word2int) + 1),
                                merged.index.values)
    missed_embedding = pd.DataFrame(data=np.random.normal(
        size=(len(missed_index), config.word_embedding_dim)))
    missed_embedding['int'] = missed_index
    missed_embedding.set_index('int', inplace=True)

    final_embedding = pd.concat([merged, missed_embedding]).sort_index()
    np.save(target, final_embedding.values)

    print(
        f'Rate of word missed in pretrained embedding: {(len(missed_index)-1)/len(word2int):.4f}'
    )




if __name__ == '__main__':
    preprocess_data_folder = './data/preprocessed_data'
    for data_type in os.listdir(preprocess_data_folder):
        if '(type2)' in data_type:
            news_dir = f'./data/preprocessed_data/{data_type}'
            train_dir = f'./data/preprocessed_data/{data_type}/train'
            test_dir = f'./data/preprocessed_data/{data_type}/test'
            file_name = 'behaviors.tsv'
            news_file = 'news_parsed.tsv'
            print()
            print(f'Process {data_type} data for training')
            
            print('Parse training behaviors')
            parse_behaviors(path.join(train_dir, file_name),
                            path.join(train_dir, 'behaviors.tsv'),
                            path.join(train_dir, 'user2int.tsv'),
                            mode='train')


            print('Parse news(raw)')
            parse_news(path.join(news_dir, 'total_news(raw).tsv'),
                    path.join(train_dir, news_file),
                    path.join(train_dir, 'roberta'),
                    path.join(train_dir, 'category2int.tsv'),
                    path.join(train_dir, 'word2int.tsv'),
                    path.join(train_dir, 'entity2int.tsv'),
                    mode='train')
            shutil.copyfile(path.join(train_dir, news_file), path.join(test_dir, news_file))

            print('Generate word embedding')
            generate_word_embedding(
                f'./data/emb/model.txt',
                path.join(train_dir, 'pretrained_word_embedding.npy'),
                path.join(train_dir, 'word2int.tsv'))


            # # print('\nProcess data for validation')
            
            # parse_behaviors(path.join(test_dir, file_name),
            #                 path.join(test_dir, 'behaviors.tsv'),
            #                 path.join(train_dir, 'user2int.tsv'),
            #                 mode='val')
            # print('Parse test news(raw)')
            # parse_news(path.join(test_dir, 'news(raw).tsv'),
            #         path.join(test_dir, news_file),
            #         path.join(test_dir, 'roberta'),
            #         path.join(train_dir, 'category2int.tsv'),
            #         path.join(train_dir, 'word2int.tsv'),
            #         path.join(train_dir, 'entity2int.tsv'),
            #         mode='test')

            
