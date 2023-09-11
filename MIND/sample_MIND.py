import json
import random
sample_num = 200000
# 1. randomly sample by users
user_set = set()

with open('./data/MIND-large/train/behaviors.tsv', 'r', encoding='utf-8') as f:
    for line in f:
        impression_ID, user_ID, time, history, impressions = line.strip().split('\t')
        user_set.add(user_ID)
with open('./data/MIND-large/val/behaviors.tsv', 'r', encoding='utf-8') as f:
    for line in f:
        impression_ID, user_ID, time, history, impressions = line.strip().split('\t')
        user_set.add(user_ID)
user_list = list(user_set)
random.shuffle(user_list)
assert sample_num <= len(user_list), 'sample num must be less than or equal to 1000000'
sample_user_list = random.sample(user_list, sample_num)
with open('./data/MIND200k/sample_users.json', 'w', encoding='utf-8') as f:
    json.dump(sample_user_list, f)
sampled_user_set = set(sample_user_list)
# 2. write sampled behavior file
with open('./data/MIND-large/train/behaviors.tsv', 'r', encoding='utf-8') as f:
    with open('./data/MIND200k/train/behaviors.tsv', 'w', encoding='utf-8') as train_f:
        for line in f:
            impression_ID, user_ID, time, history, impressions = line.strip().split('\t')
            if user_ID in sampled_user_set:
                train_f.write(line)
cnt = 0
with open('./data/MIND-large/val/behaviors.tsv', 'r', encoding='utf-8') as f:
    with open('./data/MIND200k/val/behaviors.tsv', 'w', encoding='utf-8') as dev_f:
        with open('./data/MIND200k/test/behaviors.tsv', 'w', encoding='utf-8') as test_f:
            for line in f:
                impression_ID, user_ID, time, history, impressions = line.strip().split('\t')
                if user_ID in sampled_user_set:
                    if cnt % 2 == 0:
                        dev_f.write(line)  # half-split for val
                    else:
                        test_f.write(line) # half-split for test
                    cnt += 1
# 3. write sampled news file
for mode in ['train', 'val', 'test']:
    with open('./data/MIND200k/%s/behaviors.tsv' % (mode), 'r', encoding='utf-8') as f:
        news_set = set()
        for line in f:
            impression_ID, user_ID, time, history, impressions = line.strip().split('\t')
            if len(history) > 0:
                news = history.split(' ')
                for n in news:
                    news_set.add(n)
            if len(impressions) > 0:
                news = impressions.split(' ')
                for n in news:
                    news_set.add(n[:-2])
        with open('./data/MIND-large/%s/news.tsv' % ('val' if mode == 'test' else mode), 'r', encoding='utf-8') as _f:
            with open('./data/MIND200k/%s/news.tsv' % mode, 'w', encoding='utf-8') as __f:
                for line in _f:
                    news_ID, category, subCategory, title, abstract, _, _, _ = line.split('\t')
                    if news_ID in news_set:
                        __f.write(line)
