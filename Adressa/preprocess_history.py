import os
import json
import time
from datetime import datetime
# from tqdm import tqdm

RAW_DATA_DIR = './data/raw_data/Adressa/log_data'
NEWS_DIR = './data/preprocessed_data'
BODY_NEWS_DIR = './data/raw_data/Adressa/content_data'
DATA_LENGTH_LIST = ['5w']
body_news_list = os.listdir(BODY_NEWS_DIR)
DATA_SPLIT_WEEK = [3, 1, 1]    # History, training, and test periods

## Not necessary...