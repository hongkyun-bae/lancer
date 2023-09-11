
import os
import json
import time
from datetime import datetime
from tqdm import tqdm

RAW_DATA_DIR = './data/raw_data/Adressa/log_data'
NEWS_DIR = './data/preprocessed_data'
BODY_NEWS_DIR = './data/raw_data/Adressa/content_data'
# DATA_LENGTH_LIST = ['5w','6w','7w']
DATA_LENGTH_LIST = ['5w']
body_news_list = os.listdir(BODY_NEWS_DIR)
# DATA_SPLIT_WEEK = [3,3,1]            #history, test week
DATA_SPLIT_WEEK = [4,1,1]            #history, test week

##

# def parse_time(timestamp: str) -> str:
def parse_time(timestamp):
    return datetime.fromtimestamp(float(timestamp)).strftime('%Y-%m-%d %H:%M:%S')

def str_to_timestamp(string):
    return datetime.timestamp(datetime.strptime(string,'%Y-%m-%d %H:%M:%S'))


def split_data(data_length,data_split_ratio):
    if data_length.endswith('w'):
        TOT_WEEK = int(data_length[:-1])
        TRAIN_WEEK = TOT_WEEK - data_split_ratio[0] - data_split_ratio[2]
    DAY = sum(data_split_ratio)*7
    HIS_DAY = [(data_split_ratio[1] - TRAIN_WEEK)*7, (data_split_ratio[1] - TRAIN_WEEK)*7 + data_split_ratio[0]*7 ]
    TRAIN_DAY = [HIS_DAY[1],HIS_DAY[1]+TRAIN_WEEK*7]
    TEST_DAY = [TRAIN_DAY[1], DAY]   

    print(f'{DAY}DAYS: {HIS_DAY[1]-HIS_DAY[0]}/{TRAIN_DAY[1]-TRAIN_DAY[0]}/{TEST_DAY[1]-TEST_DAY[0]}')
    return DAY,HIS_DAY,TRAIN_DAY,TEST_DAY 

def preprocess_line(body):
    global user_cnt
    global news_cnt
    global user2id
    global news2id
    global user_dic
    global news_dic
    global behaviors_list
    global bodynews_click_cnt
    global body_filter_cnt
    global news_click_cnt
    global categorynews_click_cnt
    global raw_news_set
    global cat_news_set
    global body_news_set
    global body_filter_news_set
    global history_click_cnt
    global reversed_time_filter
    filter_list = ["time", "publishtime",
                "title", "userId","url", "canonicalUrl"]
    if all(filter in body.keys() for filter in filter_list):
        news_click_cnt += 1
        raw_news_set.add(body['canonicalUrl'])
        if 'category1' in body.keys():
            categorynews_click_cnt += 1
            cat_news_set.add(body['canonicalUrl'])
            
        if body['id'] in body_news_list:
            bodynews_click_cnt += 1
            body_news_set.add(body['canonicalUrl'])
            
        if 'category1' in body.keys():
            if body['id'] in body_news_list:

                body['publishtime'] = body['publishtime'].split('.')[0].replace('T',' ')
                body['time'] = parse_time(body['time'])
                if str_to_timestamp(body['publishtime']) > str_to_timestamp(body['time']):
                    reversed_time_filter += 1
                    return
                if body['canonicalUrl'] not in news2id.keys():
                    news_id = 'N'+str(news_cnt)
                    news_cnt += 1
                    news2id[body['canonicalUrl']] = news_id
                if body['userId'] not in user2id.keys():
                    user_id = 'U'+str(user_cnt)
                    user_cnt += 1
                    user2id[body['userId']] = user_id
                    
                news_id = news2id[body['canonicalUrl']]
                user_id = user2id[body['userId']]            
                if body['canonicalUrl'] not in news_dic.keys():
                    news_body_file = os.path.join(BODY_NEWS_DIR,body['id'])
                    
                    body_line = ''
                    with open(news_body_file,'r') as rf:
                        body_lines = json.loads(rf.readline())
                    for i in range(len(body_lines['fields'])):
                        if body_lines['fields'][i]['field'] == 'body':
                            body_line = ' '.join(body_lines['fields'][i]['value']).replace('\t','#TAB#').replace('\n','#ENTER#')
                            break
                    if body_line == '':
                        body_filter_cnt += 1
                        body_filter_news_set.add(body['canonicalUrl'])
                        return
                    
                    category = body['category1'].split('|')[0]
                    subcategory = body['category1'].split('|')[1]
                    body['title'] = body['title'].replace('\t','#TAB#').replace('\n','#ENTER#')
                    
                    news_dic[body['canonicalUrl']] = [news_id,category,subcategory,body['title'],body_line,body['id'],body['publishtime']]
                    news_dic[body['canonicalUrl']].append([body['time']])

                else:
                    news_dic[body['canonicalUrl']][7].append(body['time'])

                ### news_df 
                ### user.tsv 
                add_click = news_id+ ','+body['publishtime']+ ','+body['time']
                if body['userId'] not in user_dic.keys():
                    user_dic[body['userId']] =[user_id]
                    user_dic[body['userId']].append([])
                    user_dic[body['userId']].append([])
                    user_dic[body['userId']].append([])
                if data_type == 'history':
                    data_type_index = 1
                elif data_type == "train":
                    data_type_index = 2
                elif data_type == 'test':
                    data_type_index = 3
                user_dic[body['userId']][data_type_index].append([add_click])


                if data_type == "train":
                    history_list = list(set([j.split(',')[0] for i in user_dic[body['userId']][1] for j in i ]))
                    write_line = user_id+'\t'+body['time'] + '\t' + ' '.join(history_list) + '\t'+news_id+'-1'
                    behaviors_list.append(write_line)

                elif data_type == "test":
                    history_list = list(set(
                                            [j.split(',')[0] for i in user_dic[body['userId']][1] for j in i ]
                                            + [j.split(',')[0] for i in user_dic[body['userId']][2] for j in i ]))
                    write_line = user_id+'\t'+body['time'] + '\t' + ' '.join(history_list) + '\t'+news_id+'-1'
                    behaviors_list.append(write_line)
                elif data_type == "history":
                    history_click_cnt += 1

def make_user_filter():
    user_filter = []
    with open(os.path.join(NEWS_DIRECTORY,'user(raw).tsv'),'w',encoding='UTF-8') as wf3:
        for key in tqdm(user_dic.keys(),desc="make raw user tsv"):
            writeline = user_dic[key][0]+'\t'
            for click in user_dic[key][1]:
                click = click[0]
                writeline += click+';'

            writeline.strip(';')
            writeline += '\t'
            for click in user_dic[key][2]:
                click = click[0]
                writeline += click+';'

            writeline.strip(';')
            writeline += '\t'
            for click in user_dic[key][3]:
                click = click[0]
                writeline += click+';'

            writeline.strip(';')
            writeline += '\n'
            wf3.write(writeline)    
    
    
    
    with open(os.path.join(NEWS_DIRECTORY,'user.tsv'),'w',encoding='UTF-8') as wf2:
        for key in tqdm(user_dic.keys(),desc="make user filter"):
            if len(user_dic[key][1]) + len(user_dic[key][2]) + len(user_dic[key][3]) < 5:
                continue
            # if (len(user_dic[key][1]) == 0) or (len(user_dic[key][2]) == 0) or (len(user_dic[key][3]) == 0):
            #     user_filter.append(key)            
            else:
                writeline = user_dic[key][0]+'\t'
                user_filter.append(key)
                for click in user_dic[key][1]:
                    click = click[0]
                    writeline += click+';'

                writeline.strip(';')
                writeline += '\t'
                for click in user_dic[key][2]:
                    click = click[0]
                    writeline += click+';'

                writeline.strip(';')
                writeline += '\t'
                for click in user_dic[key][3]:
                    click = click[0]
                    writeline += click+';'

                writeline.strip(';')
                writeline += '\n'
                wf2.write(writeline)

    user_filter = set(user_filter)
    user_filter_dic = {x:True for x in user_filter}

    return user_filter_dic

def userfiltered_preprocess_line(body,user_filter_dic):
    global user_cnt
    global news_cnt
    global user2id
    global news2id
    global user_dic
    global news_dic
    global behaviors_list
    global body_filter_cnt2
    global history_click_cnt2
    global behaviors_user_filter_cnt
    global reversed_time_filter2

    filter_list = ["time", "publishtime",
                   'category1',
                "title", "userId","url", "canonicalUrl"]
    if all(filter in body.keys() for filter in filter_list):
        if user_filter_dic.get(body['userId'],False):
            if body['id'] in body_news_list:

                body['publishtime'] = body['publishtime'].split('.')[0].replace('T',' ')
                body['time'] = parse_time(body['time'])

                if str_to_timestamp(body['publishtime']) > str_to_timestamp(body['time']):
                    reversed_time_filter2 += 1
                    return
                news_id = news2id[body['canonicalUrl']]
                user_id = user2id[body['userId']]
                if body['canonicalUrl'] not in news_dic.keys():

                    news_body_file = os.path.join(BODY_NEWS_DIR,body['id'])
                    body_line = ''
                    with open(news_body_file,'r') as rf:
                        body_lines = json.loads(rf.readline())
                    for i in range(len(body_lines['fields'])):
                        if body_lines['fields'][i]['field'] == 'body':
                            body_line = ' '.join(body_lines['fields'][i]['value']).replace('\t','#TAB#').replace('\n','#ENTER#')
                            break
                    if body_line == '':
                        body_filter_cnt2 += 1
                        return
                    
                    category = body['category1'].split('|')[0]
                    subcategory = body['category1'].split('|')[1]
                    body['title'] = body['title'].replace('\t','#TAB#').replace('\n','#ENTER#')
                    
                    news_dic[body['canonicalUrl']] = [news_id,category,subcategory,body['title'],body_line,body['id'],body['publishtime']]
                    news_dic[body['canonicalUrl']].append([body['time']])

                else:

                    news_dic[body['canonicalUrl']][7].append(body['time'])


                ### user.tsv 
                add_click = news_id+ ','+body['publishtime']+ ','+body['time']
                if body['userId'] not in user_dic.keys():
                    user_dic[body['userId']] =[user_id]
                    user_dic[body['userId']].append([])
                    user_dic[body['userId']].append([])
                    user_dic[body['userId']].append([])
                if data_type == 'history':
                    data_type_index = 1
                elif data_type == "train":
                    data_type_index = 2
                elif data_type == 'test':
                    data_type_index = 3
                user_dic[body['userId']][data_type_index].append([add_click])
                
                if data_type == "train":
                    history_list = list(set([j.split(',')[0] for i in user_dic[body['userId']][1] for j in i ]))
                    write_line = user_id+'\t'+body['time'] + '\t' + ' '.join(history_list) + '\t'+news_id+'-1'
                    behaviors_list.append(write_line)

                elif data_type == "test":
                    history_list = list(set(
                                            [j.split(',')[0] for i in user_dic[body['userId']][1] for j in i ]
                                            + [j.split(',')[0] for i in user_dic[body['userId']][2] for j in i ]))
                    write_line = user_id+'\t'+body['time'] + '\t' + ' '.join(history_list) + '\t'+news_id+'-1'
                    behaviors_list.append(write_line)
                elif data_type == "history":
                    history_click_cnt2 += 1
        else:
            behaviors_user_filter_cnt += 1

def preprocess_file(file,userfilter = None):
    global total_cnt
    file1 = open(file, 'r')
    while True:
        total_cnt += 1
        line = file1.readline()
        if not line:
            break
        body = json.loads(line)
        if userfilter:
            userfiltered_preprocess_line(body,userfilter)
        else:
            preprocess_line(body)

def make_folder(DIR):
    try:
        os.mkdir(DIR)
    except:
        pass
def write_behavior_file(write_folder,file_name='behavior.tsv'):
    global behaviors_list
    with open(os.path.join(write_folder,file_name),'w',encoding='UTF-8') as wf:
        for line in behaviors_list:
            wf.write(line+'\n')
    behaviors_list = []
def write_news_file(write_folder,file_name='news.tsv'):
    global news_dic
    global total_news_dic
    with open(os.path.join(write_folder,file_name),'w',encoding='UTF-8') as wf:
        for key in news_dic.keys():
            wf.write(news_dic[key][0]+'\t'+news_dic[key][1]+'\t'+news_dic[key][2]+'\t'+news_dic[key][3]+'\t'+news_dic[key][4]+'\t'+news_dic[key][5]+'\t'+news_dic[key][6]+'\t'+','.join(news_dic[key][7])+'\n')
    total_news_dic = {**total_news_dic, **news_dic}
    news_dic = {}
def write_total_news_file(write_folder,file_name='total_news.tsv'):
    global total_news_dic
    with open(os.path.join(write_folder,file_name),'w',encoding='UTF-8') as wf:
        for key in total_news_dic.keys():
            wf.write(total_news_dic[key][0]+'\t'+total_news_dic[key][1]+'\t'+total_news_dic[key][2]+'\t'+total_news_dic[key][3]+'\t'+total_news_dic[key][4]+'\t'+total_news_dic[key][5]+'\t'+total_news_dic[key][6]+'\t'+','.join(total_news_dic[key][7])+'\n')
    total_news_dic = {}
        

def preprocess():
    global data_type
    global start_day_sec
    global user_filter
    DAY,HIS_DAY,TRAIN_DAY,TEST_DAY = split_data(DATA_LENGTH,DATA_SPLIT_WEEK)

        
    ## HISTORY
    print(f"\nPreprocess History({HIS_DAY[1]-HIS_DAY[0]}DAY)")
    for news in tqdm(os.listdir(RAW_DATA_DIR)[HIS_DAY[0] : HIS_DAY[1]]):
        data_type = 'history'
        data_folder = os.path.join(NEWS_DIRECTORY,data_type)
        make_folder(data_folder)
        preprocess_file(os.path.join(RAW_DATA_DIR,news))
    line = f'History news cnt: {len(news_dic)}\nHistory click cnt: {history_click_cnt}\n'
    total_click_cnt = history_click_cnt
    write_news_file(data_folder,'news(raw).tsv')
    ## TRAIN
    print(f"\nPreprocess Train Data({TRAIN_DAY[1]-TRAIN_DAY[0]}DAY)")
    for news in tqdm(os.listdir(RAW_DATA_DIR)[TRAIN_DAY[0] : TRAIN_DAY[1]]):
        data_type = 'train'
        data_folder = os.path.join(NEWS_DIRECTORY,data_type)
        make_folder(data_folder)
        preprocess_file(os.path.join(RAW_DATA_DIR,news))
    line +=  f'Train news cnt: {len(news_dic)}\nTrain click cnt: {len(behaviors_list)}\n'
    total_click_cnt += len(behaviors_list)
    write_news_file(data_folder,'news(raw).tsv')
    write_behavior_file(data_folder,'behaviors(raw).tsv')
    ## TEST
    print(f"\nPreprocess Test Data({TEST_DAY[1]-TEST_DAY[0]}DAY)")
    for news in tqdm(os.listdir(RAW_DATA_DIR)[TEST_DAY[0] : TEST_DAY[1]]):
        data_type = 'test'
        data_folder = os.path.join(NEWS_DIRECTORY,data_type)
        make_folder(data_folder)
        preprocess_file(os.path.join(RAW_DATA_DIR,news))
    line +=  f'Test news cnt: {len(news_dic)}\nTest click cnt: {len(behaviors_list)}\n'
    total_click_cnt += len(behaviors_list)
    write_news_file(data_folder,'news(raw).tsv')
    write_behavior_file(data_folder,'behaviors(raw).tsv')

    write_total_news_file(NEWS_DIRECTORY,'total_news(raw).tsv')
    total_click_cnt = 0
    if len(user_filter) == 0:
        user_filter = make_user_filter()
    wf.write(f'\user filter cnt:{len(user_filter)}\n')
    ## HISTORY
    print(f"\nPreprocess UserFiltered History({HIS_DAY[1]-HIS_DAY[0]}DAY)")
    for news in tqdm(os.listdir(RAW_DATA_DIR)[HIS_DAY[0] : HIS_DAY[1]]):
        data_type = 'history'
        data_folder = os.path.join(NEWS_DIRECTORY,data_type)
        make_folder(data_folder)
        preprocess_file(os.path.join(RAW_DATA_DIR,news),user_filter)
    line2 = f'History news cnt: {len(news_dic)}\nHistory click cnt: {history_click_cnt2}\n'
    total_click_cnt = history_click_cnt2
    write_news_file(data_folder,'news.tsv')
    ## TRAIN
    print(f"\nPreprocess UserFiltered Train Data({TRAIN_DAY[1]-TRAIN_DAY[0]}DAY)")
    for news in tqdm(os.listdir(RAW_DATA_DIR)[TRAIN_DAY[0] : TRAIN_DAY[1]]):
        data_type = 'train'
        data_folder = os.path.join(NEWS_DIRECTORY,data_type)
        make_folder(data_folder)
        preprocess_file(os.path.join(RAW_DATA_DIR,news),user_filter)
    line2 +=  f'Train news cnt: {len(news_dic)}\nTrain click cnt: {len(behaviors_list)}\n'
    total_click_cnt += len(behaviors_list)
    write_news_file(data_folder,'news.tsv')
    write_behavior_file(data_folder,'behaviors.tsv')
    ## TEST
    print(f"\nPreprocess UserFiltered Test Data({TEST_DAY[1]-TEST_DAY[0]}DAY)")
    for news in tqdm(os.listdir(RAW_DATA_DIR)[TEST_DAY[0] : TEST_DAY[1]]):
        data_type = 'test'
        data_folder = os.path.join(NEWS_DIRECTORY,data_type)
        make_folder(data_folder)
        preprocess_file(os.path.join(RAW_DATA_DIR,news),user_filter)
    line2 +=  f'Test news cnt: {len(news_dic)}\nTest click cnt: {len(behaviors_list)}\n'
    total_click_cnt += len(behaviors_list)
    write_news_file(data_folder,'news.tsv')
    write_behavior_file(data_folder,'behaviors.tsv')
    
    write_total_news_file(NEWS_DIRECTORY)

user2id = {}
news2id = {}
news_cnt = 1
user_cnt = 1
user_filter = {}
for DATA_LENGTH in DATA_LENGTH_LIST:
    print(f'\n\n## Preprocess Adressa {DATA_LENGTH} Data ##')
    raw_news_set = set()    #total news (cat,body)
    cat_news_set = set()    #cat news cnt
    body_news_set = set()   #body news cnt
    body_filter_news_set = set()    #empty body cnt
    total_news_dic = {}     #history, train, test news cnt
    news_dic = {}           #news info  len(): news cnt
    user_dic = {}           # user info len(): user cnt
    behaviors_list = []     #behaviors  len(): click cnt 
    history_click_cnt = 0           #history click cnt 
    body_filter_cnt = 0    # body empty click cnt
    behaviors_user_filter_cnt= 0    # user_filter click cnt
    total_cnt = 0   #total log cnt
    news_click_cnt = 0  #total news click cnt 
    categorynews_click_cnt = 0      #cat news click cnt
    bodynews_click_cnt = 0          # body news click cnt
    history_click_cnt2 = 0
    body_filter_cnt2 = 0
    reversed_time_filter = 0
    reversed_time_filter2 = 0
    NEWS_DIRECTORY = os.path.join(NEWS_DIR,"Adressa_"+DATA_LENGTH+"(type3)")
    make_folder(NEWS_DIRECTORY)

    preprocess()