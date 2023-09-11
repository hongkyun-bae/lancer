
import os
import pandas as pd
preprocess_data_folder = './data/preprocessed_data'

data_folder_list = ['history','train','test']
print(os.listdir(preprocess_data_folder))

for data_type in os.listdir(preprocess_data_folder):
    if data_type.startswith('Adressa'):

        df_list = []
        raw_df_list = []
        for data_folder in data_folder_list:
            data_folder = os.path.join(preprocess_data_folder,data_type,data_folder)
            df_tmp = pd.read_table(os.path.join(data_folder, 'news.tsv'),header=None, names=['id', 'category', 'subcategory', 'title', 'abstract','raw_id','publish_time','click_time'])
            raw_df_tmp = pd.read_table(os.path.join(data_folder, 'news(raw).tsv'),header=None, names=['id', 'category', 'subcategory', 'title', 'abstract','raw_id','publish_time','click_time'])
            df_list.append(df_tmp)
            raw_df_list.append(raw_df_tmp)
        df = pd.concat(df_list)
        raw_df = pd.concat(raw_df_list)

        df_tmp = df.groupby(['id'])['click_time'].apply(lambda x: ','.join(x))
        raw_df_tmp = raw_df.groupby(['id'])['click_time'].apply(lambda x: ','.join(x))
        df.drop_duplicates(['id'],keep='first',inplace=True)
        raw_df.drop_duplicates(['id'],keep='first',inplace=True)


        df.set_index('id',inplace=True,drop=True)
        raw_df.set_index('id',inplace=True,drop=True)

        del df['click_time']
        del raw_df['click_time']
        df = df.merge(df_tmp,left_index=True,right_index=True)
        raw_df = raw_df.merge(raw_df_tmp,left_index=True,right_index=True)

        df.to_csv(os.path.join(preprocess_data_folder,data_type,'total_news.tsv'),sep='\t')
        raw_df.to_csv(os.path.join(preprocess_data_folder,data_type,'total_news(raw).tsv'),sep='\t')


