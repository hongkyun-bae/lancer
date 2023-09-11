# LANCER: A Lifetime-Aware News Recommender System


## Note
1. Save Adressa log and content data in ./data/raw_data/Adressa (download link: https://reclab.idi.ntnu.no/dataset/) <br/>
2. Save pretrained embedding file (model.txt) in ./data/emb (download link: http://vectors.nlpl.eu/repository/) <br/>
3. Run train_adressa.bat for the experiments displayed in the paper <br/>


## Brief descriptions
1. config.py : each model's configuration file <br/>
2. data_preprocess.py : preprocess data to fit model <br/>
3. dataset.py : load data for training <br/>
4. evaluate.py : evaluate the trained model <br/>
5. make_total_news_dataset.py : make total (including history, train, test periods) news dataset <br/>
6. preprocess_raw_data.py : preprocess adressa log and content data <br/>
7. sample_test_negative.py : sample negative news with options (test period) <br/>
8. sample_train_negative.py : sample negative news with options (train period) <br/>
9. train.py : train and evaluate model <br/>
10. write_new_config.py : write new config.py (read from config/config_Adressa_5w(type1).py) and dataset.py (read from dataset_origin.py) <br/>