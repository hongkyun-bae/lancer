# LANCER: A Lifetime-Aware News Recommender System


## Note
1. Save MIND-large data in ./data/MIND-large(download link: https://msnews.github.io/) <br/>
2. Save pretrained embedding file(glove.840B.300d.txt) in ./data/emb (download link: https://nlp.stanford.edu/projects/glove/) <br/>
3. Run train_MIND.bat for the experiments displayed in the paper <br/>


## Brief descriptions
1. config.py : each model's configuration file <br/>
2. data_preprocess.py : preprocess data to fit model <br/>
3. dataset.py : load preprocessed dataset for training <br/>
4. evaluate.py : evaluate the trained model <br/>
5. sample_MIND.py : sample 200,000 users from MIND-large dataset <br/>
6. sample_test_negative.py : sample negative news with options (test period) <br/>
7. sample_train_negative.py : sample negative news with options (train period) <br/>
8. train.py : train and evaluate model <br/>
9. write_new_config.py : write new config.py (read from config_200k) <br/>