@REM preprocess adressa log data
python preprocess_raw_data.py  
python data_preprocess.py  
python make_total_news_dataset.py
python sample_test_negative.py
python sample_train_negative.py

@REM python write_new_config.py {training_type} {loss_type} {model} {negative sampling size} {lifetime} {training_data_type} {test type} {test_filter alpha} {history_sampletype} {num}


python write_new_config.py impre CEL LSTUR 4 36 Adressa_5w(type1) 20_ltlive False random 1
python train.py

python write_new_config.py impre CEL NAML 4 36 Adressa_5w(type1) 20_ltlive False random 1
python train.py

python write_new_config.py impre CEL NRMS 4 36 Adressa_5w(type1) 20_ltlive False random 1
python train.py


python write_new_config.py rev_current_log_pop CEL LSTUR 4 36 Adressa_5w(type1) 20_ltlive False random 1
python train.py

python write_new_config.py rev_current_log_pop CEL NAML 4 36 Adressa_5w(type1) 20_ltlive False random 1
python train.py

python write_new_config.py rev_current_log_pop CEL NRMS 4 36 Adressa_5w(type1) 20_ltlive False random 1
python train.py


python write_new_config.py rev_current_log_pop CEL LSTUR 4 36 Adressa_5w(type1) 20_ltlive 0.1 random 1
python train.py

python write_new_config.py rev_current_log_pop CEL NAML 4 36 Adressa_5w(type1) 20_ltlive 0.1 random 1
python train.py

python write_new_config.py rev_current_log_pop CEL NRMS 4 36 Adressa_5w(type1) 20_ltlive 0.1 random 1
python train.py
