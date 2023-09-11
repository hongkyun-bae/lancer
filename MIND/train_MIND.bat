@REM preprocess MIND dataset
python sample_MIND.py
python write_config.py NRMS 200k random_4 lt36_K20 0.1 1
python data_preprocess.py
python sample_test_negative.py
python sample_train_negative.py

@REM python write_config.py {model} {dataset} {trainingType_negativeSampleRatio} {test_type} {num}
python write_config.py NAML 200k recent_4 lt36_K20 0 1
python train.py
python evaluate.py

python write_config.py NRMS 200k recent_4 lt36_K20 0 1
python train.py
python evaluate.py

python write_config.py LSTUR 200k recent_4 lt36_K20 0 1
python train.py
python evaluate.py

python write_config.py NAML 200k rev_current_log_ltpop_4 lt36_K20 0 2
python train.py
python evaluate.py
python write_config.py NAML 200k rev_current_log_ltpop_4 lt36_K20 0.1 2
python evaluate.py

python write_config.py NRMS 200k rev_current_log_ltpop_4 lt36_K20 0 2
python train.py
python evaluate.py
python write_config.py NRMS 200k rev_current_log_ltpop_4 lt36_K20 0.1 2
python evaluate.py

python write_config.py LSTUR 200k rev_current_log_ltpop_4 lt36_K20 0 2
python train.py
python evaluate.py
python write_config.py LSTUR 200k rev_current_log_ltpop_4 lt36_K20 0.1 2
python evaluate.py

