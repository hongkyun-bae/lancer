# LANCER: A Lifetime-Aware News Recommender System

## Environment
Our framework were implemented with PyTorch 1.11.0. All experiments were conducted on desktops with 64GB memory, Intel i9-9900K CPU (3.6 GHz, 16M cache), and NVIDIA GeForce RTX 3070.

## Dataset Preprocessing
We conducted experiments on both of two popular real-world datasets, MIND and Adressa.

For MIND, which is a large-scale dataset of English news built from MSN news and anonymized user click logs, we randomly sampled 200K users' click logs then divide the training and test sets, following the previous studies which adopted MIND for their evaluation (i.e., 6 days and 1 day for the training and test sets, respectively). With respect to historical data for constructing the embedding vectors of user and news, we employed it as given (i.e., 4 weeks).

For Adressa, which contains a large number of Norwegian news in conjunction with anonymized users, we took the first 3 weeks of click logs as historical data. Then, the click logs from the 4-th and 5-th week were employed as training and test sets, respectively. We excluded users who have less than 5 clicks for news during a total of 5 weeks.


> ![image](https://user-images.githubusercontent.com/54279688/213063328-47314f63-79bc-4a10-93ec-eb882fbd084e.png)
