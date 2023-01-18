# LANCER: A Lifetime-Aware News Recommender System

## Environment
Our framework were implemented with PyTorch 1.11.0. All experiments were conducted on desktops with 64GB memory, Intel i9-9900K CPU (3.6 GHz, 16M cache), and NVIDIA GeForce RTX 3070.

## Dataset Preprocessing
We conducted experiments on both of two popular real-world datasets, MIND and Adressa.

For MIND, which is a large-scale dataset of English news built from MSN news and anonymized user click logs, we randomly sampled 200K users' click logs then divide the training and test sets, following the previous studies which adopted MIND for their evaluation (i.e., 6 days and 1 day for the training and test sets, respectively). With respect to historical data for constructing the embedding vectors of user and news, we employed it as given (i.e., 4 weeks).

For Adressa, which contains a large number of Norwegian news in conjunction with anonymized users, we took the first 3 weeks of click logs as historical data. Then, the click logs from the 4-th and 5-th week were employed as training and test sets, respectively. We excluded users who have less than 5 clicks for news during a total of 5 weeks.


> ![image](https://user-images.githubusercontent.com/54279688/213063328-47314f63-79bc-4a10-93ec-eb882fbd084e.png)

## Evaluation Metrics
We performed top-*N* recommendations and used the following three measures to evaluate the recommendation accuracy: AUC; MRR; and NDCG. 
First, we employed *Area Under Curve* (AUC) that is computed by:

> ![image](https://user-images.githubusercontent.com/54279688/213065609-67fbdc22-2e58-4178-9e60-634fe81b82f9.png)

where $\mathcal{E}$ denotes a set of test entries; $E$ indicates a test entry that includes a ground-truth news and 20 negative news; ${rank_{gt}}$ indicates the ranking of the ground-truth in $E$ predicted by the trained model. Each test entry consists of 1 positive news and 20 negative news.

Second, we employed *Mean Reciprocal Rank* (MRR) which reflects the average inversed rankings of the ground-truth with the following equation:

> ![image](https://user-images.githubusercontent.com/54279688/213065744-3924537d-062b-4b92-9375-4c4ff07d6177.png)                                     

Finally, we employed *Normalized Discounted Cumulative Gain* (NDCG) to reflect the importance of ranked positions of ground-truth, which is computed as follows:

> ![image](https://user-images.githubusercontent.com/54279688/213065819-e25e3f8f-e291-4df0-a82b-2e660514f8cd.png)
