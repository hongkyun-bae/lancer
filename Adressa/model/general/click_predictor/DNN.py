import torch
import torch.nn as nn
from math import sqrt
import torch.nn.functional as F

class DNNClickPredictor(torch.nn.Module):
    def __init__(self, input_size, hidden_size=None):
        super(DNNClickPredictor, self).__init__()
        if hidden_size is None:
            # TODO: is sqrt(input_size) a good default value?
            # print(input_size)
            hidden_size = int(sqrt(input_size))
        self.dnn = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, candidate_news_vector, user_vector):
        """
        Args:
            candidate_news_vector: batch_size, X
            user_vector: batch_size, X
        Returns:
            (shape): batch_size
        """

        user_vector = user_vector.squeeze()
        candidate_news_vector = candidate_news_vector.squeeze()
            # print(torch.cat((candidate_news_vector, user_vector),dim=1).shape)
            # print(self.dnn(torch.cat((candidate_news_vector, user_vector),dim=1)).shape)
        return self.dnn(torch.cat((candidate_news_vector, user_vector),
                                  dim=1)).squeeze(dim=1)
        # except:
        #     print()
        #     print("DNN exception")
        #     print(user_vector.shape)
        #     print(candidate_news_vector.shape)
        #     return self.dnn(torch.cat((candidate_news_vector, user_vector),
        #                               dim=2).squeeze()).squeeze(dim=1)
