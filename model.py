#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# AUTHOR: Ryan Hu
# DATE: 2021/03/19 Fri
# TIME: 19:12:23
# DESCRIPTION:
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchtext


class WordAvg(nn.Module):
    """
    sequence每个token的embedding求平均
    送到MLP输出分类结果
    """
    def __init__(self, num_embeddings, embedding_dim, in_features, num_class):
        super(WordAvg, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=num_embeddings,
                                        embedding_dim=embedding_dim)

        self.linear = nn.Sequential(nn.Linear(embedding_dim*2, in_features),
                                   nn.ReLU(),
                                    nn.Linear(in_features, in_features),
                                    nn.ReLU(),
                                    nn.Linear(in_features, in_features),
                                    nn.ReLU())

        self.predict = nn.Linear(in_features=in_features,
                                    out_features=num_class)

    def forward(self, premise, hypothesis):
        premise_emb = torch.mean(self.embedding(premise), dim=1)
        hypothesis_emb = torch.mean(self.embedding(hypothesis), dim=1)
        x = torch.cat((premise_emb, hypothesis_emb), dim=-1)
        x = self.linear(x)
        x = self.predict(x)
        return x


class BiLSTM(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_size, num_class):
        super(BiLSTM, self).__init__()

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.bilstm = nn.LSTM(input_size=embedding_dim, 
                                hidden_size=hidden_size,
                                num_layers=2,
                                batch_first=True,
                                bidirectional=True)
        
        self.linear = nn.Sequential(nn.Linear(hidden_size*2*2, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, hidden_size),
                                    nn.ReLU())
        
        self.predict = nn.Linear(hidden_size, num_class)
    
    def forward(self, premise, hypothesis):
        premise_emb = self.embedding(premise)
        hypothesis_emb = self.embedding(hypothesis)
        p_h, _ = self.bilstm(premise_emb)
        h_h, _ = self.bilstm(hypothesis_emb)
        p_h = torch.mean(p_h, dim=1)
        h_h = torch.mean(h_h, dim=1)
        
        x = torch.cat((p_h, h_h), dim=-1)
        x = self.linear(x)
        x = self.predict(x)
        return x