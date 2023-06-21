import random
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import device
from tqdm import tqdm #产生进度条
from sklearn.metrics import roc_auc_score,precision_score,recall_score,accuracy_score
import matplotlib.pyplot as plt
import torch.optim as optim
import pandas as pd
import math
from NGCF.utility.batch_test import data_generator, args
import heapq
import torch.nn.functional as F
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from time import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
t0 = time()

#需要注意的是 若计算出推荐度=0 则肯定不推荐，若推荐度>0 则说明相似用户选择了该课程  算出来的选择度都偏小，因为可选的课程太多了 能有一个相同的课程就很不容易惹

class User_choice(nn.Module):
    # def __init__(self,n_users, n_entitys,course):
    def __init__(self,course):
        super(User_choice, self).__init__()
        # self.n_users = n_users
        # self.n_entitys = n_entitys
        self.course = course


    def fun(self, u1_g_embeddings):
        self.users_embeddings = u1_g_embeddings

    def forward(self,n_users,n_entitys,course):
    # def forward(self):

        user_choice_degree = self.get_eur_similar_list(self.users_embeddings,n_users,n_entitys)
        return user_choice_degree
    #
    # def get_cos_similar(self,v1, v2):
    #
    #     num = torch.sum(v1 * v2) #向量点乘
    #
    #     denom1 = math.sqrt(torch.sum(v1 * v1))
    #     denom2 = math.sqrt(torch.sum(v2 * v2))
    #     denom = denom1 * denom2
    #     return (num / denom) if denom != 0 else 0
    #
    #     # 计算一个向量与所有向量的余弦相似度并保存到list
    # def get_cos_similar_list(self,users_embeddings):
    #     similar_list = {}
    #     for user_i in users_embeddings:
    #         similar_i=[]
    #         for user_j in users_embeddings:
    #             cos = self.get_cos_similar(user_i,user_j)  # 计算余弦相似度
    #
    #             similar_i.append(cos)
    #         print('user_i',similar_i)
    #
    #         #然后筛选出top10相似的用户
    #
    #         similar_list[user_i] = similar_i

    def choice(self,course,user,Items):
        user = int(user)
        course = int(course)
        if user not in Items:
            Items[user] = []
        if course in Items[user]:
            return 1
        else:
            return 0

    #计算用户支持度
    def get_user_choice(self,similar_user_index,n_users,n_entitys,users_embeddings):
        #看一下相似的用户是否选择对应的课程
        #l代表的是第几个用户和第几个课程
        Items = self.course
        l=0
        user_choice=[]
        for course in n_entitys:
            p=0
            for idx in similar_user_index[l]:
                if(self.choice(course,n_users[idx],Items)):
                    i=1
                else:
                    i=0
                #需要找到相似用户user的表征,则需要计算user的index
                # user_index = torch.tensor(user)
                # idx = (n_users==user_index).nonzero().flatten()
                #用两者表征计算相似度
                sim=self.get_eur_similar(users_embeddings[l],users_embeddings[idx])
                p=p+sim*i

            p=p/6.0
            p = float(p)
            user_choice.append(p)
            l=l+1

        return user_choice


    # def get_eur_similar(self,v1, v2): #计算欧氏距离
    #     v1 = v1.cpu().detach().numpy()
    #     v2 = v2.cpu().detach().numpy()
    #     dis = np.linalg.norm(v1 - v2)
    #     return 1/(dis + 1.0)

    def get_eur_similar(self, v1, v2):  # 计算余弦相似度

        v1 = v1.cpu().detach().numpy()
        v2 = v2.cpu().detach().numpy()

        a_norm = np.linalg.norm(v1)
        b_norm = np.linalg.norm(v2)
        cos = np.dot(v1, v2) / (a_norm * b_norm)
        return cos


    # 计算一个向量与所有向量的余弦相似度并保存到list
    def get_eur_similar_list(self,users_embeddings,n_users,n_entitys):
        similar_user_index = {}
        u=0
        for user_i in users_embeddings:
            similar_i=[]
            for user_j in users_embeddings:
                cos = self.get_eur_similar(user_i,user_j)  # 计算余弦相似度
                similar_i.append(cos)
            # 然后筛选出top10相似的用户
            # 获取下标，
            top_index = heapq.nlargest(6, range(len(similar_i)), similar_i.__getitem__)
            top_index = torch.tensor(top_index)
            #取得具体的用户
            # n_users=torch.tensor(n_users)
            # top_user_list = torch.index_select(n_users, 0, top_index)

            similar_user_index[u] = top_index
            u = u + 1

        user_choice_degree = self.get_user_choice(similar_user_index,n_users,n_entitys,users_embeddings)

        return user_choice_degree









