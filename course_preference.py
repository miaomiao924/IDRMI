import random
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

class Course_preference(nn.Module):
    def __init__(self,n_users, n_entitys,User):
        super(Course_preference, self).__init__()
        self.n_users = n_users
        self.n_entitys = n_entitys
        self.User=User

    def fun(self, i2_g_embeddings):
        self.items_embeddings = i2_g_embeddings

    def forward(self,n_users,n_entitys,User):

        course_preference_degree = self.get_eur_similar_list(self.items_embeddings,n_users,n_entitys)
        return course_preference_degree
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


    def choice(self,user,course,Userss):
        user = int(user)
        course = int(course)
        if course not in Userss:
            Userss[course] = []
        if user in Userss[course]:
            return 1
        else:
            return 0

    #计算课程需求度
    def get_course_preference(self,similar_item_index,n_users,n_entitys,items_embeddings):
        #看一下相似的用户是否选择对应的课程
        #l代表的是第几个用户和第几个课程
        Userss = self.User
        l=0
        courses_preference=[]
        for user in n_users:
            p=0
            for idx in similar_item_index[l]:
                if(self.choice(user,n_entitys[idx],Userss)):
                    i = 1
                else:
                    i = 0
                #需要找到相似项目course的表征,则需要计算course的index

                # course_index = torch.tensor(course)
                # idx = (n_entitys==course_index).nonzero().flatten()
                #用两者表征计算相似度
                sim=self.get_eur_similar(items_embeddings[l],items_embeddings[idx])
                p=p+sim*i


            p=p/3.0
            p = float(p)
            courses_preference.append(p)
            l=l+1
        # print(courses_preference)
        return courses_preference



    def get_eur_similar(self,v1, v2): #计算欧氏距离
        v1 = v1.cpu().detach().numpy()
        v2 = v2.cpu().detach().numpy()
        dis = np.linalg.norm(v1-v2)
        return 1 / (dis + 1.0)


        # 计算一个向量与所有向量的余弦相似度并保存到list
    def get_eur_similar_list(self,items_embeddings,n_users,n_entitys):
        similar_item_index = {}
        v=0
        for item_i in items_embeddings:
            similar_i=[]
            for item_j in items_embeddings:
                cos = self.get_eur_similar(item_i,item_j)  # 计算余弦相似度
                if cos == 1.0:
                    similar_i.append(0.0)
                else:
                    similar_i.append(cos)

            # 然后筛选出topk相似的课程
            # 获取下标，
            top_index = heapq.nlargest(3, range(len(similar_i)), similar_i.__getitem__)
            top_index = torch.tensor(top_index)
            #获取具体的课程

            similar_item_index[v] = top_index
            v = v + 1

        course_preference_degree=self.get_course_preference(similar_item_index,n_users,n_entitys,items_embeddings)
        # print(course_preference_degree)
        return course_preference_degree










