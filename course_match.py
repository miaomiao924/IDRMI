import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import device
from tqdm import tqdm #产生进度条
from sklearn.metrics import roc_auc_score,precision_score,recall_score,accuracy_score
import matplotlib.pyplot as plt
import torch.optim as optim
from NGCF.utility.batch_test import data_generator, args
import warnings
warnings.filterwarnings('ignore')
from time import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
t0 = time()

class Course_match(nn.Module):
    def __init__(self,User,Course):
        super(Course_match, self).__init__()
        self.User = User
        self.Course = Course


    def forward(self, n_users, n_items):  #将32个user_id和32个item_id传入

        #第一步 从用户u的课程集合里面随机选择k个课程
        course_ids = self.getText(n_users,n_items)
        #第二步，计算待推荐课程和已有的k个课程的平均搭配度数
        match = self.getMatch(course_ids,n_items.cuda())
        return match


    def getText(self,n_users,n_items):  #第一步

        course_ids=[]
        course = self.Course
        l=0
        for user in n_users:
            user = int(user)
            entry = list(set(course[user]))
            if n_items[l] in entry:
                entry.remove(n_items[l])
            course_ids.append(entry)
            l=l+1

        print('cid',course_ids)
        return course_ids

    def getMatch(self,course_ids,n_items):
        l=0
        match=[]
        for c_list in course_ids:
            m = self.get_match(c_list, n_items[l])
            match.append(m)
            l = l + 1
        return match

    # 计算平均搭配度
    def get_match(self,c_list,c):
        m=0
        l=len(c_list)
        for course in c_list:
            course=int(course)
            c=int(c)
            q1,q2 = self.calculation(course,c)
            m += (q1+q2)/2


        if l==0:
            return 0
        else:
            return m/l

    def calculation(self,course,c):
        User = self.User
        if c not in User:
            User[c] = []
        N_course=len(User[course])
        N_c = len(User[c])

        set1 = set(User[course])
        set2 = set(User[c])
        # print('cou',course)
        # print('c',c)
        tmp = set1&set2

        q1=len(tmp)/(N_course+0.0001)
        q2=len(tmp)/(N_c+0.0001)

        return q1,q2










