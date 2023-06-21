'''
Created on March 24, 2020

@author: Tinglin Huang (huangtinglin@outlook.com)
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class NGCF(nn.Module):
    def __init__(self, n_user, n_item, norm_adj, args):
        super(NGCF, self).__init__()
        self.n_user = n_user
        self.n_item = n_item

        self.device = args.device
        self.emb_size = args.embed_size
       # print(self.emb_size)
        self.batch_size = args.batch_size
        self.node_dropout = args.node_dropout[0]
        self.mess_dropout = args.mess_dropout
        self.batch_size = args.batch_size

        self.norm_adj = norm_adj

        self.layers = eval(args.layer_size)
        #print(args.layer_size)
        self.decay = eval(args.regs)[0]

        """
        *********************************************************
        Init the weight of user-item.
        """
        self.embedding_dict, self.weight_dict = self.init_weight()

        """
        *********************************************************
        Get sparse adj.
        """
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj).to(self.device)

    def init_weight(self):  #初始化权重
        # xavier init
        initializer = nn.init.xavier_uniform_

        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.n_user,
                                                 self.emb_size))), #初始化user-embedding

            'item_emb': nn.Parameter(initializer(torch.empty(self.n_item,
                                                 self.emb_size)))  #初始化item-embedding
        })

        weight_dict = nn.ParameterDict()
        layers = [self.emb_size] + self.layers
        #print(self.emb_size) #4
        #print(self.layers)   #[4 4 4]
        #print(layers)   #[4 4 4 4]
        for k in range(len(self.layers)):  #初始化权重系数  4层
            weight_dict.update({'W_gc_%d'%k: nn.Parameter(initializer(torch.empty(layers[k],
                                                                      layers[k+1])))})
            weight_dict.update({'b_gc_%d'%k: nn.Parameter(initializer(torch.empty(1, layers[k+1])))})

            #weight_dict.update({'W_bi_%d'%k: nn.Parameter(initializer(torch.empty(layers[k],
                                                                     # layers[k+1])))})
            #weight_dict.update({'b_bi_%d'%k: nn.Parameter(initializer(torch.empty(1, layers[k+1])))})


       # print(embedding_dict['user_emb'].shape)  #[943 64]
        return embedding_dict, weight_dict

    def _convert_sp_mat_to_sp_tensor(self, X): #稠密矩阵转换为稀疏矩阵
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def sparse_dropout(self, x, rate, noise_shape):
        random_tensor = 1 - rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    # def create_bpr_loss(self, users, pos_items, neg_items): #构建损失函数
    #     pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
    #     neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)
    #
    #     maxi = nn.LogSigmoid()(pos_scores - neg_scores) #这一行就是损失函数公式
    #
    #     mf_loss = -1 * torch.mean(maxi)
    #
    #     # cul regularizer
    #     regularizer = (torch.norm(users) ** 2
    #                    + torch.norm(pos_items) ** 2
    #                    + torch.norm(neg_items) ** 2) / 2
    #     emb_loss = self.decay * regularizer / self.batch_size
    #
    #     return mf_loss + emb_loss, mf_loss, emb_loss  #损失函数构建好啦
    #
    # def rating(self, u_g_embeddings, pos_i_g_embeddings):
    #    return torch.matmul(u_g_embeddings, pos_i_g_embeddings.t())  #将user-embedding和item-embedding进行一个内积运算，从而预测出u会选择i的概率

    #def forward(self, users, pos_items, neg_items, drop_flag=True):
    #def forward(self, _users, _items, drop_flag=True):
    def forward(self, users,  drop_flag = True):
        #print('user',_users)
        A_hat = self.sparse_dropout(self.sparse_norm_adj, #构造拉普拉斯矩阵
                                    self.node_dropout,
                                    self.sparse_norm_adj._nnz()) if drop_flag else self.sparse_norm_adj

        ego_embeddings = torch.cat([self.embedding_dict['user_emb'],
                                    self.embedding_dict['item_emb']], 0)   #第一步将user-embedding和item-embedding拼接在一起

        all_embeddings = [ego_embeddings]

        for k in range(len(self.layers)):
            side_embeddings = torch.sparse.mm(A_hat, ego_embeddings) #A.hat是拉普阿斯矩阵，这个是L*E

            # transformed sum messages of neighbors.
            #sum_embeddings = torch.matmul(side_embeddings, self.weight_dict['W_gc_%d' % k]) \
                                            # + self.weight_dict['b_gc_%d' % k]     #这就是展开的公式的前半截 L*E*W

            # bi messages of neighbors.
            # element-wise product
            #bi_embeddings = torch.mul(ego_embeddings, side_embeddings)  #L*E@E
            # transformed bi messages of neighbors.
            ego_embeddings = torch.matmul(side_embeddings, self.weight_dict['W_gc_%d' % k]) \
                                            + self.weight_dict['b_gc_%d' % k]   #这就是展开的公式的后半部分 L*E*W

            # non-linear activation.
            ego_embeddings = nn.LeakyReLU(negative_slope=0.2)(ego_embeddings)
           # ego_embeddings = nn.LeakyReLU(negative_slope=0.2)(side_embeddings)
            #加起来进行一个激活函数 这就得到了新一层的embedding
           # ego_embeddings = side_embeddings + ego_embeddings
            #ego_embeddings = nn.LeakyReLU(negative_slope=0.2)(ego_embeddings)
            #ego_embeddings = side_embeddings
            # message dropout.  dropout防止过拟合 以mess_dropout[k]的概率 使得embedding置为0
            #print('k',k)
            ego_embeddings = nn.Dropout(self.mess_dropout[k])(ego_embeddings)

            # normalize the distribution of embeddings.
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)    #进行规范化

            all_embeddings += [norm_embeddings]    #将每一层的embedding都存在all-embedding里面

       # all_embeddings = torch.cat(all_embeddings, 1)     #for循环执行完之后，将每一层的enbedding进行一个拼接
       # print(all_embeddings[0])
        #print(all_embeddings[1])
        #print(all_embeddings[2])
        #print(all_embeddings[3])
        #print(all_embeddings[4])
        all_embeddings = all_embeddings[0]+all_embeddings[1]+all_embeddings[2]+all_embeddings[3]#拼接变成相加


        u_g_embeddings = all_embeddings[:self.n_user, :]   #把原来合在一起的user-embedding和item-embedding分开
        i_g_embeddings = all_embeddings[self.n_user:, :]   #把原来合在一起的user-embedding和item-embedding分开

        """
        *********************************************************
        look up.
        """
        u_g_embeddings = u_g_embeddings[users, :]          #随机筛选出了1024个用户的样本
       # pos_i_g_embeddings = i_g_embeddings[pos_items, :]  #筛选出了1024个用户的正样本
       # neg_i_g_embeddings = i_g_embeddings[neg_items, :]  #筛选出了1024个用户的负样本

        #print('u',u_g_embeddings.shape)
        #return u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings  #然后返回 目的就是为了计算损失函数 从而优化参数
        #print(u_g_embeddings.shape)
        #print(u_g_embeddings.shape)

        return u_g_embeddings   # 返回 user-embedding

'''
Created on March 24, 2020

@author: Tinglin Huang (huangtinglin@outlook.com)
'''
#原NGCF
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class NGCF(nn.Module):
#     def __init__(self, n_user, n_item, norm_adj, args):
#         super(NGCF, self).__init__()
#         self.n_user = n_user
#         self.n_item = n_item
#         self.device = args.device
#         self.emb_size = args.embed_size
#        # print(self.emb_size)
#         self.batch_size = args.batch_size
#         self.node_dropout = args.node_dropout[0]
#         self.mess_dropout = args.mess_dropout
#         self.batch_size = args.batch_size
#
#         self.norm_adj = norm_adj
#
#         self.layers = eval(args.layer_size)
#         self.decay = eval(args.regs)[0]
#
#         """
#         *********************************************************
#         Init the weight of user-item.
#         """
#         self.embedding_dict, self.weight_dict = self.init_weight()
#
#         """
#         *********************************************************
#         Get sparse adj.
#         """
#         self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj).to(self.device)
#
#     def init_weight(self):  #初始化权重
#         # xavier init
#         initializer = nn.init.xavier_uniform_
#
#         embedding_dict = nn.ParameterDict({
#             'user_emb': nn.Parameter(initializer(torch.empty(self.n_user,
#                                                  self.emb_size))), #初始化user-embedding
#
#             'item_emb': nn.Parameter(initializer(torch.empty(self.n_item,
#                                                  self.emb_size)))  #初始化item-embedding
#         })
#
#         weight_dict = nn.ParameterDict()
#         layers = [self.emb_size] + self.layers
#         #print(self.emb_size) #4
#         #print(self.layers)   #[4 4 4]
#         #print(layers)   #[4 4 4 4]
#         for k in range(len(self.layers)):  #初始化权重系数  4层
#             weight_dict.update({'W_gc_%d'%k: nn.Parameter(initializer(torch.empty(layers[k],
#                                                                       layers[k+1])))})
#             weight_dict.update({'b_gc_%d'%k: nn.Parameter(initializer(torch.empty(1, layers[k+1])))})
#
#             weight_dict.update({'W_bi_%d'%k: nn.Parameter(initializer(torch.empty(layers[k],
#                                                                       layers[k+1])))})
#             weight_dict.update({'b_bi_%d'%k: nn.Parameter(initializer(torch.empty(1, layers[k+1])))})
#
#
#        # print(embedding_dict['user_emb'].shape)  #[943 64]
#         return embedding_dict, weight_dict
#
#     def _convert_sp_mat_to_sp_tensor(self, X): #稠密矩阵转换为稀疏矩阵
#         coo = X.tocoo()
#         i = torch.LongTensor([coo.row, coo.col])
#         v = torch.from_numpy(coo.data).float()
#         return torch.sparse.FloatTensor(i, v, coo.shape)
#
#     def sparse_dropout(self, x, rate, noise_shape):
#         random_tensor = 1 - rate
#         random_tensor += torch.rand(noise_shape).to(x.device)
#         dropout_mask = torch.floor(random_tensor).type(torch.bool)
#         i = x._indices()
#         v = x._values()
#
#         i = i[:, dropout_mask]
#         v = v[dropout_mask]
#
#         out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
#         return out * (1. / (1 - rate))
#
#     # def create_bpr_loss(self, users, pos_items, neg_items): #构建损失函数
#     #     pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
#     #     neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)
#     #
#     #     maxi = nn.LogSigmoid()(pos_scores - neg_scores) #这一行就是损失函数公式
#     #
#     #     mf_loss = -1 * torch.mean(maxi)
#     #
#     #     # cul regularizer
#     #     regularizer = (torch.norm(users) ** 2
#     #                    + torch.norm(pos_items) ** 2
#     #                    + torch.norm(neg_items) ** 2) / 2
#     #     emb_loss = self.decay * regularizer / self.batch_size
#     #
#     #     return mf_loss + emb_loss, mf_loss, emb_loss  #损失函数构建好啦
#     #
#     # def rating(self, u_g_embeddings, pos_i_g_embeddings):
#     #    return torch.matmul(u_g_embeddings, pos_i_g_embeddings.t())  #将user-embedding和item-embedding进行一个内积运算，从而预测出u会选择i的概率
#
#     #def forward(self, users, pos_items, neg_items, drop_flag=True):
#     #def forward(self, _users, _items, drop_flag=True):
#     def forward(self, users,  drop_flag = True):
#         #print('user',_users)
#         A_hat = self.sparse_dropout(self.sparse_norm_adj, #构造拉普拉斯矩阵
#                                     self.node_dropout,
#                                     self.sparse_norm_adj._nnz()) if drop_flag else self.sparse_norm_adj
#
#         ego_embeddings = torch.cat([self.embedding_dict['user_emb'],
#                                     self.embedding_dict['item_emb']], 0)   #第一步将user-embedding和item-embedding拼接在一起
#
#         all_embeddings = [ego_embeddings]
#
#         for k in range(len(self.layers)):
#             side_embeddings = torch.sparse.mm(A_hat, ego_embeddings) #A.hat是拉普阿斯矩阵，这个是L*E
#
#             # transformed sum messages of neighbors.
#             sum_embeddings = torch.matmul(side_embeddings, self.weight_dict['W_gc_%d' % k]) \
#                                              + self.weight_dict['b_gc_%d' % k]     #这就是展开的公式的前半截 L*E*W
#
#             # bi messages of neighbors.
#             # element-wise product
#             bi_embeddings = torch.mul(ego_embeddings, side_embeddings)  #L*E@E
#             # transformed bi messages of neighbors.
#             bi_embeddings = torch.matmul(bi_embeddings, self.weight_dict['W_bi_%d' % k]) \
#                                             + self.weight_dict['b_bi_%d' % k]   #这就是展开的公式的后半部分 L*E@E*W
#
#             # non-linear activation.
#             ego_embeddings = nn.LeakyReLU(negative_slope=0.2)(sum_embeddings + bi_embeddings)
#             #加起来进行一个激活函数 这就得到了新一层的embedding
#
#
#             # message dropout.  dropout防止过拟合 以mess_dropout[k]的概率 使得embedding置为0
#             ego_embeddings = nn.Dropout(self.mess_dropout[k])(ego_embeddings)
#
#             # normalize the distribution of embeddings.
#             norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)    #进行规范化
#
#             all_embeddings += [norm_embeddings]    #将每一层的embedding都存在all-embedding里面
#
#
#         #all_embeddings = torch.cat(all_embeddings, 1)     #for循环执行完之后，将每一层的enbedding进行一个拼接
#         all_embeddings = all_embeddings[0] + all_embeddings[1] + all_embeddings[2] + all_embeddings[3]  # 拼接变成相加
#         u_g_embeddings = all_embeddings[:self.n_user, :]   #把原来合在一起的user-embedding和item-embedding分开
#         i_g_embeddings = all_embeddings[self.n_user:, :]   #把原来合在一起的user-embedding和item-embedding分开
#
#         """
#         *********************************************************
#         look up.
#         """
#         u_g_embeddings = u_g_embeddings[users, :]          #随机筛选出了1024个用户的样本
#        # pos_i_g_embeddings = i_g_embeddings[pos_items, :]  #筛选出了1024个用户的正样本
#        # neg_i_g_embeddings = i_g_embeddings[neg_items, :]  #筛选出了1024个用户的负样本
#
#        # print('u',u_g_embeddings.shape)
#         #return u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings  #然后返回 目的就是为了计算损失函数 从而优化参数
#       #  print(u_g_embeddings.shape)
#        # print(i_g_embeddings.shape)
#         return u_g_embeddings   # 返回 user-embedding