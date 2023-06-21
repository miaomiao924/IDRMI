import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score, f1_score
from torch import device
from tqdm import tqdm #产生进度条

import dataloader4kg
from NGCF.utility.batch_test import data_generator, args
from KGCN import KGCN

from NGCF import NGCF
from course_match import Course_match
from user_choice import User_choice
from course_preference import Course_preference

class NG_KGCN(nn.Module):
    def __init__(self, n_user, n_item, norm_adj, args, n_users, n_entitys, n_relations,
                 adj_entity, adj_relation, n_neighbors, e_dim =32,
                 aggregator_method = 'sum',
                 act_method = F.relu, drop_rate=0.5):
        super(NG_KGCN, self).__init__()  #NGCF
        self.args = args
        self.n_user = n_user
        self.n_item = n_item
        self.device = args.device
        self.device = device
        self.emb_size = args.embed_size
        self.batch_size = args.batch_size
        self.node_dropout = args.node_dropout[0]
        self.mess_dropout = args.mess_dropout
        self.batch_size = args.batch_size

        self.norm_adj = norm_adj

        self.layers = eval(args.layer_size)
        self.decay = eval(args.regs)[0]

        #self.init_weight = NGCF.init_weight(self)
        self.e_dim = e_dim  # 特征向量维度
        self.aggregator_method = aggregator_method  # 消息聚合方法
        self.n_neighbors = n_neighbors  # 邻居的数量
        self.user_embedding = nn.Embedding(n_users, e_dim, max_norm=1)
        self.entity_embedding = nn.Embedding(n_entitys, e_dim, max_norm=1)
        self.relation_embedding = nn.Embedding(n_relations, e_dim, max_norm=1)



        self.adj_entity = adj_entity  # 节点的邻接列表
        self.adj_relation = adj_relation  # 关系的邻接列表
        """
        *********************************************************
        Init the weight of user-item.
        """
        self.embedding_dict, self.weight_dict = NGCF.NGCF.init_weight(self)

        """
        *********************************************************
        Get sparse adj.
        """
        #self.sparse_norm_adj = NGCF.NGCF._convert_sp_mat_to_sp_tensor(self,self.norm_adj).to(self.device)
        self.sparse_norm_adj = NGCF.NGCF._convert_sp_mat_to_sp_tensor(self, self.norm_adj).to(args.device)
        #这里不确定加不加 .to(device)
        self.model1 = NGCF.NGCF(data_generator.n_users,
                      data_generator.n_items,
                      self.norm_adj,
                      args).to(args.device)

        self.model2 = KGCN(n_users, n_entitys, n_relations,
                      adj_entity, adj_relation,
                      n_neighbors=n_neighbors, e_dim=32,
                      aggregator_method=aggregator_method,
                      act_method=act_method,
                      drop_rate=drop_rate,
                          ).to(args.device)
        self.model3 = Course_match(data_generator.User
                           ,data_generator.course
        ).to(args.device)
        #
        self.model4 = User_choice(data_generator.course).to(args.device)
        self.model5 = Course_preference(n_users, n_entitys,data_generator.User).to(args.device)


    # def init_weight(self):  #初始化权重
    #     # xavier init
    #     initializer = nn.init.xavier_uniform_
    #
    #     embedding_dict = nn.ParameterDict({
    #         'user_emb': nn.Parameter(initializer(torch.empty(self.n_user,
    #                                              self.emb_size))), #初始化user-embedding
    #         'item_emb': nn.Parameter(initializer(torch.empty(self.n_item,
    #                                             # self.emb_size)))  #初始化item-embedding
    #     })
    #
    #     weight_dict = nn.ParameterDict()
    #     layers = [self.emb_size] + self.layers
    #     for k in range(len(self.layers)):  #更新权重系数
    #         weight_dict.update({'W_gc_%d'%k: nn.Parameter(initializer(torch.empty(layers[k],
    #                                                                   layers[k+1])))})
    #         weight_dict.update({'b_gc_%d'%k: nn.Parameter(initializer(torch.empty(1, layers[k+1])))})
    #
    #         weight_dict.update({'W_bi_%d'%k: nn.Parameter(initializer(torch.empty(layers[k],
    #                                                                   layers[k+1])))})
    #         weight_dict.update({'b_bi_%d'%k: nn.Parameter(initializer(torch.empty(1, layers[k+1])))})
    #
    #     return embedding_dict, weight_dict
    #
    # def _convert_sp_mat_to_sp_tensor(self, X): #稠密矩阵转换为稀疏矩阵
    #     coo = X.tocoo()
    #     i = torch.LongTensor([coo.row, coo.col])
    #     v = torch.from_numpy(coo.data).float()
    #     return torch.sparse.FloatTensor(i, v, coo.shape)
    #
    # def sparse_dropout(self, x, rate, noise_shape):
    #     random_tensor = 1 - rate
    #     random_tensor += torch.rand(noise_shape).to(x.device)
    #     dropout_mask = torch.floor(random_tensor).type(torch.bool)
    #     i = x._indices()
    #     v = x._values()
    #
    #     i = i[:, dropout_mask]
    #     v = v[dropout_mask]
    #
    #     out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
    #     return out * (1. / (1 - rate))


    # def rating(self, u_g_embeddings, pos_i_g_embeddings):
    #     return torch.matmul(u_g_embeddings, pos_i_g_embeddings.t())  #将user-embedding和item-embedding进行一个内积运算，从而预测出u会选择i的概率

    def forward(self, n_user, n_item, n_users,n_entitys,is_evaluate = False):
        #users, pos_items, neg_items = data_generator.sample()  # 1024个 user 1024个正样本  1024个负样本
        #print('user', users)
        #print('pos_items',pos_items)
        #u1_g_embeddings, i1_g_embeddings = self.model1(n_user,n_item,drop_flag=self.args.node_dropout_flag) #NGCF 传入的是用户数量和item数量
        #u1_g_embeddings = self.model1(users,pos_items,neg_items,drop_flag=args.node_dropout_flag)
        u1_g_embeddings = self.model1(n_users, drop_flag=args.node_dropout_flag)
        i2_g_embeddings = self.model2(n_users, n_entitys,is_evaluate = False) #KGCN 传过去的是id



        # out = torch.sigmoid(torch.sum(torch.matmul(u1_g_embeddings, i2_g_embeddings.T).T, axis=-1))
        out1 = self.model3(n_users, n_entitys)

        self.model4.fun(u1_g_embeddings)
        out2 = self.model4(n_users, n_entitys, data_generator.course)


        self.model5.fun(i2_g_embeddings)
        out3 = self.model5(n_users, n_entitys,data_generator.User)


        out = torch.sigmoid(torch.sum(torch.multiply(u1_g_embeddings, i2_g_embeddings), axis=-1))
        # print('out',out)
        # print('out1 课程搭配度', out1)
        # print('out2 用户选择度', out2)
        # print('out3 课程偏爱度', out3)
        #print('点乘out',out)
        result = list(numpy.add(out1, out2))
        result = list(numpy.add(result, out3))
        result = numpy.divide(result,3)
        # result = out1
        #
        xmax = max(result)
        xmin = min(result)



        result = [0.5+(1/(xmax-xmin+0.00001))*(x-xmin) for x in result]

        result = torch.tensor(result)
        result = result.cuda()


        Result = torch.tanh(torch.mul(out, result))


        return Result

    def do_evaluate(self, model: object, testSet: object) -> object:
        dataIter = dataloader4kg.DataIter()
        model.eval()
        P=0.0
        R=0.0
        ACC=0.0
        F1=0.0
        AUC=0.0
        l=len(testSet)//32
        for datas in tqdm(dataIter.iter(testSet, batchSize=32)):

            with torch.no_grad():
                user_ids = datas[:, 0]
                item_ids = datas[:, 1]
                labels = datas[:, 2]

                logits = model(data_generator.n_users,
                    data_generator.n_items,user_ids, item_ids, True)
                predictions = [1 if i >= 0.5 else 0 for i in logits]
                # print('pre',predictions)
                p = precision_score(y_true=labels, y_pred=predictions)
                r = recall_score(y_true=labels, y_pred=predictions)
                acc = accuracy_score(labels, y_pred=predictions)
                auc = roc_auc_score(y_true=labels, y_score=logits.cpu().numpy())

                f1 = f1_score(y_true=labels, y_pred=predictions)
                P += p
                R += r
                ACC += acc
                AUC += auc
                F1 += f1

        return P/l, R/l, ACC/l, F1/l, AUC/l

   # def create_bpr_loss(self, users, pos_items, neg_items):  # 构建损失函数
   #       pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
   #       neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)
   #
   #       maxi = nn.LogSigmoid()(pos_scores - neg_scores) #这一行就是损失函数公式
   #
   #       mf_loss = -1 * torch.mean(maxi)
   #       # cul regularizer
   #       regularizer = (torch.norm(users) ** 2
   #                      + torch.norm(pos_items) ** 2
   #                      + torch.norm(neg_items) ** 2) / 2
   #       emb_loss = self.decay * regularizer / self.batch_size
   #
   #       return mf_loss + emb_loss, mf_loss, emb_loss  #损失函数构建好啦
   #
   #   def rating(self, u_g_embeddings, pos_i_g_embeddings):
   #      return torch.matmul(u_g_embeddings, pos_i_g_embeddings.t())  #将user-embedding和item-embedding进行一个内积运算，从而预测出u会选择i的概率

# def do_evaluate(model, testSet):
#     testSet = torch.LongTensor(testSet)
#     model.eval()
#     with torch.no_grad():
#         user_ids = testSet[:, 0]
#         item_ids = testSet[:, 1]
#         labels = testSet[:, 2]
#         logits = model(data_generator.n_users,
#                 data_generator.n_items,user_ids, item_ids, True)
#         predictions = [1 if i >= 0.5 else 0 for i in logits]
#         p = precision_score(y_true=labels, y_pred=predictions)
#         r = recall_score(y_true=labels, y_pred=predictions)
#         acc = accuracy_score(labels, y_pred=predictions)
#         return p, r, acc
#
#
#
# def train( epochs, batchSize, lr, n_user, n_item, norm_adj, n_users, n_entitys, n_relations,
#       adj_entity, adj_relation,
#       train_set, test_set,
#       n_neighbors,
#       aggregator_method='sum',
#       act_method=F.relu, drop_rate=0.5, weight_decay=5e-4
#       ):
#     model = NG_KGCN(data_generator.n_users,
#                     data_generator.n_items,
#                     norm_adj, n_users, n_entitys, n_relations,
#                     10, adj_entity, adj_relation,
#                     n_neighbors=n_neighbors,
#                     aggregator_method=aggregator_method,
#                     act_method=act_method,
#                     drop_rate=drop_rate
#                     )
#
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
#     loss_fcn = nn.BCELoss()
#     dataIter = dataloader4kg.DataIter()
#     print(len(train_set) // batchSize)
#
#     for epoch in range(epochs):
#         total_loss = 0.0
#         for datas in tqdm(dataIter.iter(train_set, batchSize=batchSize)):
#             user_ids = datas[:, 0]
#             item_ids = datas[:, 1]
#             labels = datas[:, 2]
#             logits = model.forward(data_generator.n_users,
#                                    data_generator.n_items, user_ids, item_ids)
#             loss = loss_fcn(logits, labels.float())
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#         p, r, acc = model.do_evaluate(model, test_set)
#         print("Epoch {} | Loss {:.4f} | Precision {:.4f} | Recall {:.4f} | Accuracy {:.4f} "
#               .format(epoch, total_loss / (len(train_set) // batchSize), p, r, acc))
