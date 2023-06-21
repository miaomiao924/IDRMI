# import torch
# import torch.optim as optim
#
# from NGCF import NGCF
# from KGCN import KGCN
# from utility.helper import *
# from utility.batch_test import *
#
# import warnings
# warnings.filterwarnings('ignore')
# from time import time
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#
#     def train( epochs, batchSize, lr,
#            n_users, n_entitys, n_relations,
#            adj_entity, adj_relation,
#            train_set, test_set,
#            n_neighbors,
#            aggregator_method = 'sum',
#            act_method = F.relu, drop_rate = 0.5, weight_decay=5e-4
#          ):
#
#     def create_bpr_loss(self, users, pos_items, neg_items): #构建损失函数
#         pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
#         neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)
#
#         maxi = nn.LogSigmoid()(pos_scores - neg_scores) #这一行就是损失函数公式
#
#         mf_loss = -1 * torch.mean(maxi)
#
#        # cul regularizer
#         regularizer = (torch.norm(users) ** 2
#                      + torch.norm(pos_items) ** 2
#                     + torch.norm(neg_items) ** 2) / 2
#         emb_loss = self.decay * regularizer / self.batch_size
#
#         return mf_loss + emb_loss, mf_loss, emb_loss  #损失函数构建好啦
#
#     def rating(self, u_g_embeddings, pos_i_g_embeddings):
#         return torch.matmul(u_g_embeddings, pos_i_g_embeddings.t())  #将user-embedding和item-embedding进行一个内积运算，从而预测出u会选择i的概率
#
#     model1 = NGCF(data_generator.n_users,
#                  data_generator.n_items,
#                  norm_adj,
#                  args).to(args.device)
#
#     model2 = KGCN(n_users, n_entitys, n_relations,
#                  10, adj_entity, adj_relation,
#                  n_neighbors = n_neighbors,
#                  aggregator_method = aggregator_method,
#                  act_method = act_method,
#                  drop_rate = drop_rate)
#
#     #将user-embedding与item-embedding耦合一下
#
#
#     t0 = time()
#     """
#     *********************************************************
#     Train.
#     """
#     cur_best_pre_0, stopping_step = 0, 0
#     optimizer = optim.Adam(model.parameters(), lr=args.lr)
#
#     loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
#     for epoch in range(args.epoch):
#         t1 = time()
#         loss, mf_loss, emb_loss = 0., 0., 0.
#         n_batch = data_generator.n_train // args.batch_size + 1
#
#         for idx in range(n_batch):
#             users, pos_items, neg_items = data_generator.sample()
#             #模型调用不对
#             u_g_embeddings = model1(users, pos_items,neg_items,drop_flag=args.node_dropout_flag)
#
#             i_g_embeddings = model2(n_users, n_entitys, n_relations,
#                           10, adj_entity, adj_relation,
#                           n_neighbors=n_neighbors,
#                           aggregator_method=aggregator_method,
#                           act_method=act_method,
#                           drop_rate=drop_rate)
#
#
#             pos_i_g_embeddings = i_g_embeddings[pos_items, :]  # 筛选出了该用户的正样本
#             neg_i_g_embeddings = i_g_embeddings[neg_items, :]  # 筛选出了该用户的负样本
#
#             # 耦合一下  代码还没找到
#
#             #做内积
#             rating(u_g_embeddings,i_g_embeddings)
#
#
#             #损失函数还要改
#             batch_loss, batch_mf_loss, batch_emb_loss = create_bpr_loss(u_g_embeddings,
#                                                                               pos_i_g_embeddings,
#                                                                               neg_i_g_embeddings)
#             optimizer.zero_grad()
#             batch_loss.backward()
#             optimizer.step()
#
#             loss += batch_loss
#             mf_loss += batch_mf_loss
#             emb_loss += batch_emb_loss
#
#         if (epoch + 1) % 100 != 0:
#             if args.verbose > 0 and epoch % args.verbose == 0:
#                 perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
#                     epoch, time() - t1, loss, mf_loss, emb_loss)
#                 print(perf_str)
#             continue
#
#         t2 = time()
#         users_to_test = list(data_generator.test_set.keys())
#         #这点需要改
#         ret = test(model, users_to_test, drop_flag=False)
#
#         t3 = time()
#
#         loss_loger.append(loss)
#         rec_loger.append(ret['recall'])
#         pre_loger.append(ret['precision'])
#         ndcg_loger.append(ret['ndcg'])
#         hit_loger.append(ret['hit_ratio'])
#
#         if args.verbose > 0:
#             perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f], recall=[%.5f, %.5f], ' \
#                        'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
#                        (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, ret['recall'][0], ret['recall'][-1],
#                         ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
#                         ret['ndcg'][0], ret['ndcg'][-1])
#             print(perf_str)
#
#         cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
#                                                                     stopping_step, expected_order='acc', flag_step=5)
#
#         # *********************************************************
#         # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
#         if should_stop == True:
#             break
#
#         # *********************************************************
#         # save the user & item embeddings for pretraining.
#         if ret['recall'][0] == cur_best_pre_0 and args.save_flag == 1:
#             torch.save(model.state_dict(), args.weights_path + str(epoch) + '.pkl')
#             print('save the weights in path: ', args.weights_path + str(epoch) + '.pkl')
#
#     recs = np.array(rec_loger)
#     pres = np.array(pre_loger)
#     ndcgs = np.array(ndcg_loger)
#     hit = np.array(hit_loger)
#
#     best_rec_0 = max(recs[:, 0])
#     idx = list(recs[:, 0]).index(best_rec_0)
#
#     final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
#                  (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
#                   '\t'.join(['%.5f' % r for r in pres[idx]]),
#                   '\t'.join(['%.5f' % r for r in hit[idx]]),
#                   '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
#     print(final_perf)
#
#
# if __name__ == '__main__':
#     print(torch.cuda.is_available())
#     args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  #用GPU跑数据
#
#     #这是NGCF的初始化
#     plain_adj, norm_adj, mean_adj = data_generator.get_adj_mat()  # 得到邻接矩阵 在load_data第83行
#     # ().to(args.device)
#     args.node_dropout = eval(args.node_dropout)
#     args.mess_dropout = eval(args.mess_dropout)
#     n_neighbors = 10
#
#     #这是KGCN的初始化
#     users, items, train_set, test_set = dataloader4kg.readRecData( dataloader4kg.Ml_100K.RATING )
#     entitys, relations, kgTriples = dataloader4kg.readKgData( dataloader4kg.Ml_100K.KG )
#     adj_kg = dataloader4kg.construct_kg( kgTriples )
#     adj_entity, adj_relation = dataloader4kg.construct_adj( n_neighbors, adj_kg, len( entitys ) )
#
#     train(epochs=1000, batchSize=1024, lr=0.01,
#           n_users=max(users) + 1, n_entitys=max(entitys) + 1,
#           n_relations=max(relations) + 1, adj_entity=adj_entity,
#           adj_relation=adj_relation, train_set=train_set,
#           test_set=test_set, n_neighbors=n_neighbors,
#           aggregator_method='sum', act_method=F.relu, drop_rate=0.5)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import device
from tqdm import tqdm #产生进度条
from sklearn.metrics import roc_auc_score,precision_score,recall_score,accuracy_score
import matplotlib.pyplot as plt


import torch.optim as optim
from NGCF.utility import helper
from NGCF.utility.helper import *
from NGCF.utility.batch_test import *
from NGCF import NG_KGCN
from NGCF.NG_KGCN import NG_KGCN

import warnings
import dataloader4kg
from NGCF.utility.batch_test import data_generator
from KGCN import KGCN
from NGCF import NGCF

warnings.filterwarnings('ignore')
from time import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
t0 = time()


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



def train( epochs, batchSize, lr, n_user, n_item, norm_adj, n_users, n_entitys, n_relations,
      adj_entity, adj_relation,
      train_set,eval_set,test_set,
      n_neighbors,
      aggregator_method='sum',
      act_method=F.relu, drop_rate=0, weight_decay=5e-4,
      ):
    model = NG_KGCN(data_generator.n_users,
                    data_generator.n_items,
                    norm_adj, args, n_users, n_entitys, n_relations,
                    adj_entity, adj_relation,
                    n_neighbors=n_neighbors, e_dim=32,
                    aggregator_method=aggregator_method,
                    act_method=act_method,
                    drop_rate=drop_rate).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fcn = nn.BCELoss()
    dataIter = dataloader4kg.DataIter()
    print(len(train_set) // batchSize)

    for epoch in range(epochs):
        total_loss = 0.0
        for datas in tqdm(dataIter.iter(train_set, batchSize=batchSize)):
            user_ids = datas[:, 0]
            item_ids = datas[:, 1]

            labels = torch.tensor(datas[:, 2]).cuda()
            logits = model.forward(data_generator.n_users,
                                   data_generator.n_items, user_ids, item_ids)


            loss = loss_fcn(logits.double(), labels.double())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()


       #print(test_set)
        p_eval, r_eval, acc_eval, f1_eval, auc_eval = model.do_evaluate(model, eval_set)  # 验证集
        p_test, r_test, acc_test, f1_test ,auc_test = model.do_evaluate(model,test_set)  #测试集

        print("Epoch {} |Eval: Loss {:.4f} | Precision {:.4f} | Recall {:.4f} | Accuracy {:.4f}  | F1 {:.4f} | AUC {:.4f}| Test: Precision {:.4f} | Recall {:.4f} | Accuracy {:.4f}  | F1 {:.4f} | AUC {:.4f}"
              .format(epoch, total_loss / (len(train_set) // batchSize),p_eval, r_eval, acc_eval, f1_eval, auc_eval, p_test, r_test, acc_test, f1_test ,auc_test))

        loss_f.append(total_loss / (len(train_set) // batchSize))
        pre.append(p_test)
        recall.append(r_test)
        acc1.append(acc_test)
        AUC.append(auc_test)
        F1.append(f1_test)

    print('best_loss', min(loss_f))
    print('best_pre', max(pre))
    print('best_recall', max(recall))
    print('best_acc', max(acc1))
    print('best_f1', max(F1))
    print('best_auc', max(AUC))



if __name__ == '__main__':
    print(torch.cuda.is_available())
    args.device:device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 用GPU跑数据
    print(device.type)
    plain_adj, norm_adj, mean_adj = data_generator.get_adj_mat()  # 得到邻接矩阵 在load_data第83行

    args.node_dropout = eval(args.node_dropout)
    args.mess_dropout = eval(args.mess_dropout)

    # 这是KGCN的初始化
    n_neighbors = 8

    users, items, train_set,eval_set,test_set = dataloader4kg.readRecData(dataloader4kg.Ml_100K.RATING)
    entitys, relations, kgTriples = dataloader4kg.readKgData(dataloader4kg.Ml_100K.KG)
    #print('entity',entitys)
    adj_kg = dataloader4kg.construct_kg(kgTriples)
    adj_entity, adj_relation = dataloader4kg.construct_adj(n_neighbors, adj_kg, len(entitys))

    loss_f = []
    pre = []
    recall = []
    acc1 = []
    AUC = []
    F1 = []


    train(epochs=10, batchSize=32, lr=5e-6,n_user=data_generator.n_users,
          n_item=data_generator.n_items, norm_adj=norm_adj,
          n_users=max(users) + 1, n_entitys=max(entitys) + 1,
          n_relations=max(relations) + 1, adj_entity=adj_entity,
          adj_relation=adj_relation, train_set=train_set,eval_set=eval_set,
          test_set=test_set, n_neighbors=n_neighbors,
          aggregator_method='sum', act_method=F.relu, drop_rate=0, weight_decay=1e-4)

    # cur_best_pre_0, stopping_step = 0, 0
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    #
    # loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    # for epoch in range(args.epoch):
    #     t1 = time()
    #     loss, mf_loss, emb_loss = 0., 0., 0.
    #     n_batch = data_generator.n_train // args.batch_size + 1
    #
    #     for idx in range(n_batch):
    #         users, pos_items, neg_items = data_generator.sample()
    #         #输出改一下   改为耦合后的user-embedding和 item-embedding
    #         user_embeddings, item_g_embeddings = model(users, pos_items,neg_items,drop_flag=args.node_dropout_flag)
    #         #这里需要改一下啦
    #         batch_loss, batch_mf_loss, batch_emb_loss = model.create_bpr_loss(u_g_embeddings,
    #                                                                           pos_i_g_embeddings,
    #                                                                           neg_i_g_embeddings)
    #         optimizer.zero_grad()
    #         batch_loss.backward()
    #         optimizer.step()
    #
    #         loss += batch_loss
    #         mf_loss += batch_mf_loss
    #         emb_loss += batch_emb_loss
    #
    #     if (epoch + 1) % 10 != 0:
    #         if args.verbose > 0 and epoch % args.verbose == 0:
    #             perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
    #                 epoch, time() - t1, loss, mf_loss, emb_loss)
    #             print(perf_str)
    #         continue
    #
    #     t2 = time()
    #     users_to_test = list(data_generator.test_set.keys())
    #     ret = test(model, users_to_test, drop_flag=False)
    #
    #     t3 = time()
    #
    #     loss_loger.append(loss)
    #     rec_loger.append(ret['recall'])
    #     pre_loger.append(ret['precision'])
    #     ndcg_loger.append(ret['ndcg'])
    #     hit_loger.append(ret['hit_ratio'])
    #
    #     if args.verbose > 0:
    #         perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f], recall=[%.5f, %.5f], ' \
    #                    'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
    #                    (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, ret['recall'][0], ret['recall'][-1],
    #                     ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
    #                     ret['ndcg'][0], ret['ndcg'][-1])
    #         print(perf_str)
    #
    #     cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
    #                                                                 stopping_step, expected_order='acc', flag_step=5)
    #
    #     # *********************************************************
    #     # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
    #     if should_stop == True:
    #         break
    #
    #     # *********************************************************
    #     # save the user & item embeddings for pretraining.
    #     if ret['recall'][0] == cur_best_pre_0 and args.save_flag == 1:
    #         torch.save(model.state_dict(), args.weights_path + str(epoch) + '.pkl')
    #         print('save the weights in path: ', args.weights_path + str(epoch) + '.pkl')
    #
    # recs = np.array(rec_loger)
    # pres = np.array(pre_loger)
    # ndcgs = np.array(ndcg_loger)
    # hit = np.array(hit_loger)
    #
    # best_rec_0 = max(recs[:, 0])
    # idx = list(recs[:, 0]).index(best_rec_0)
    #
    # final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
    #              (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
    #               '\t'.join(['%.5f' % r for r in pres[idx]]),
    #               '\t'.join(['%.5f' % r for r in hit[idx]]),
    #               '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
    # print(final_perf)

    # 验证
    x = range(10)
    ax = plt.gca()
    plt.plot(x, loss_f, 'b', label="loss")
    # plt.plot(x, pre, 'r', label="Precision")
    plt.title("loss")
    plt.legend(loc='upper right')
    plt.figure()

    plt.plot(x, pre, 'b', label="pre")
    # plt.plot(x, pre, 'r', label="Precision")
    plt.title("pre")
    plt.legend(loc='upper right')
    plt.figure()

    plt.plot(x, recall, 'b', label="recall")
    # plt.plot(x, pre, 'r', label="Precision")
    plt.title("recall")
    plt.legend(loc='upper right')
    plt.figure()

    plt.plot(x, acc1, 'b', label="acc")
    # plt.plot(x, pre, 'r', label="Precision")
    plt.title("acc")
    plt.legend(loc='upper right')
    plt.figure()

    plt.plot(x, F1, 'b', label="F1")
    # plt.plot(x, pre, 'r', label="Precision")
    plt.title("F1")
    plt.legend(loc='upper right')
    plt.figure()

    plt.plot(x, AUC, 'b', label="auc")
    # plt.plot(x, pre, 'r', label="Precision")
    plt.title("auc")
    plt.legend(loc='upper right')
    plt.show()
















