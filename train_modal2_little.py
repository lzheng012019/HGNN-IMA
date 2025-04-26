import scipy.io
import urllib.request
import dgl
import math
import numpy as np


import modal2_missing

import torch

from data_loader import data_loader
from utils.data import load_data
from utils.pytorchtools import EarlyStopping
import argparse
import time
import process
import random
from sklearn import metrics
from sklearn.metrics import f1_score,accuracy_score
from sklearn.utils import resample
import os
import open_clip
import json
import re
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score,adjusted_rand_score
from sklearn.utils import shuffle as skshuffle
import datetime
import random
# data_url = 'https://s3.us-east-2.amazonaws.com/dgl.ai/dataset/ACM.mat'
# data_file_path = '/tmp/ACM.mat'
writer = SummaryWriter('runs/logs')
ap = argparse.ArgumentParser(description='MRGNN testing for the DBLP dataset')
ap.add_argument('--feats-type', type=int, default=3,
                help='Type of the node features used. ' +
                        '0 - loaded features; ' +
                        '1 - only target node features (zero vec for others); ' +
                        '2 - only target node features (id vec for others); ' +
                        '3 - all id vec. Default is 2;' +
                    '4 - only term features (id vec for others);' + 
                    '5 - only term features (zero vec for others).')
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--hidden_dim', type=int, default=64, help='Dimension of the node hidden state. Default is 64.')
ap.add_argument('--num_heads', type=int, default=8, help='Number of the attention heads. Default is 8.')
ap.add_argument('--epoch', type=int, default=300, help='Number of epochs.')
ap.add_argument('--patience', type=int, default=100, help='Patience.')
ap.add_argument('--repeat', type=int, default=1, help='Repeat the training and testing for N times. Default is 1.')
ap.add_argument('--num_layers', type=int, default=1)
ap.add_argument('--lr', type=float, default=1e-3)
ap.add_argument('--dropout', type=float, default=0.5)
ap.add_argument('--batch-size', type=int, default=64)
ap.add_argument('--weight-decay', type=float, default=5e-3)
ap.add_argument('--slope', type=float, default=0.05)
ap.add_argument('--dataset', type=str, default='ACM')
ap.add_argument('--edge-feats', type=int, default=64)
ap.add_argument('--device', type=int, default=1)
ap.add_argument('--schedule_step', type=int, default=1000)
ap.add_argument('--use_norm', type=bool, default=False)
ap.add_argument('--type', type=str, default='late')

args = ap.parse_args()
if args.device >= 0:
    device = torch.device("cuda:"+str(args.device))
    torch.cuda.set_device(args.device) 
else:
    device = torch.device('cpu')
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

def visual_embd(embd, label, seed=0):
    visual_vec = TSNE(n_components=2).fit_transform(embd)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(visual_vec[:,0], visual_vec[:,1], c=label, s=3)

    plt.show()

    fig_name = 'test.jpg'
    plt.savefig(fig_name)
    plt.close(fig)
def sp_to_spt(mat):
    coo = mat.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


def mat2tensor(mat):
    if type(mat) is np.ndarray:
        return torch.from_numpy(mat).type(torch.FloatTensor)
    return sp_to_spt(mat)
# dataset = data_loader('/home/zyd/HGB/'+args.dataset)
# edge_dict = {}

# for i, meta_path in dataset.links['meta'].items():
#     edge_dict[(str(meta_path[0]), str(meta_path[0]) + '_' + str(meta_path[1]), str(meta_path[1]))] = (torch.tensor(dataset.links['data'][i].tocoo().row - dataset.nodes['shift'][meta_path[0]]), torch.tensor(dataset.links['data'][i].tocoo().col - dataset.nodes['shift'][meta_path[1]]))

# node_count = {}
# for i, count in dataset.nodes['count'].items():
#     print(i, node_count)
#     node_count[str(i)] = count

# G = dgl.heterograph(edge_dict, num_nodes_dict = node_count, device=device)
# """
# for ntype in G.ntypes:
#     G.nodes[ntype].data['inp'] = dataset.nodes['attr'][ntype]
#     # print(G.nodes['attr'][ntype].shape)
# """

# G.node_dict = {}
# G.edge_dict = {}
# for ntype in G.ntypes:
#     G.node_dict[ntype] = len(G.node_dict)
# for etype in G.etypes:
#     G.edge_dict[etype] = len(G.edge_dict)
#     G.edges[etype].data['id'] = torch.ones(G.number_of_edges(etype), dtype=torch.long).to(device) * G.edge_dict[etype] 


# feats_type = args.feats_type
# features_list, adjM, labels, train_val_test_idx, dl = load_data(args.dataset)
# features_list = [mat2tensor(features).to(device) for features in features_list]

# if feats_type == 0:
#     in_dims = [features.shape[1] for features in features_list]
# elif feats_type == 1 or feats_type == 5:
#     save = 0 if feats_type == 1 else 2
#     in_dims = []#[features_list[0].shape[1]] + [10] * (len(features_list) - 1)
#     for i in range(0, len(features_list)):
#         if i == save:
#             in_dims.append(features_list[i].shape[1])
#         else:
#             in_dims.append(10)
#             features_list[i] = torch.zeros((features_list[i].shape[0], 10)).to(device)
# elif feats_type == 2 or feats_type == 4:
#     save = feats_type - 2
#     in_dims = [features.shape[0] for features in features_list]
#     for i in range(0, len(features_list)):
#         if i == save:
#             in_dims[i] = features_list[i].shape[1]
#             continue
#         dim = features_list[i].shape[0]
#         indices = np.vstack((np.arange(dim), np.arange(dim)))
#         indices = torch.LongTensor(indices)
#         values = torch.FloatTensor(np.ones(dim))
#         features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)
# elif feats_type == 3:
#     in_dims = [features.shape[0] for features in features_list]
#     for i in range(len(features_list)):
#         dim = features_list[i].shape[0]
#         indices = np.vstack((np.arange(dim), np.arange(dim)))
#         indices = torch.LongTensor(indices)
#         values = torch.FloatTensor(np.ones(dim))
#         features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)


# for ntype in G.ntypes:
#     G.nodes[ntype].data['inp'] = features_list[int(ntype)]#.to(device)
def my_Kmeans(x, y, k=4, time=10, return_NMI=True):

    x = np.array(x)
    x = np.squeeze(x)
    y = np.array(y)

    if len(y.shape) > 1:
        y = np.argmax(y, axis=1)

    estimator = KMeans(n_clusters=k)
    ARI_list = []  # adjusted_rand_score(
    NMI_list = []
    if time:
        # print('KMeans exps {}次 æ±~B平å~]~G '.format(time))
        for i in range(time):
            estimator.fit(x, y)
            y_pred = estimator.predict(x)
            score = normalized_mutual_info_score(y, y_pred)
            NMI_list.append(score)
            s2 = adjusted_rand_score(y, y_pred)
            ARI_list.append(s2)
        # print('NMI_list: {}'.format(NMI_list))
        score = sum(NMI_list) / len(NMI_list)
        s2 = sum(ARI_list) / len(ARI_list)
        print('NMI (10 avg): {:.4f} , ARI (10avg): {:.4f}'.format(score, s2))

    else:
        estimator.fit(x, y)
        y_pred = estimator.predict(x)
        score = normalized_mutual_info_score(y, y_pred)
        print("NMI on all label data: {:.5f}".format(score))
    if return_NMI:
        return score, s2

def load_g(datasets):
    print("dataset:"+datasets)
    print(device)
    if datasets == 'IMDB':
        features_list,features_list_img,node_test,labels, train_val_test_idx, rdf, g = process.load_IMDB_data(
        train_val_test_dir='/train_val_test_idx1.npz')
    elif datasets == 'AMAZON':
        features_list,features_list_img, node_test,labels, train_val_test_idx, rdf, g = process.load_AMAZON_data(
            train_val_test_dir='/train_val_test_idx_amaze.npz')
    elif datasets == 'DOUBAN':
        features_list,features_list_img,node_test, labels, train_val_test_idx, rdf, g = process.load_DOUBAN_929_data(
            train_val_test_dir='/train_val_test_idx_929.npz')
    elif datasets == 'MMDB':
        features_list,features_list_img,node_test, labels, train_val_test_idx, rdf, g = process.load_imdb2_data(
            train_val_test_dir='train_val_test_idx_imdb2.npy')
    elif datasets == 'AMAZON1':
        features_list,features_list_img,node_test, labels, train_val_test_idx, rdf, g = process.load_amazon1_data(
        train_val_test_dir='')
    elif datasets == 'AMAZON3':
        features_list,features_list_img,node_test, labels, train_val_test_idx, rdf, g = process.load_amazon3_data(
        train_val_test_dir='')
    features_list = [torch.FloatTensor(features).to(device) for features in features_list]
    features_list_img = [torch.FloatTensor(features).to(device) for features in features_list_img]
    # features_all = torch.cat((features_list), dim=0)
    # features_img_all = torch.cat((features_list_img), dim=0)
    num_nodes = features_list[0].shape[0]
    feature_size = [features.shape[1] for features in features_list]
    g = g.to(device)
    print(g)
    #print(g,features_list[0].shape,features_list[1].shape)

    for i,ntype in enumerate(g.ntypes):
        #print(features_list[i].shape)
        g.nodes[ntype].data['f'] = features_list[i]
        g.nodes[ntype].data['f_img'] = features_list_img[i]

    print(g)
    
    labels_ev = labels.copy()
    labels = torch.Tensor(labels).to(device)
    train_idx = train_val_test_idx['train_idx']

    val_idx = train_val_test_idx['val_idx']

    test_idx = train_val_test_idx['test_idx']
    # test_idx.append(val_idx)
    # test_idx = list(test_idx)

    # test_idx = test_idx + list(val_idx)
    # test_idx = np.array(test_idx)
    return g,train_idx,val_idx,test_idx,labels,num_nodes,feature_size,node_test,labels_ev
def micro_macro_f1_score(logits, labels):
    """计算Micro-F1和Macro-F1得分

    :param logits: tensor(N, C) 预测概率，N为样本数，C为类别数
    :param labels: tensor(N) 正确标签
    :return: float, float Micro-F1和Macro-F1得分
    """
    prediction = torch.argmax(logits, dim=1).long().cpu().numpy()
    labels = labels.cpu().numpy()
    micro_f1 = f1_score(labels, prediction, average='micro')
    macro_f1 = f1_score(labels, prediction, average='macro')
    return micro_f1, macro_f1
def evaluate(logits,labels,mask):
    return micro_macro_f1_score(logits[mask], labels[mask])
def set_random_seed(seed):
    """设置Python, numpy, PyTorch的随机数种子

    :param seed: int 随机数种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    dgl.seed(seed)

def normalizeText(text):
    text = text.lower()
    text = re.sub(r'<br />', r' ', text).strip()
    text = re.sub(r'^https?:\/\/.*[\r\n]*', ' L ', text, flags=re.MULTILINE)
    text = re.sub(r'[\~\*\+\^`_#\[\]|]', r' ', text).strip()
    text = re.sub(r'[0-9]+', r' N ', text).strip()
    text = re.sub(r'([/\'\-\.?!\(\)",:;])', r' \1 ', text).strip()
    return text
def save_result(final_embed, data_file):
    att_output_files = data_file
    if not os.path.exists(att_output_files):
        os.makedirs(att_output_files)
    result_emb = att_output_files + ".out"
    f = open(result_emb, 'w', encoding='utf-8')
    for i in range(np.shape(final_embed)[0]):
        f.write(str(i) + ' ')
        # print(vocab[i])
        for j in range(np.shape(final_embed)[1]):
            f.write(str(final_embed[i, j].real) + ' ')
        f.write('\n')
    f.close()
def main():
    set_random_seed(args.seed)
    g, train_idx, val_idx, test_idx,labels,num_nodes,feature_size,node_test,labels_ev= load_g(args.dataset)
    #train_labels = labels[train_idx].cpu().numpy()
    labels_first_two = labels  # 提取前两列标签
    labels = labels_first_two
    #print(labels[train_idx][:2],labels[val_idx][:2],labels[test_idx][:2])

    g.node_dict = {}
    g.edge_dict = {}
    for ntype in g.ntypes:
        g.node_dict[ntype] = len(g.node_dict)
    for etype in g.etypes:
        g.edge_dict[etype] = len(g.edge_dict)
        g.edges[etype].data['id'] = torch.ones(g.number_of_edges(etype), dtype=torch.long).to(device) * g.edge_dict[etype] 
    in_dims = feature_size

    
    if args.type=='late':
        HGT = model_late.HGT
  
    elif args.type=='modal2_missing':
        HGT = modal2_missing.HGT

    model = HGT(g, n_inps=in_dims[0], n_hid=args.hidden_dim, n_out=g.num_classes, n_layers=args.num_layers, n_heads=args.num_heads, use_norm = args.use_norm).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=args.schedule_step, max_lr = 1e-3, pct_start=0.05)
    #pos_weight = torch.tensor([0.5,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]).to(device)
    label_counts = labels.sum(dim=0)  # 统计每个标签的正类样本数
    total_samples = labels.size(0)  # 总样本数
    # num_labels = 23
    # # 计算每个标签的权重（反比例权重）
    # # label_weights = total_samples / (num_labels * label_counts)  # 权重是样本数与标签频率的反比
    # label_weights = torch.ones_like(label_counts, dtype=torch.float)

    # # 根据标签的出现次数赋值权重，小于 1500 的标签赋值为 5，其余赋值为 1
    # label_weights[label_counts < 4000] = 2
    # label_weights[label_counts < 2500] = 3
    # label_weights[label_counts <1000] = 5
    # print(label_counts)
    
    loss_fcn = torch.nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, save_path='./checkpoint/checkpoint_{}_{}.pt'.format(args.dataset, args.num_layers))
    train_step = 0



#     clipmodel.cuda()
#     image_features_list = []
#     text_features_list = []
#     for image_batch, text_batch in train_loader:
# # 将 numpy 数组转换为 torch 张量，并移动到正确的设备
#         image_batch = (image_batch).to(device)
#         text_batch = (text_batch).to(device)

#         # 使用 CLIP 模型提取图像和文本特征
#         with torch.no_grad(), torch.cuda.amp.autocast():
#             image_features = clipmodel.encode_image(image_batch)
#             text_features = clipmodel.encode_text(text_batch)
#         image_features_list.append(image_features.cpu().numpy())
#         text_features_list.append(text_features.cpu().numpy())

#     image_features = np.concatenate(image_features_list, axis=0)
#     text_features = np.concatenate(text_features_list, axis=0)
#     np.save('./amazon-3/item_image_features.npy', image_fea


    best_dev_metric = 0
    for epoch in range(args.epoch):
        t_start = time.time()
        model.train()

        logits,text,img,s_loss = model(g, g.predict_ntype)
        #print(logits.shape, labels.shape)
        #print(logits,labels)
        logits= logits.to(torch.float32)
        labels=labels.to(torch.float32)

        train_loss = loss_fcn(logits[train_idx], labels[train_idx].long())+s_loss#+ loss_fcn(text[train_idx], labels[train_idx].long()) + loss_fcn(img[train_idx], labels[train_idx].long()))/3.0  + s_loss
        pred = logits.argmax(1)
        train_acc = (pred[train_idx] == labels[train_idx]).float().mean()

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        train_step += 1
        scheduler.step()
        t_end = time.time()
        print('Epoch {:05d} | Train_Loss: {:.4f} | Train_acc: {:.4f} |Time: {:.4f}'.format(epoch, train_loss.item(), train_acc.item(), t_end-t_start))
        t_start = time.time()
        model.eval()
        with torch.no_grad():
            logits,text,img,s_loss = model(g, g.predict_ntype)
            val_loss = loss_fcn(logits[val_idx], labels[val_idx].long())+s_loss# loss_fcn(text[val_idx], labels[val_idx].long()) + loss_fcn(img[val_idx], labels[val_idx].long()))/3.0 + s_loss
            pred = logits.cpu().numpy().argmax(axis=1)
        val_acc   = (pred[val_idx] == labels[val_idx].cpu().numpy()).mean()
        val_scores = evaluate(logits,labels,val_idx)
        if val_scores[1] >= best_dev_metric:  # this epoch get best performance
    # pdb.set_trace()
            best_dev_epoch = epoch
            best_dev_metric = val_scores[1] # update best metric(f1 score)
            torch.save(model.state_dict(),'./checkpoint/checkpoint_{}_{}_{}.pt'.format(args.dataset, args.num_layers,args.type))
            print(f'save best model: {epoch}')
        # print('pred', pred[test_idx], labels[test_idx].cpu().numpy())

        # 保存正确和错误的索引到txt文件


        test_acc  = (pred[test_idx] == labels[test_idx].cpu().numpy()).mean()
        test_scores = evaluate(logits,labels,test_idx)
        # print('test_pred', pred[test_idx])
        # print(test_acc)
        test_pred_ = pred[test_idx]
        t_end = time.time()
        print('Epoch {:05d} | Val_Loss {:.4f} | Val_acc {:.4f} | Test_acc {:.4f} | Time(s) {:.4f}'.format(
            epoch, val_loss.item(), val_acc.item(), test_acc.item(), t_end - t_start))
        print('Test Micro-F1 {:.4f} | Test Macro-F1 {:.4f}'.format(*test_scores))
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print('Early stopping!')
            break

    writer.close()
    #model.load_state_dict(torch.load("best_model.pth"))
    model.load_state_dict(torch.load('./checkpoint/checkpoint_{}_{}_{}.pt'.format(args.dataset, args.num_layers,args.type)))
    #model.load_state_dict(torch.load('./checkpoint/checkpoint_AMAZON2_1_105.pt'))
    model.eval()
    test_logits = []
    with torch.no_grad():
        logits,text,img ,s_loss = model(g, g.predict_ntype)
        # test_logits = logits[test_idx]
        pred = logits.cpu().numpy().argmax(axis=1)
        # onehot = np.eye(g.num_classes, dtype=np.int32)
        # pred = onehot[pred]
        test_scores = evaluate(logits,labels,test_idx)
        print('Test Micro-F1 {:.4f} | Test Macro-F1 {:.4f}'.format(*test_scores))
    #     model.embedding_out(g,g.predict_ntype)
    # visual_embd(logits.cpu().numpy(), labels.cpu().numpy(), seed=0)
    correct_idx = (pred[test_idx] == labels[test_idx].cpu().numpy())
    correct_indices = test_idx[correct_idx]
    incorrect_indices = test_idx[~correct_idx]
    my_Kmeans(logits[test_idx].cpu().numpy(),labels[test_idx].cpu().numpy(),k=g.num_classes,time=10,return_NMI=True)

    # with open('correct_late.txt', 'w') as f_correct, open('incorrect_late.txt', 'w') as f_incorrect:
    # # 将正确和错误的索引写入文件
    #     f_correct.write('\n'.join(map(str, correct_indices.tolist())))
    #     f_incorrect.write('\n'.join(map(str, incorrect_indices.tolist())))

    print("Indices saved to 'correct_indices.txt' and 'incorrect_indices.txt'.")
# def main():
#     set_random_seed(args.seed)
#     g, train_idx, val_idx, test_idx, labels, num_nodes, feature_size, node_test, labels_ev = load_g(args.dataset)
#     g=g.to(device)

#     # 创建节点和边类型映射字典
#     g.node_dict = {ntype: i for i, ntype in enumerate(g.ntypes)}
#     g.edge_dict = {etype: i for i, etype in enumerate(g.etypes)}
#     for etype in g.etypes:
#         g.edges[etype].data['id'] = (
#             torch.ones(g.number_of_edges(etype), dtype=torch.long).to(device) * g.edge_dict[etype]
#         )
#     in_dims = feature_size
#     model = HGT(g, n_inps=in_dims, n_hid=args.hidden_dim, n_out=g.num_classes, 
#                 n_layers=args.num_layers, n_heads=args.num_heads, use_norm=args.use_norm).to(device)

#     optimizer = torch.optim.AdamW(model.parameters(), weight_decay=args.weight_decay)
#     scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=args.schedule_step, max_lr=1e-3, pct_start=0.05)
#     early_stopping = EarlyStopping(patience=args.patience, verbose=True, save_path=f'./checkpoint/checkpoint_{args.dataset}_{args.num_layers}.pt')

#     # 使用DGL的DataLoader和MultiLayerNeighborSampler进行分批次训练
#     sampler = dgl.dataloading.MultiLayerNeighborSampler([10, 10])  # 示例采样邻居数
#     train_idx = torch.tensor(train_idx).to(device)
#     train_dataloader = dgl.dataloading.NodeDataLoader(
#         g, {g.predict_ntype: train_idx}, sampler, batch_size=args.batch_size, shuffle=True, device=device
#     )

#     for epoch in range(args.epoch):
#         model.train()
#         total_loss = 0
#         total_acc = 0
#         batch_count = 0
#         for input_nodes, output_nodes, blocks in train_dataloader:
#             blocks = [block.to(device) for block in blocks]
#             batch_labels = labels[output_nodes[g.predict_ntype]].to(device)
#             batch_logits = model(blocks, blocks[0].srcdata['feat'])
#             loss = F.cross_entropy(batch_logits, batch_labels.long())
#             pred = batch_logits.argmax(1)
#             acc = (pred == batch_labels).float().mean()

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             scheduler.step()

#             total_loss += loss.item()
#             total_acc += acc.item()
#             batch_count += 1

#         avg_train_loss = total_loss / batch_count
#         avg_train_acc = total_acc / batch_count
#         print(f'Epoch {epoch:05d} | Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.4f}')

#         # 验证阶段
#         model.eval()
#         with torch.no_grad():
#             val_logits = model(g, g.predict_ntype)
#             val_loss = F.cross_entropy(val_logits[val_idx], labels[val_idx])
#             pred = val_logits[val_idx].argmax(1)
#             val_acc = (pred == labels[val_idx].to(device)).float().mean().item()

#             # 测试阶段
#             test_logits = model(g, g.predict_ntype)
#             test_pred = test_logits[test_idx].argmax(1)
#             test_acc = (test_pred == labels[test_idx].to(device)).float().mean().item()

#             print(f'Epoch {epoch:05d} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}')

#             early_stopping(val_loss, model)
#             if early_stopping.early_stop:
#                 print('Early stopping!')
#                 break

#     # 测试阶段
#     model.load_state_dict(torch.load(f'./checkpoint/checkpoint_{args.dataset}_{args.num_layers}.pt'))
#     model.eval()
#     with torch.no_grad():
#         test_logits = model(g, g.predict_ntype)
#         test_scores = evaluate(test_logits, labels, test_idx)
#         print(f'Test Micro-F1 {test_scores[0]:.4f} | Test Macro-F1 {test_scores[1]:.4f}')
if __name__ == '__main__':
    main()