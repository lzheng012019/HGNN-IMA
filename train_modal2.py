import scipy.io
import urllib.request
import dgl
import math
import numpy as np


import modal2_align
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
ap.add_argument('--hidden_dim', type=int, default=128, help='Dimension of the node hidden state. Default is 64.')
ap.add_argument('--num_heads', type=int, default=8, help='Number of the attention heads. Default is 8.')
ap.add_argument('--epoch', type=int, default=300, help='Number of epochs.')
ap.add_argument('--patience', type=int, default=50, help='Patience.')
ap.add_argument('--repeat', type=int, default=1, help='Repeat the training and testing for N times. Default is 1.')
ap.add_argument('--num_layers', type=int, default=1)
ap.add_argument('--lr', type=float, default=1e-3)
ap.add_argument('--dropout', type=float, default=0.5)
ap.add_argument('--batch-size', type=int, default=64)
ap.add_argument('--weight-decay', type=float, default=5e-3)
ap.add_argument('--slope', type=float, default=0.05)
ap.add_argument('--dataset', type=str, default='ACM')
ap.add_argument('--edge-feats', type=int, default=64)
ap.add_argument('--device', type=int, default=0)
ap.add_argument('--schedule_step', type=int, default=1000)
ap.add_argument('--use_norm', type=bool, default=False)
ap.add_argument('--type', type=str, default='late')

args = ap.parse_args()
if args.device >= 0:
    device = torch.device("cuda:"+str(args.device))
    torch.cuda.set_device(args.device) 
else:
    device = torch.device('cpu')
def sp_to_spt(mat):
    coo = mat.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))
from sklearn.manifold import TSNE
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
def mat2tensor(mat):
    if type(mat) is np.ndarray:
        return torch.from_numpy(mat).type(torch.FloatTensor)
    return sp_to_spt(mat)

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
    features_list = [torch.FloatTensor(features).to(device) for features in features_list]
    features_list_img = [torch.FloatTensor(features).to(device) for features in features_list_img]
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
    elif args.type=='modal2_align':
        HGT = modal2_align.HGT
    elif args.type=='modal2_missing':
        HGT = modal2_missing.HGT

    model = HGT(g, n_inps=in_dims[0], n_hid=args.hidden_dim, n_out=g.num_classes, n_layers=args.num_layers, n_heads=args.num_heads, use_norm = args.use_norm).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=args.schedule_step, max_lr = 1e-3, pct_start=0.05)
    label_counts = labels.sum(dim=0)  
    total_samples = labels.size(0)  

    loss_fcn = torch.nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, save_path='./checkpoint/checkpoint_{}_{}.pt'.format(args.dataset, args.num_layers))
    train_step = 0



    best_dev_metric = 0
    for epoch in range(args.epoch):
        t_start = time.time()
        model.train()

        logits,text,img = model(g, g.predict_ntype)

        logits= logits.to(torch.float32)
        labels=labels.to(torch.float32)

        train_loss = (loss_fcn(logits[train_idx], labels[train_idx].long()) + loss_fcn(text[train_idx], labels[train_idx].long()) + loss_fcn(img[train_idx], labels[train_idx].long()))/3.0
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
            logits,text,image = model(g, g.predict_ntype)
            val_loss = loss_fcn(logits[val_idx], labels[val_idx].long())#+ loss_fcn(text[val_idx], labels[val_idx].long()) + loss_fcn(img[val_idx], labels[val_idx].long()))/3.0
            pred = logits.cpu().numpy().argmax(axis=1)
        val_acc   = (pred[val_idx] == labels[val_idx].cpu().numpy()).mean()
        val_scores = evaluate(logits,labels,val_idx)
        if val_scores[0] >= best_dev_metric:  # this epoch get best performance
            best_dev_epoch = epoch
            best_dev_metric = val_scores[0] # update best metric(f1 score)
            torch.save(model.state_dict(),'./checkpoint/checkpoint_{}_{}_{}.pt'.format(args.dataset, args.num_layers,args.type))
            print(f'save best model: {epoch}')


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
        logits,text,img = model(g, g.predict_ntype)
        pred = logits.cpu().numpy().argmax(axis=1)
        test_scores = evaluate(logits,labels,test_idx)
        print('Test Micro-F1 {:.4f} | Test Macro-F1 {:.4f}'.format(*test_scores))
        model.embedding_out(g,g.predict_ntype)
    visual_embd(logits.cpu().numpy(), labels.cpu().numpy(), seed=0)
    correct_idx = (pred[test_idx] == labels[test_idx].cpu().numpy())
    correct_indices = test_idx[correct_idx]
    incorrect_indices = test_idx[~correct_idx]
    my_Kmeans(logits[test_idx].cpu().numpy(),labels[test_idx].cpu().numpy(),k=12,time=10,return_NMI=True)

    print("Indices saved to 'correct_indices.txt' and 'incorrect_indices.txt'.")

if __name__ == '__main__':
    main()