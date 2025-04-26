# import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score,adjusted_rand_score
from sklearn.utils import shuffle as skshuffle
import classify
import datetime
import random
import matplotlib.pyplot as plt


def plot_embeddings(embeddings,X,Y,filename):
    # X, Y = read_node_label('../data/wiki/wiki_labels.txt')

    emb_list = []
    for k in X:
        emb_list.append(embeddings[k])
    emb_list = np.array(emb_list)

    model = TSNE(n_components=2)
    node_pos = model.fit_transform(emb_list)

    color_idx = {}
    for i in range(len(X)):
        color_idx.setdefault(Y[i][0], [])
        color_idx[Y[i][0]].append(i)

    for c, idx in color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c,s=12)
    plt.legend()
    plt.savefig("result_img.jpg")

def model_classification(X, Y, num_shuffle=10):
    shuffles = []
    for x in range(num_shuffle):
        shuffles.append(skshuffle(X, Y))
    #print(shuffles)#每次都打乱 次序不一样 崩溃
    #indices = np.arange(X.shape[0])
    #X, Y, shuffled_indices = skshuffle(X, Y, indices)
    #print(shuffled_indices)
    precisions = []
    for x, y in shuffles:
        prec = []
        for tr_train in np.arange(0.2, 1, 0.2):
            clf = classify.Classifier(clf=LogisticRegression(max_iter=5000))
            clfres = clf.split_train_evaluate(x,y,tr_train, prints=False)
            prec.append([clfres['micro'], clfres['macro'], clfres['samples'], clfres['weighted'], clfres['auc']])
        precisions.append(prec)
    precisions = np.array(precisions)
    return np.mean(precisions, axis=0)

def evaluationDataset(X, Y, method, dataset, us=None, vs=None, labs=None):
    dt = datetime.datetime.now()
    log_file = 'results/evaluation.logd.txt'


    print('【{}】 embedded by 【{}-{}】 evaluating...'.format(method, dataset,dim))
    precisions = model_classification(X, Y)

    csv_res = []
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write('----------------           {}--{}--{}        ---------------------------\n'.format(method, dataset, dim))
        for k, p in enumerate(precisions):
            csv_res.append([p[4], p[0], p[1]])
            f.write('tranin_percent {:.2f} :\tmicro-f1 : {:.4f},\tmacro-f1 : {:.4f}\tsamples-f1 : {:.4f}\tweighted-f1 : {:.4f}\tauc : {:.4f}\n'.format(0.2 * (k+1), p[0], p[1], p[2], p[3], p[4]))

    # with open('results/evaluation.log_{}.csv'.format(dt.date()), 'a', encoding='utf-8') as f:
    #     f.write('{}\t{}\n'.format('{}-{}'.format(method, dataset),
    #                                   '\t'.join([str(x) for x in np.concatenate(csv_res, axis=-1)])))

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
def main_ev(labels,num_nodes,final_embed,node_test,test_idx,k):
    #from main_amzon import load_g
    def set_random_seed(seed=0):
        """Set random seed.
        Parameters
        ----------
        seed : int
            Random seed to use
        """
        random.seed(seed)
        np.random.seed(seed)




    labels = labels
    X = []
    Y = []
    for i,label in enumerate(labels):
        X.append(i)
        Y.append([str(label)])




    dembs = []
    indice = []
    set_random_seed(1)
    with open('result/{}.out'.format(final_embed), 'r', encoding='utf-8') as f:
        flag = True
        c = 0
        for line in f.readlines():
            ls = line.strip().split()
            indice.append(int(ls[0]))
            dembs.append([float(x) for x in ls[1:]])
                # print(indice)
    dembs = np.array(dembs[:node_test])
    #print(labels.shape)
    evaluate_finnal = model_classification(dembs,labels)
    #print(evaluate_finnal)
    #print(dembs)
    file_name = 'logging.txt'
    NMI, ARI = my_Kmeans(dembs[test_idx], labels[test_idx], k)
    with open(file_name, 'a') as f:
        f.write('{}\n'.format(final_embed))


        for k,p in enumerate(evaluate_finnal):
            f.write(
                'tranin_percent {:.2f} :\tmicro-f1 : {:.4f},\tmacro-f1 : {:.4f}\tsamples-f1 : {:.4f}\tweighted-f1 : {:.4f}\tauc : {:.4f}\n'.format(
                    0.2 * (k + 1), p[0], p[1], p[2], p[3], p[4]))
            print(
                'tranin_percent {:.2f} :\tmicro-f1 : {:.4f},\tmacro-f1 : {:.4f}\tsamples-f1 : {:.4f}\tweighted-f1 : {:.4f}\tauc : {:.4f}\n'.format(
                    0.2 * (k + 1), p[0], p[1], p[2], p[3], p[4]))
        f.write(
            'NMI{:.4f} :\tARI : {:.4f}\n'.format(
                NMI,ARI))
        f.write('\n')
        print(
            'NMI{:.4f} :\tARI : {:.4f}\n'.format(
                NMI,ARI))
        print('\n')


    plot_embeddings(dembs,X,Y,filename=final_embed)


def main_ev_B(labels,num_nodes,final_embed,node_test,train_idx,test_idx,k):
    #from main_amzon import load_g

    def set_random_seed(seed=0):
        """Set random seed.
        Parameters
        ----------
        seed : int
            Random seed to use
        """
        random.seed(seed)
        np.random.seed(seed)




    labels = labels

    dembs = []
    indices = []
    set_random_seed(1)
    with open('result/{}.out'.format(final_embed), 'r', encoding='utf-8') as f:
        for line in f.readlines():
            ls = line.strip().split()
            indices.append(int(ls[0]))
            dembs.append([float(x) for x in ls[1:]])

    # 将数据转换为NumPy数组
    indices = np.array(indices)
    dembs = np.array(dembs)

    # 根据索引划分数据集
    x_train = dembs[train_idx]
    y_train = labels[train_idx]
    #print(y_train)
    x_test = dembs[test_idx]
    y_test = labels[test_idx]
    test_indices=indices[test_idx]
                # print(indice)
    # dembs = np.array(dembs[:node_test])
    clf = classify.Classifier(clf=LogisticRegression(max_iter=5000))
    clfres = clf.split_train_evaluate_B(x_train, y_train, x_test, y_test,test_indices, prints=False)
    print('micro: ',clfres['micro'],'macro: ',clfres['macro'])

    

# 打印或处理结果
    #print(results)
    #print(labels.shape)
    #evaluate_finnal = model_classification(dembs,labels)
    #print(evaluate_finnal)
    #print(dembs)
    # file_name = 'logging.txt'
    # NMI, ARI = my_Kmeans(dembs[test_idx], labels[test_idx], k)
    # with open(file_name, 'a') as f:
    #     f.write('{}\n'.format(final_embed))


    #     for k,p in enumerate(evaluate_finnal):
    #         f.write(
    #             'tranin_percent {:.2f} :\tmicro-f1 : {:.4f},\tmacro-f1 : {:.4f}\tsamples-f1 : {:.4f}\tweighted-f1 : {:.4f}\tauc : {:.4f}\n'.format(
    #                 0.2 * (k + 1), p[0], p[1], p[2], p[3], p[4]))
    #         print(
    #             'tranin_percent {:.2f} :\tmicro-f1 : {:.4f},\tmacro-f1 : {:.4f}\tsamples-f1 : {:.4f}\tweighted-f1 : {:.4f}\tauc : {:.4f}\n'.format(
    #                 0.2 * (k + 1), p[0], p[1], p[2], p[3], p[4]))
    #     f.write(
    #         'NMI{:.4f} :\tARI : {:.4f}\n'.format(
    #             NMI,ARI))
    #     f.write('\n')
    #     print(
    #         'NMI{:.4f} :\tARI : {:.4f}\n'.format(
    #             NMI,ARI))
    #     print('\n')


    #plot_embeddings(dembs,X,Y,filename=final_embed)
