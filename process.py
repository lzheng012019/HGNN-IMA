from re import A
import networkx as nx
import numpy as np
import scipy
import pickle
import dgl
import torch

def load_DOUBAN_929_data(prefix='/share/zhangyudong1-nfs/HGNN-IMA/douban',train_val_test_dir = '/'):

    features_0 = np.load(prefix + '/features_douban_928_1_movie.npz.npy')
    features_1 =np.load(prefix + '/features_douban_928_1_directors.npz.npy')
    features_2= np.load(prefix + '/features_douban_928_1_actor.npz.npy')
    node_test = features_0.shape[0]

    #img
    features_0_img = np.load(prefix + '/img_feature_douban_512_199.npz')['feature']

    features_1_img = np.load(prefix + '/features_douban_928_1_directors.npz.npy')
    features_2_img =np.load(prefix + '/features_douban_928_1_actor.npz.npy')
    features_1_img = np.zeros(features_1.shape)
    features_2_img = np.zeros(features_2.shape)

    labels = np.load(prefix + '/labels.npy')#movies类别
    train_val_test_idx = np.load(prefix + train_val_test_dir)
    rdf                = np.load(prefix+'/douban_actor_rdf.npy')
    md, ma = set(), set()
    for e1,e2,ntype in rdf:
    #d = directors.index(row['director_name'])
        if int(ntype)== 0 :
            md.add((e1, e2))
        elif int(ntype)== 2:
            ma.add((e1, e2))
    md, ma = list(md), list(ma)
    #print(ir)
    #1888 1472 3267
    #4278 2081 5257
    ma_m, ma_a = [e[0] for e in ma], [e[1]-3360 for e in ma]
    md_m, md_d = [e[0] for e in md], [e[1]-1888 for e in md]
    g=dgl.heterograph({
        ('0','ma','2'): (ma_m, ma_a),
        ('2','am','0'): (ma_a, ma_m),
        ('0','md','1'): (md_m, md_d),
        ('1','dm','0'): (md_d, md_m)
    })
    g.predict_ntype='0'
    g.num_classes=2
    meta_paths = [['ma', 'am'], ['md', 'dm']]
    train = train_val_test_idx['train_idx']
    return [features_0, features_1, features_2],[features_0_img, features_1_img, features_2_img],node_test,labels,train_val_test_idx,rdf,g

def load_AMAZON_data(prefix='/share/zhangyudong1-nfs/HGNN-IMA/amazon',train_val_test_dir = '/'):

    features_0 = np.load(prefix + '/feature_item_amazon.npz')['feature']
    features_1 =np.load(prefix + '/feature_review_amazon.npz')['feature']
    node_test = features_0.shape[0]
    #img
    features_0_img = np.load(prefix + '/img_feature_20_1_25_199.npz')['feature']

    features_1_img = np.load(prefix + '/feature_review_amazon.npz')['feature']
    features_1_img  = np.zeros(features_1.shape)

    labels = np.load(prefix + '/labels.npy')#movies类别
    train_val_test_idx = np.load(prefix + train_val_test_dir)
    rdf                = np.load(prefix+'/rdf.npy')
    ir, ii = set(), set()
    for e1,e2,ntype in rdf:
        #d = directors.index(row['director_name'])
        if int(ntype)== 0 or int(ntype)==1:
            ii.add((e1, e2))
        elif int(ntype)== 2:
            ir.add((e1, e2))


    ir, ii = list(ir), list(ii)
    #print(ir)
    ir_i, ir_r = [e[0] for e in ir], [e[1]-6158 for e in ir]
    ii_1, ii_2 = [e[0] for e in ii], [e[1] for e in ii]

    g=dgl.heterograph({
        ('item', 'ii', 'item'): (ii_1, ii_2),
        ('item', 'ii', 'item'): (ii_2, ii_1),
        ('item', 'ir', 'review'): (ir_i, ir_r),
        ('review', 'ri', 'item'): (ir_r, ir_i)
    })
    g.predict_ntype='item'
    g.num_classes=3
    train = train_val_test_idx['train_idx']
    meta_paths = [['ir', 'ri'], ['ii']]

    return [features_0, features_1],[features_0_img, features_1_img],node_test,labels,train_val_test_idx,rdf,g

def load_IMDB_data(prefix='/share/zhangyudong1-nfs/HGNN-IMA/imdb',train_val_test_dir = '/'):

    features_0 = scipy.sparse.load_npz(prefix + '/features_0.npz').todense()
    features_1 = scipy.sparse.load_npz(prefix + '/features_1.npz').todense()
    features_2 = scipy.sparse.load_npz(prefix + '/features_2.npz').todense()
    node_test = features_0.shape[0]

    features_0_img = np.load(prefix + '/img_feature_20_6_25_199.npz')['feature']
    features_1_img = scipy.sparse.load_npz(prefix + '/features_1.npz').todense()
    features_2_img = scipy.sparse.load_npz(prefix + '/features_2.npz').todense()

    type_mask = np.load(prefix + '/node_types.npy')
    labels = np.load(prefix + '/labels.npy')
    train_val_test_idx = np.load(prefix + train_val_test_dir)
    rdf                = np.load(prefix+'/rdf.npy')
    md,ma = set(), set() 
    for e1,e2,ntype in rdf:
        #d = directors.index(row['director_name'])
        if int(ntype)== 0 :
            md.add((e1, e2))
        elif int(ntype)== 2:
            ma.add((e1, e2))


    md, ma = list(md), list(ma)

    ma_m, ma_a = [e[0] for e in ma], [e[1]-6359 for e in ma]
    md_m, md_d = [e[0] for e in md], [e[1]-4278 for e in md]
    g=dgl.heterograph({
        ('0','ma','2'): (ma_m, ma_a),
        ('2','am','0'): (ma_a, ma_m),
        ('0','md','1'): (md_m, md_d),
        ('1','dm','0'): (md_d, md_m)
    })
    meta_paths = [['ma', 'am'], ['md', 'dm']]
    train = train_val_test_idx['train_idx']
    g.predict_ntype='0'
    g.predict_ntype='0'
    g.num_classes=3
    return [features_0, features_1, features_2],[features_0_img, features_1_img, features_2_img],node_test ,labels,train_val_test_idx,rdf,g

def load_amazon1_data(prefix='./amazon-1',train_val_test_dir = '/'):

    features_text = np.load(prefix + '/item_text_features.npy')
    features_img = np.load(prefix + '/item_image_features.npy')
    print(features_text.shape,features_img.shape)
    node_test=features_text.shape[0]

    #type_mask = np.load(prefix + '/node_types.npy')#节点类型标记

    labels = np.load(prefix + '/item_labels.npy')


    train_val_test_idx = np.load(prefix + '/item_data_dict.npy',allow_pickle=True).item()
    rdf                = np.load(prefix+'/triples.npy',allow_pickle=True)
    i1,i2,i3,i4 = set(), set(), set(), set()
    r1,r2,r3,r4 = set(), set(), set(), set()
    # dm,am,gm,ad,gd,ga = set(), set(), set(), set(), set(), set()
    print(labels[90],labels[627],labels[135],labels[141],labels[91])

    for e1,e2,ntype in rdf:
        #d = directors.index(row['director_name'])
        if (ntype)== "also_viewed" :
            i1.add((e1, e2))
            r1.add((e2, e1))

            #++dm.add((e2, e1))
        if (ntype)== "also_bought" :
            i2.add((e1, e2))
            r2.add((e2, e1))
            #dm.add((e2, e1))
        elif (ntype)== 'buy_after_viewing':
            i3.add((e1, e2))
            r3.add((e2, e1))
            #am.add((e2, e1))
        elif (ntype)== "bought_together":
            i4.add((e1, e2))
            r4.add((e2, e1))
            #gm.add((e2, e1))


    i1, i2,i3,i4 = list(i1), list(i2), list(i3), list(i4)
    r1, r2,r3,r4 = list(r1), list(r2), list(r3), list(r4)
    i1 = i1+r1
    i2 = i2+r2
    i3 = i3+r3
    i4 = i4+r4
    i1_1 ,i1_2 = [int(e[0]) for e in i1], [int(e[1]) for e in i1]

    i2_1 ,i2_2 = [int(e[0]) for e in i2], [int(e[1]) for e in i2]

    i3_1 ,i3_2 = [int(e[0]) for e in i3], [int(e[1]) for e in i3]

    i4_1 ,i4_2 = [int(e[0]) for e in i4], [int(e[1]) for e in i4]



    # 创建图
    g = dgl.heterograph({
        ('0', "also_viewed", '0'): (i1_1, i1_2),       
        ('0', "also_bought", '0'): (i2_1, i2_2),
        ('0', 'buy_after_viewing', '0'): (i3_1, i3_2),
        ('0', "bought_together", '0'):(i4_1, i4_2),
    })

    #meta_paths = [['ma', 'am'], ['md', 'dm']]
    train = train_val_test_idx['train_idx']

    meta_paths = [["also_viewed"], ["also_bought"],['buy_after_viewing'],["bought_together"]]
    train = train_val_test_idx['train_idx']

    g.predict_ntype='0'
    g.num_classes=12
    return [features_text],[features_img],node_test ,labels,train_val_test_idx,rdf,g


def load_amazon2_data(prefix='./amazon-1',train_val_test_dir = '/'):

    features_text = np.load(prefix + '/item_text_features.npy')
    features_img = np.load(prefix + '/item_image_features.npy')
    #feature_structure = np.load(prefix + '/entity_transh.npy')
    features_num =np.load(prefix + '/item_price.npy').squeeze()
    print(features_text.shape,features_img.shape)
    node_test=features_text.shape[0]

    #type_mask = np.load(prefix + '/node_types.npy')#节点类型标记

    labels = np.load(prefix + '/item_labels.npy')

    train_val_test_idx = np.load(prefix + '/item_data_dict.npy',allow_pickle=True).item()
    rdf                = np.load(prefix+'/triples.npy',allow_pickle=True)
    i1,i2,i3,i4 = set(), set(), set(), set()
    r1,r2,r3,r4 = set(), set(), set(), set()
    # dm,am,gm,ad,gd,ga = set(), set(), set(), set(), set(), set()

    for e1,e2,ntype in rdf:
        #d = directors.index(row['director_name'])
        if (ntype)== "also_viewed" :
            i1.add((e1, e2))
            r1.add((e2, e1))

            #++dm.add((e2, e1))
        if (ntype)== "also_bought" :
            i2.add((e1, e2))
            r2.add((e2, e1))
            #dm.add((e2, e1))
        elif (ntype)== 'buy_after_viewing':
            i3.add((e1, e2))
            r3.add((e2, e1))
            #am.add((e2, e1))
        elif (ntype)== "bought_together":
            i4.add((e1, e2))
            r4.add((e2, e1))
            #gm.add((e2, e1))


    i1, i2,i3,i4 = list(i1), list(i2), list(i3), list(i4)
    r1, r2,r3,r4 = list(r1), list(r2), list(r3), list(r4)
    i1 = i1+r1
    i2 = i2+r2
    i3 = i3+r3
    i4 = i4+r4
    i1_1 ,i1_2 = [int(e[0]) for e in i1], [int(e[1]) for e in i1]

    i2_1 ,i2_2 = [int(e[0]) for e in i2], [int(e[1]) for e in i2]

    i3_1 ,i3_2 = [int(e[0]) for e in i3], [int(e[1]) for e in i3]

    i4_1 ,i4_2 = [int(e[0]) for e in i4], [int(e[1]) for e in i4]

    g = dgl.heterograph({
        ('0', "also_viewed", '0'): (i1_1, i1_2),       
        ('0', "also_bought", '0'): (i2_1, i2_2),
        ('0', 'buy_after_viewing', '0'): (i3_1, i3_2),
        ('0', "bought_together", '0'):(i4_1, i4_2),
    })

    train = train_val_test_idx['train_idx']

    meta_paths = [["also_viewed"], ["also_bought"],['buy_after_viewing'],["bought_together"]]
    train = train_val_test_idx['train_idx']

    g.predict_ntype='0'
    g.num_classes=12
    return [features_text],[features_img],[features_num],node_test ,labels,train_val_test_idx,rdf,g



