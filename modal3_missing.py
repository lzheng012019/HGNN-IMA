#cross-neighbors 双线性 和类型无关
import dgl
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
class CrossModalLayer(nn.Module):
    def __init__(self, num_modalities, feature_size,num_relation,num_head):
        super(CrossModalLayer, self).__init__()
        self.num_modalities = num_modalities
        self.feature_size = feature_size
        self.d_k = feature_size // num_head

        
        # 初始化每个模态的权重和偏置
        self.w_m = nn.ParameterDict({
            f'modal_{i}': nn.Parameter(torch.randn(num_relation,num_head, self.d_k, self.d_k))
            for i in range(num_modalities)
        })

        
        # 用于输出特征变换

    def forward(self, g,ntype,id):
        """
        g: DGLGraph 对象
        features_dict: 包含每个模态特征的字典，key 是模态，value 是特征张量
        """

        src_k=dict()
        dst_q=dict()
        src_k['text']=g.edata['src_text'][ntype]
        src_k['img']=g.edata['src_img'][ntype]
        src_k['transh'] = g.edata['src_transh'][ntype]
        dst_q['text']=g.edata['dst_text'][ntype]
        dst_q['img']=g.edata['dst_img'][ntype]
        dst_q['transh'] = g.edata['dst_transh'][ntype]

        lambda_scores = []
        for M, m1 in enumerate(src_k):

            # 获取当前模态的目标特征和聚合特征
            key   = torch.bmm(src_k[m1].transpose(1,0), self.w_m[f'modal_{M}'][id]).transpose(1,0)
            att   = (dst_q[m1] * key).sum(dim=-1)
            #print(torch.squeeze(score).shape)
            lambda_scores.append(torch.squeeze(att))

        lambda_scores = torch.stack(lambda_scores, dim=0)

        lambda_ir = F.softmax(lambda_scores, dim=0) 
        # print(lambda_ir[:,0])
       
        return lambda_ir
    
class HGTLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_types, num_relations, n_heads, dropout = 0.2, use_norm = False):
        super(HGTLayer, self).__init__()

        self.in_dim        = in_dim
        self.out_dim       = out_dim
        self.num_types     = num_types
        self.num_relations = num_relations
        self.n_heads       = n_heads
        self.d_k           = out_dim // n_heads
        self.sqrt_dk       = math.sqrt(self.d_k)
        
        self.k_linears   = nn.ModuleList()
        self.q_linears   = nn.ModuleList()
        self.v_linears   = nn.ModuleList()
        self.a_linears   = nn.ModuleList()
        self.norms       = nn.ModuleList()
        self.use_norm    = use_norm
        
        for t in range(num_types):
            self.k_linears.append(nn.Linear(in_dim,   out_dim))
            self.q_linears.append(nn.Linear(in_dim,   out_dim))
            self.v_linears.append(nn.Linear(in_dim,   out_dim))
            self.a_linears.append(nn.Linear(out_dim,  out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))
            
        self.relation_pri   = nn.Parameter(torch.ones(num_relations, self.n_heads))
        self.relation_att   = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg   = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))
        self.relation_modal = nn.Parameter(torch.Tensor( 2, n_heads, self.d_k, self.d_k))
        #self.relation_bias  = nn.Parameter(torch.Tensor(num_relations, 2,1))
        self.skip           = nn.Parameter(torch.ones(num_types))
        self.drop           = nn.Dropout(dropout)
        
        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)
        self.crossmodal = CrossModalLayer(3, out_dim,num_relations,n_heads)
    def edge_attention(self, edges):
        etype = edges.data['id'][0]
        relation_att = self.relation_att[etype]
        relation_pri = self.relation_pri[etype]
        relation_msg = self.relation_msg[etype]
        key   = torch.bmm(edges.src['k'].transpose(1,0), relation_att).transpose(1,0)
        key_img = torch.bmm(edges.src['k_img'].transpose(1,0), relation_att).transpose(1,0)
        key_s = torch.bmm(edges.src['k_s'].transpose(1,0), relation_att).transpose(1,0)


        att   = (edges.dst['q'] * key).sum(dim=-1) * relation_pri / self.sqrt_dk#替换att (node,head)
        att_img = (edges.dst['q_img'] * key_img).sum(dim=-1) * relation_pri / self.sqrt_dk
        att_s = (edges.dst['q_s'] * key_s).sum(dim=-1) * relation_pri / self.sqrt_dk
        #print(key.shape)
        val   = torch.bmm(edges.src['v'].transpose(1,0), relation_msg).transpose(1,0)
        val_img = torch.bmm(edges.src['v_img'].transpose(1,0), relation_msg).transpose(1,0)
        val_img_s = torch.bmm(edges.src['v_s'].transpose(1,0), relation_msg).transpose(1,0)
        # print(edges.data['lambda_txt'].shape,att.shape)
        # print(att[0],att_img[0],att_s[0])
        #beta = att *  edges.data['lambda_txt'].unsqueeze(1) + att_img * edges.data['lambda_img'].unsqueeze(1) + att_s * edges.data['lambda_s'].unsqueeze(1)
        # print(beta[0])
        return {'a': att,'a_img':att_img, 'a_s':att_s, 'v': val,'v_img':val_img,'v_s':val_img_s,'lambda_txt':edges.data['lambda_txt'],'lambda_img':edges.data['lambda_img'],'lambda_s':edges.data['lambda_s']}
    
    def message_func(self, edges):
        return {'v': edges.data['v'],'v_img':edges.data['v_img'],'v_s':edges.data['v_s'],'a': edges.data['a'],'a_img': edges.data['a_img'],'a_s':edges.data['a_s'],'lambda_txt':edges.data['lambda_txt'],'lambda_img':edges.data['lambda_img'],'lambda_s':edges.data['lambda_s']}
    
    def reduce_func(self, nodes):
        att = nodes.mailbox['a']*10
        att_img = nodes.mailbox['a_img']*10
        att_s = nodes.mailbox['a_s']*10
        att_align =-(torch.abs(att - att_img) + torch.abs(att - att_s) + torch.abs(att_img - att_s))/3.0
        # print('att_align',att_align.shape)
        # print(att_align[0])
        beta2 = F.softmax(att_align, dim=1) 
        att = F.softmax(att, dim=1)
        att_img = F.softmax(att_img, dim=1)
        att_s = F.softmax(att_s, dim=1)

        l_text = nodes.mailbox['lambda_txt']
        l_img = nodes.mailbox['lambda_img']
        l_s = nodes.mailbox['lambda_s']
        #print(att.shape,l_text.shape)#过于接近

        # print('beta2',beta2[0])

        # print('att',att[0])
        # print('att_img',att_img[0])

        beta1 = att * l_text + att_img * l_img + att_s * l_s
        beta = beta1 * beta2
        beta = F.softmax(beta, dim=1)

        # print(att.shape,l_text.shape)
        

        #print(beta.shape,nodes.mailbox['v'].shape)
        
        h   = torch.sum(beta.unsqueeze(dim = -1) * nodes.mailbox['v'], dim=1)
        h_img = torch.sum(beta.unsqueeze(dim = -1) * nodes.mailbox['v_img'], dim=1)
        h_s = torch.sum(beta.unsqueeze(dim = -1) * nodes.mailbox['v_s'], dim=1)
        return {'t': h.view(-1, self.out_dim),'t_img':h_img.view(-1, self.out_dim),'t_s':h_s.view(-1, self.out_dim)}
        
    def forward(self, G, inp_key, out_key,inp_key_img,out_key_img,inp_key_s,out_key_s):
        node_dict, edge_dict = G.node_dict, G.edge_dict

        for srctype, etype, dsttype in G.canonical_etypes:
            dst_features=dict()
            src_features=dict()
            src_to_dst = dict()
            #print(srctype, etype, dsttype)
            ntype=(srctype,etype,dsttype)
            #print(ntype)
            k_linear = self.k_linears[node_dict[srctype]]
            v_linear = self.v_linears[node_dict[srctype]] 
            q_linear = self.q_linears[node_dict[dsttype]]
            #print('inp_key',inp_key)
            
            G.nodes[srctype].data['k'] = k_linear(G.nodes[srctype].data[inp_key]).view(-1, self.n_heads, self.d_k)
            G.nodes[srctype].data['v'] = v_linear(G.nodes[srctype].data[inp_key]).view(-1, self.n_heads, self.d_k)
            G.nodes[dsttype].data['q'] = q_linear(G.nodes[dsttype].data[inp_key]).view(-1, self.n_heads, self.d_k)
            G.nodes[srctype].data['k_img'] = k_linear(G.nodes[srctype].data[inp_key_img]).view(-1, self.n_heads, self.d_k)
            G.nodes[srctype].data['v_img'] = v_linear(G.nodes[srctype].data[inp_key_img]).view(-1, self.n_heads, self.d_k)
            G.nodes[dsttype].data['q_img'] = q_linear(G.nodes[dsttype].data[inp_key_img]).view(-1, self.n_heads, self.d_k)
            G.nodes[srctype].data['k_s'] = k_linear(G.nodes[srctype].data[inp_key_s]).view(-1, self.n_heads, self.d_k)
            G.nodes[srctype].data['v_s'] = v_linear(G.nodes[srctype].data[inp_key_s]).view(-1, self.n_heads, self.d_k)
            G.nodes[dsttype].data['q_s'] = q_linear(G.nodes[dsttype].data[inp_key_s]).view(-1, self.n_heads, self.d_k)
            G.apply_edges(lambda edges: {
                'src_text': edges.src['k'],
                'dst_text': edges.dst['q'],
                'src_img': edges.src['k_img'],
                'dst_img': edges.dst['q_img'],
                'src_transh': edges.src['k_s'],
                'dst_transh': edges.dst['q_s'],
            },etype=etype)
            #print(G.edata['src_text'][ntype])
            id = G.edges[etype].data['id'][0]
            lambda_ir = self.crossmodal(G,ntype,id)
            if id==5:
                s_img = lambda_ir[1].mean() + lambda_ir[2].mean()
            # print('lambda_ir',lambda_ir[0][0][0],lambda_ir[1][0][0],lambda_ir[2][0][0])    

            G.edges[etype].data['lambda_txt'] = lambda_ir[0]
            G.edges[etype].data['lambda_img'] = lambda_ir[1]
            G.edges[etype].data['lambda_s'] = lambda_ir[2]
            #print('lambda_ir',lambda_ir.shape)

            G.apply_edges(func=self.edge_attention, etype=etype)

        G.multi_update_all({etype : (self.message_func, self.reduce_func) \
                            for etype in edge_dict}, cross_reducer = 'mean')
        for ntype in G.ntypes:
            n_id = node_dict[ntype]
            alpha = torch.sigmoid(self.skip[n_id])
            trans_out = self.a_linears[n_id](G.nodes[ntype].data['t'])
            trans_out_img = self.a_linears[n_id](G.nodes[ntype].data['t_img'])
            trans_out_s = self.a_linears[n_id](G.nodes[ntype].data['t_s'])
            trans_out = trans_out * alpha + G.nodes[ntype].data[inp_key] * (1-alpha)
            trans_out_img = trans_out_img * alpha + G.nodes[ntype].data[inp_key_img] * (1-alpha)
            trans_out_s = trans_out_s * alpha + G.nodes[ntype].data[inp_key_s] * (1-alpha)
            #print('alpha',alpha)
            if self.use_norm:
                G.nodes[ntype].data[out_key] = self.drop(self.norms[n_id](trans_out))
                G.nodes[ntype].data[out_key_img] = self.drop(self.norms[n_id](trans_out_img))
                G.nodes[ntype].data[out_key_s] = self.drop(self.norms[n_id](trans_out_s))
            else:
                G.nodes[ntype].data[out_key] = self.drop(trans_out)
                G.nodes[ntype].data[out_key_img] = self.drop(trans_out_img)
                G.nodes[ntype].data[out_key_s] = self.drop(trans_out_s)
        return s_img
    def __repr__(self):
        return '{}(in_dim={}, out_dim={}, num_types={}, num_types={})'.format(
            self.__class__.__name__, self.in_dim, self.out_dim,
            self.num_types, self.num_relations)
                
class HGT(nn.Module):
    def __init__(self, G, n_inps, n_hid, n_out, n_layers, n_heads, use_norm = True):
        super(HGT, self).__init__()
        self.gcs = nn.ModuleList()
        self.n_inps = n_inps
        self.n_hid = n_hid
        self.n_out = n_out
        self.n_layers = n_layers
        self.adapt_ws_s = nn.ModuleList()
        self.adapt_ws_text  = nn.ModuleList()
        self.adapt_ws_img  = nn.ModuleList()
        for t in range(len(G.node_dict)):
            # self.adapt_ws.append(nn.Linear(n_inps, n_hid))
            self.adapt_ws_text.append(nn.Sequential(nn.BatchNorm1d(n_inps),
                nn.Linear(n_inps, n_hid)))
            self.adapt_ws_img.append(nn.Sequential(nn.BatchNorm1d(n_inps),
                nn.Linear(n_inps, n_hid)))
            self.adapt_ws_s.append(nn.Sequential(nn.BatchNorm1d(n_inps),
                                                 nn.Linear(n_inps, n_hid)))
        for _ in range(n_layers):
            self.gcs.append(HGTLayer(n_hid, n_hid, len(G.node_dict), len(G.edge_dict), n_heads, use_norm = use_norm))
        #self.out = nn.Linear(n_hid, n_out)
        self.predict = nn.Linear(n_hid, n_out)
        self.project = nn.Sequential(
            # nn.Dropout(0.6),
            # nn.Linear(in_dim, hidden_dim),
            # nn.Dropout(dropout),
            # nn.Tanh(),
            nn.Dropout(0.6),
            nn.Linear(n_hid, 1),
            nn.Dropout(0.5),
            nn.Tanh()
        )
        self.linear = nn.Sequential(
            nn.Linear(n_hid*(self.n_layers),n_hid,bias=False),
            # nn.Dropout(0.6),
            # nn.Tanh(),
            
            nn.Linear(n_hid,n_out)

            # nn.BatchNorm1d(num_features=(self.out_dim))
        )
        #self.linear_out = nn.Linear(n_hid,n_out)

    def forward(self, G, predict_type):
        for ntype in G.ntypes:
            n_id = G.node_dict[ntype]
            G.nodes[ntype].data['h'] = torch.tanh(self.adapt_ws[n_id](G.nodes[ntype].data['f']))
            G.nodes[ntype].data['h_img'] = torch.tanh(self.adapt_ws[n_id](G.nodes[ntype].data['f_img']))
            G.nodes[ntype].data['h_s'] = torch.tanh(self.adapt_ws[n_id](G.nodes[ntype].data['f_s']))
        h_list=[]
        s_loss=0
        for i in range(self.n_layers):
            s_img = self.gcs[i](G, 'h', 'h','h_img','h_img','h_s','h_s')
            s_loss+=s_img
            #self.gcs[i](G, 'h_img', 'h_img')
            msg = G.nodes[predict_type].data['h'].unsqueeze(dim=1)
            msg_img = G.nodes[predict_type].data['h_img'].unsqueeze(dim=1)
            msg_s = G.nodes[predict_type].data['h_s'].unsqueeze(dim=1)
            msg_all = torch.cat([msg, msg_img,msg_s], dim=1)#(node,64)
            
            #print('msg_all',msg_all.shape)
            w = self.project(msg_all)
            beta = torch.softmax(w, dim=1)
            # print('beta',beta.shape)
            h = (beta * msg_all).sum(1)
            # h_list.append(h)
            
        # h =torch.cat(h_list,dim=1)
        # h = self.linear(h)
        # h = h.cpu().detach().numpy()
        # out = (self.predict(h))  # tensor(N_i, d_out)
        # h =torch.cat(h_list,dim=1)
        # h = self.linear(h)
        text = self.predict(msg.squeeze(1))
        image =  self.predict(msg_img.squeeze(1))
        number = self.predict(msg_s.squeeze(1))
        out  =self.predict(h)
        return out,text,image,number,s_loss
    def __repr__(self):
        return '{}(n_inp={}, n_hid={}, n_out={}, n_layers={})'.format(
            self.__class__.__name__, self.n_inp, self.n_hid,
            self.n_out, self.n_layers)
