import torch
import numpy as np
from collections import defaultdict
import dgl
from batch import HEBatchGenerator

def gen_DGLGraph(args, ground):
    device = 'cuda:{}'.format(args.gpu) if args.gpu != -1 else 'cpu'
    he = []
    hv = []
    for i, edge in enumerate(ground):
        for v in edge :
            he.append(i)
            hv.append(v)
    data_dict = {
        ('node', 'in', 'edge'): (hv, he),        
        ('edge', 'con', 'node'): (he, hv)
    }
    g = dgl.heterograph(data_dict)
    return g.to(device)

def gen_data(args, dataset_name):
    device = 'cuda:{}'.format(args.gpu) if args.gpu != -1 else 'cpu'
    data_path = None
    try:
        data_path = f'/data/{dataset_name}.pt'             
    except:
        raise Exception('dataset {} not supported!'.format(dataset_name))

    data_dict = torch.load(data_path)
    NodeEdgePair = torch.LongTensor(data_dict['NodeEdgePair'])
    EdgeNodePair = torch.LongTensor(data_dict['EdgeNodePair'])
    ne = data_dict['N_edges']
    nv = data_dict['N_nodes']
    node_feat = data_dict['node_feat']
    nodewt = data_dict['nodewt']
    edgewt = data_dict['edgewt']
    
    args.input_dim = node_feat.shape[-1]
    args.ne = ne
    args.nv = nv
    
    if isinstance(node_feat, np.ndarray):
        args.v = torch.from_numpy(node_feat.astype(np.float32)).to(device)
    else:
        args.v = torch.from_numpy(np.array(node_feat.astype(np.float32).todense())).to(device)
        
    args.vidx = NodeEdgePair[:, 0].to(device)
    args.eidx = NodeEdgePair[:, 1].to(device)
    args.incidence = torch.zeros(ne, nv)
    for elem in EdgeNodePair :
        e, v = elem
        args.incidence[e, v]=1
    args.v_feat = torch.tensor(node_feat).type('torch.FloatTensor').to(device)
    args.e_feat = torch.ones(args.ne, args.dim_edge).to(device)
    
    # HNHN terms
    args.v_weight = torch.Tensor([(1/w if w > 0 else 1) for w in nodewt]).unsqueeze(-1).to(device)
    args.e_weight = torch.Tensor([(1/w if w > 0 else 1) for w in edgewt]).unsqueeze(-1).to(device)
    node2sum = defaultdict(list)
    edge2sum = defaultdict(list)
    e_reg_weight = torch.zeros(args.ne) 
    v_reg_weight = torch.zeros(args.nv) 
    for i, (node_idx, edge_idx) in enumerate(NodeEdgePair.tolist()):
        e_wt = args.e_weight[edge_idx]
        e_reg_wt = e_wt**args.alpha_e 
        e_reg_weight[edge_idx] = e_reg_wt
        node2sum[node_idx].append(e_reg_wt) 
        
        v_wt = args.v_weight[node_idx]
        v_reg_wt = v_wt**args.alpha_v
        v_reg_weight[node_idx] = v_reg_wt
        edge2sum[edge_idx].append(v_reg_wt)      
        
    v_reg_sum = torch.zeros(nv) 
    e_reg_sum = torch.zeros(ne) 
    for node_idx, wt_l in node2sum.items():
        v_reg_sum[node_idx] = sum(wt_l)
    for edge_idx, wt_l in edge2sum.items():
        e_reg_sum[edge_idx] = sum(wt_l)

    e_reg_sum[e_reg_sum==0] = 1
    v_reg_sum[v_reg_sum==0] = 1
    args.e_reg_weight = torch.Tensor(e_reg_weight).unsqueeze(-1).to(device)
    args.v_reg_sum = torch.Tensor(v_reg_sum).unsqueeze(-1).to(device)
    args.v_reg_weight = torch.Tensor(v_reg_weight).unsqueeze(-1).to(device)
    args.e_reg_sum = torch.Tensor(e_reg_sum).unsqueeze(-1).to(device)

    return args

def load_train(data_dict, bs, device):
    train_pos = data_dict["train_only_pos"] + data_dict["ground_train"]
    train_pos_label = [1 for i in range(len(train_pos))]
    train_batchloader = HEBatchGenerator(train_pos, train_pos_label, bs, device, test_generator=False)    
    return train_batchloader

def load_val(data_dict, bs, device, label):
    if label=="pos":
        val = data_dict["train_only_pos"] + data_dict["ground_train"]
        val_label = [1 for i in range(len(val))]
    else:
        val = data_dict[f"valid_{label}"]
        val_label = [0 for i in range(len(val))]
    val_batchloader = HEBatchGenerator(val, val_label, bs, device, test_generator=True)    
    return val_batchloader

def load_test(data_dict, bs, device, label):
    test = data_dict[f"test_{label}"]
    if label=="pos":
        test_label = [1 for i in range(len(test))]
    else:
        test_label = [0 for i in range(len(test))]
    test_batchloader = HEBatchGenerator(test, test_label, bs, device, test_generator=True)    
    return test_batchloader