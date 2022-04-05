
'''
Processing data
'''
import torch
import numpy as np
from collections import defaultdict
import utils
import pickle
from sampler import *
import random

def get_union(union):
    ind = []
    for s in union :
        ind+=list(s)
    return set(ind)

def set_cover(universe, subsets):
    elements = set(e for s in subsets for e in s)
    if elements != universe:
        return None, None
    covered = set()
    cover = []
    idx = []
    while covered != elements:
        subset = max(subsets, key=lambda s: len(s - covered))
        cover.append(subset)
        idx.append(subsets.index(subset))
        covered |= subset
    return cover, idx

def get_cover_idx(HE):
    universe = get_union(HE)
    print(len(universe))
    tmp_HE = [set(edge) for edge in HE]
    _, cover_idx = set_cover(universe, tmp_HE)
    print(len(cover_idx))
    return cover_idx

def split_dataset(dataset):
    data_path = None
    try:
        data_path = f'/data/{dataset}.pt'             
    except:
        raise Exception('dataset {} not supported!'.format(dataset))

    data_dict = torch.load(data_path)
    ne = data_dict['N_edges']
    nv = data_dict['N_nodes']
    EdgeNodePair = torch.LongTensor(data_dict['EdgeNodePair'])
    incidence = torch.zeros(ne, nv)
    for elem in EdgeNodePair :
        e, v = elem
        incidence[e, v]=1
        
    HE = []
    for e in incidence.keys():
        HE.append(frozenset(incidence[e]))
        
    base_cover = get_cover_idx(HE)
    union = get_union(HE)
    tmp = [HE[idx] for idx in base_cover]
    assert union == get_union(tmp)
    base_num = len(base_cover)
    
    for split in range(5):
        # ground 60%, train 10(+50)%, validation 10(+10)%, test 20%
        ground_num = int(0.6*len(HE)) - base_num
        total_idx = list(range(len(HE))) 
        ground_idx = list(set(total_idx)-set(base_cover))
        ground_idx = random.sample(ground_idx, ground_num)      
        ground_num += base_num
        ground_idx += base_cover
        ground_valid_num = ground_num//6
        ground_valid_idx = random.sample(ground_idx, ground_valid_num)
        ground_train_num = ground_num - ground_valid_num
        
        ground_train_data = []
        ground_valid_data = []
        pred_data = []
        for idx in total_idx :
            if idx in ground_idx:
                if idx in ground_valid_idx:
                    ground_valid_data.append(HE[idx])
                else:
                    ground_train_data.append(HE[idx])
            else :
                pred_data.append(HE[idx])
                
        valid_only_num = int(0.25*len(pred_data))
        train_only_num = int(0.25*len(pred_data))
        test_num = len(pred_data) - (valid_only_num + train_only_num)
        
        random.shuffle(pred_data)
        train_only_data = pred_data[:train_only_num]
        valid_only_data = pred_data[train_only_num:-test_num]
        test_data = pred_data[-test_num:]
        # negatives        
        GP_train = ground_valid_data + ground_train_data + train_only_data
        GP_valid = ground_valid_data + ground_train_data + train_only_data + valid_only_data
        GP_test = GP_valid
        
        train_mns, train_sns, train_cns = neg_generator(GP_train, ground_train_num+train_only_num)
        valid_mns, valid_sns, valid_cns = neg_generator(GP_valid, ground_valid_num+valid_only_num)
        test_mns, test_sns, test_cns = neg_generator(GP_test, test_num)
            
        ground_train_data = [list(edge) for edge in ground_train_data]
        ground_valid_data = [list(edge) for edge in ground_valid_data]
        train_only_data = [list(edge) for edge in train_only_data]
        valid_only_data = [list(edge) for edge in valid_only_data]
        test_data = [list(edge) for edge in test_data]
        
        print(f"ground {len(ground_train_data)} + {len(ground_valid_data)} = {len(ground_train_data + ground_valid_data)}")
        print(f"train pos {len(ground_train_data)} + {len(train_only_data)} = {len(ground_train_data + train_only_data)}, neg {len(train_mns)}")
        print(f"valid pos {len(ground_valid_data)} + {len(valid_only_data)} = {len(ground_valid_data + valid_only_data)}, neg {len(valid_mns)}")
        print(f"test pos {len(test_data)}, neg {len(test_mns)}")
        torch.save({'ground_train': ground_train_data, 'ground_valid': ground_valid_data, \
                'train_only_pos': train_only_data, 'train_mns': train_mns, 'train_sns' : train_sns, 'train_cns' : train_cns,\
                'valid_only_pos': valid_only_data, 'valid_mns': valid_mns, 'valid_sns' : valid_sns, 'valid_cns' : valid_cns, \
                'test_pos': test_data, 'test_mns': test_mns, 'test_sns' : test_sns, 'test_cns' : test_cns},
                f'/workspace/jupyter/data/splits/{dataset}split{split}_val.pt')

def neg_generator(HE, pred_num):
    mns = MNSSampler(pred_num)
    sns = SNSSampler(pred_num)
    cns = CNSSampler(pred_num)
    
    t_mns = mns(set(HE))
    t_sns = sns(set(HE))
    t_cns = cns(set(HE))
    
    t_mns = list(t_mns)
    t_sns = list(t_sns)
    t_cns = list(t_cns)
    
    t_mns = [list(edge) for edge in t_mns]
    t_sns = [list(edge) for edge in t_sns]
    t_cns = [list(edge) for edge in t_cns]
    
    return t_mns, t_sns, t_cns
    

if __name__ == '__main__':
    args = utils.parse_args()
    if args.dataset_name == 'cora' or args.dataset_name == 'citeseer' or args.dataset_name == 'pubmed' :
        split_dataset(args, 'cocitation', args.dataset_name)
    elif 'dblp' in args.dataset_name :
        split_dataset(args, 'collaboration', args.dataset_name)        
    else : # 'coraA', 'dblpA'
        split_dataset(args, 'authorship', args.dataset_name)