import torch 
import torch.nn as nn
import numpy as np
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
from sklearn import metrics
from torchmetrics import AveragePrecision
import os
from tqdm import tqdm

import utils
from data_load import gen_data, gen_DGLGraph, load_train, load_val, load_test
import models
from sampler import *
from aggregator import *
from generator import MLPgenerator
from training import model_train, model_eval


def train(args):
    os.makedirs(f"/data/checkpoints/{args.folder_name}", exist_ok=True)
    f_log = open(f"logs/{args.folder_name}_train.log", "w")
    f_log.write(f"args: {args}\n")
    
    if args.fix_seed:
        np.random.seed(0)
        torch.manual_seed(0)
    train_DG = args.train_DG.split(":")
    args.train_DG = [int(train_DG[0][5:]), int(train_DG[1]), int(train_DG[0][5:])+int(train_DG[1])]    
    device = 'cuda:{}'.format(args.gpu) if args.gpu != -1 else 'cpu'
    

    for j in tqdm(range(args.exp_num)):    
        # Load data
        args = gen_data(args, args.dataset_name)
        data_dict = torch.load(f'/data/splits/{args.dataset_name}split{j}.pt')
        ground = data_dict["ground_train"] + data_dict["ground_valid"]
        g = gen_DGLGraph(args, ground)  
        train_batchloader = load_train(data_dict, args.bs, device) # only positives
        val_batchloader_pos = load_val(data_dict, args.bs, device, label="pos")
        val_batchloader_sns = load_val(data_dict, args.bs, device, label="sns")
        val_batchloader_mns = load_val(data_dict, args.bs, device, label="mns")
        val_batchloader_cns = load_val(data_dict, args.bs, device, label="cns")
    
        # Initialize models
        model = models.multilayers(models.HNHN, [args.input_dim, args.dim_vertex, args.dim_edge], \
                        args.n_layers, memory_dim=args.nv, K=args.memory_size)
        model.to(device)        
        Aggregator = None    
        cls_layers = [args.dim_vertex, 128, 8, 1]
        Aggregator = MaxminAggregator(args.dim_vertex, cls_layers)
        Aggregator.to(device)  
        size_dist = utils.gen_size_dist(ground)
        if args.gen == "MLP":
            dim = [64, 256, 256, args.nv]
            if args.dataset_name == "pubmed":
                dim = [128, 512, 512, args.nv]
            elif "dblp" in args.dataset_name:
                dim = [256, 1024, 2048, args.nv]
            print(f"{args.dataset_name} generator dimension: "+str(dim))
            Generator =  MLPgenerator(dim, args.nv, device, size_dist)
        Generator.to(device)
        average_precision = AveragePrecision()

        best_roc = 0
        best_epoch = 0 
        optim_D = torch.optim.RMSprop(list(model.parameters())+list(Aggregator.parameters()), lr=args.D_lr)
        optim_G = torch.optim.RMSprop(Generator.parameters(), lr=args.G_lr)
        
        for epoch in tqdm(range(args.epochs), leave=False):            
            train_pred, train_label = [], []
            d_loss_sum, g_loss_sum, count  = 0.0, 0.0, 0
            
            # Train
            while True :
                pos_hedges, pos_labels, is_last = train_batchloader.next()
                d_loss, g_loss, train_pred, train_label = model_train(args, g, model, Aggregator, Generator, optim_D, optim_G, pos_hedges, pos_labels, train_pred, train_label, device, epoch)
                d_loss_sum += d_loss
                g_loss_sum += g_loss
                count += 1
                if is_last :
                    break
                
            train_pred = torch.stack(train_pred)
            train_pred = train_pred.squeeze()
            train_label = torch.round(torch.cat(train_label, dim=0))        
            train_roc = metrics.roc_auc_score(np.array(train_label.cpu()), np.array(train_pred.cpu()))
            train_ap = average_precision(torch.tensor(train_pred), torch.tensor(train_label))            
            
            f_log.write(f'{epoch} epoch: Training d_loss : {d_loss_sum / count} / Training g_loss : {g_loss_sum / count} /')
            f_log.write(f'Training roc : {train_roc} / Training ap : {train_ap} \n')
    
            # Eval validation            
            val_pred_pos, total_label_pos = model_eval(args, val_batchloader_pos, g, model, Aggregator)
            val_pred_sns, total_label_sns = model_eval(args, val_batchloader_sns, g, model, Aggregator)
            auc_roc_sns, ap_sns = utils.measure(total_label_pos+total_label_sns, val_pred_pos+val_pred_sns)
            f_log.write(f"{epoch} epoch, SNS : Val AP : {ap_sns} / AUROC : {auc_roc_sns}\n")
            val_pred_mns, total_label_mns = model_eval(args, val_batchloader_mns, g, model, Aggregator)
            auc_roc_mns, ap_mns = utils.measure(total_label_pos+total_label_mns, val_pred_pos+val_pred_mns)
            f_log.write(f"{epoch} epoch, MNS : Val AP : {ap_mns} / AUROC : {auc_roc_mns}\n")
            val_pred_cns, total_label_cns = model_eval(args, val_batchloader_cns, g, model, Aggregator)
            auc_roc_cns, ap_cns = utils.measure(total_label_pos+total_label_cns, val_pred_pos+val_pred_cns)
            f_log.write(f"{epoch} epoch, CNS : Val AP : {ap_cns} / AUROC : {auc_roc_cns}\n")
            l = len(val_pred_pos)//3
            val_pred_all = val_pred_pos + val_pred_sns[0:l] + val_pred_mns[0:l] + val_pred_cns[0:l]
            total_label_all = total_label_pos + total_label_sns[0:l] + total_label_mns[0:l] + total_label_cns[0:l]
            auc_roc_all, ap_all = utils.measure(total_label_all, val_pred_all)
            f_log.write(f"{epoch} epoch, ALL : Val AP : {ap_all} / AUROC : {auc_roc_all}\n")
            f_log.flush()
            # Save best checkpoint
            if best_roc < (auc_roc_sns+auc_roc_mns+auc_roc_cns)/3:
                best_roc = (auc_roc_sns+auc_roc_mns+auc_roc_cns)/3
                best_epoch=epoch
                torch.save(model.state_dict(), f"/data/checkpoints/{args.folder_name}/model_{j}.pkt")
                torch.save(Aggregator.state_dict(), f"/data/checkpoints/{args.folder_name}/Aggregator_{j}.pkt")
                torch.save(Generator.state_dict(), f"/data/checkpoints/{args.folder_name}/Generator_{j}.pkt")
    f_log.close()
    
    with open(f"/data/checkpoints/{args.folder_name}/best_epochs.logs", "a") as e_log:  
        e_log.write(f"exp {j} best epochs: {best_epoch}\n")
    return args
def test(args, j):    
    args.checkpoint = f"/data/checkpoints/{args.folder_name}"
    f_log = open(f"logs/{args.folder_name}_results.log", "w")    
    f_log.write(f"{args}\n")    

    # Load data
    data_dict = torch.load(f'/data/splits/{args.dataset_name}split{j}.pt')
    args = gen_data(args, args.dataset_name, do_val=True)
    ground = data_dict["ground_train"] + data_dict["ground_valid"]
    g = gen_DGLGraph(args, ground)
    device = 'cuda:{}'.format(args.gpu) if args.gpu != -1 else 'cpu'

    # test set    
    test_batchloader_pos = load_test(data_dict, args.bs, device, label="pos")
    test_batchloader_sns = load_test(data_dict, args.bs, device, label="sns")
    test_batchloader_mns = load_test(data_dict, args.bs, device, label="mns")
    test_batchloader_cns = load_test(data_dict, args.bs, device, label="cns")
   
    # Initialize models
    model = models.multilayers(models.HNHN, [args.input_dim, args.dim_vertex, args.dim_edge], \
                    args.n_layers, memory_dim=args.nv, K=args.memory_size)
    model.to(device)
    model.load_state_dict(torch.load(f"{args.checkpoint}/model_{j}.pkt"))
    cls_layers = [args.dim_vertex, 128, 8, 1]
    Aggregator = MaxminAggregator(args.dim_vertex, cls_layers)
    Aggregator.to(device)
    Aggregator.load_state_dict(torch.load(f"{args.checkpoint}/Aggregator_{j}.pkt"))
    
    model.eval()
    Aggregator.eval()

    with torch.no_grad():
        test_pred_pos, total_label_pos = model_eval(args, test_batchloader_pos, g, model, Aggregator)
        test_pred_sns, total_label_sns = model_eval(args, test_batchloader_sns, g, model, Aggregator)
        auc_roc_sns, ap_sns = utils.measure(total_label_pos+total_label_sns, test_pred_pos+test_pred_sns)
        f_log.write(f"SNS : Test AP : {ap_sns} / AUROC : {auc_roc_sns}\n")
        test_pred_mns, total_label_mns = model_eval(args, test_batchloader_mns, g, model, Aggregator)
        auc_roc_mns, ap_mns = utils.measure(total_label_pos+total_label_mns, test_pred_pos+test_pred_mns)
        f_log.write(f"MNS : Test AP : {ap_mns} / AUROC : {auc_roc_mns}\n")
        test_pred_cns, total_label_cns = model_eval(args, test_batchloader_cns, g, model, Aggregator)
        auc_roc_cns, ap_cns = utils.measure(total_label_pos+total_label_cns, test_pred_pos+test_pred_cns)
        f_log.write(f"CNS : Test AP : {ap_cns} / AUROC : {auc_roc_cns}\n")
        l = len(test_pred_pos)//3
        test_pred_all = test_pred_pos + test_pred_sns[0:l] + test_pred_mns[0:l] + test_pred_cns[0:l]
        total_label_all = total_label_pos + total_label_sns[0:l] + total_label_mns[0:l] + total_label_cns[0:l]
        auc_roc_all, ap_all = utils.measure(total_label_all, test_pred_all)
        f_log.write(f"ALL : Test AP : {ap_all} / AUROC : {auc_roc_all}\n")
        f_log.flush()
        
if __name__ == "__main__":
    args = utils.parse_args()
    args.folder_name = "exp1"
    train(args)
    args = utils.parse_args()
    for j in range(args.exp_num):
        test(args, j)

    