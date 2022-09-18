import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import random
import json
import pickle
from collections import defaultdict
from operator import itemgetter
import logging


from torch_geometric.data import Data, DataLoader
from torch.optim.lr_scheduler import _LRScheduler


from models import  GraphCAD, outlierLoss
from utils import *
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# CUDA_VISIBLE_DEVICES=1 python main.py --data_dir '/raid/chenbo/outlier_detection/release_data/alpha_data.pkl' 

def add_arguments(args):
    # essential paras
    args.add_argument('--data_dir', type=str, help="data_dir", required = True)
    args.add_argument('--saved_dir', type=str, help="log_name", default= "saved_model")
    args.add_argument('--log_name', type=str, help="log_name", default = "log")

    # training paras.
    args.add_argument('--epochs', type=int, help="training #epochs", default=1000)
    args.add_argument('--seed', type=int, help="seed", default=1)
    args.add_argument('--lr', type=float, help="learning rate", default=5e-4)
    args.add_argument('--min_lr', type=float, help="min lr", default=1e-4)
    args.add_argument('--bs', type=int, help="batch size", default=1)
    args.add_argument('--input_dim', type=int, help="input dimension", default=256)
    args.add_argument('--out_dim', type=int, help="output dimension", default=256)
    args.add_argument('--verbose', type=int, help="eval", default=1)

    # model paras.
    args.add_argument('--outer_layer', type=int, help="#layers of GraphCAD", default = 2)
    args.add_argument('--inner_layer', type=int, help="#layers of node_update", default = 1)
    args.add_argument('--is_global', help="whether to add global information", action = "store_false")
    args.add_argument('--is_edge', help="whether to use edge update", action = "store_false")
    args.add_argument('--pooling', type=str, help="pooing_type", choices=['memory', 'avg', 'min', 'max'], default = "memory")

    args.add_argument('--is_lp', help="whether to use link prediction loss", action = "store_false")
    args.add_argument("--lp_weight", type = float, help="the weight of link prediction loss", default=0.1)

    args = args.parse_args()
    return args





def logging_builder(args):
    logger = logging.getLogger(__file__)
    logger.setLevel(logging.DEBUG)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.DEBUG)
    fileHandler = logging.FileHandler(os.path.join(os.getcwd(), args.log_name), mode='w')
    fileHandler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    consoleHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)
    logger.addHandler(fileHandler)
    return logger


class WarmupLinearLR(_LRScheduler):
    def __init__(self, optimizer, step_size, min_lr, peak_percentage=0.1, last_epoch=-1):
        self.step_size = step_size
        self.peak_step = peak_percentage * step_size
        self.min_lr = min_lr
        super(WarmupLinearLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        ret = []
        for tmp_min_lr, tmp_base_lr in zip(self.min_lr, self.base_lrs):
            if self._step_count <= self.peak_step:
                ret.append(tmp_min_lr + (tmp_base_lr - tmp_min_lr) * self._step_count / self.peak_step)
            else:
                ret.append(tmp_min_lr + max(0, (tmp_base_lr - tmp_min_lr) * (self.step_size - self._step_count) / (self.step_size - self.peak_step)))
        # print(self._step_count, self.step_size, ret)
        return ret

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args = add_arguments(args)
    setup_seed(args.seed)
    logger = logging_builder(args)
    print(args)
    os.makedirs(os.path.join(os.getcwd(), args.saved_dir), exist_ok = True)

    encoder = GraphCAD(logger, args, args.input_dim, args.out_dim, args.outer_layer, args.inner_layer, is_global = args.is_global, is_edge = args.is_edge, pooling= args.pooling).cuda()
    criterion = outlierLoss(args, logger, is_lp = args.is_lp, lp_weight = args.lp_weight).cuda()
    

    with open(args.data_dir, 'rb') as files:
        data_collection = pickle.load(files)
    data, y, train_mask, train_label_index, train_edge_ids, train_edge_labels, test_mask, test_label_index,test_edge_ids, test_edge_labels = data_collection
    # for older version of pyg
    data = Data(**data.__dict__)

    edges_attrs = torch.ones(data.edge_index.size(0))
    data_set = DataLoader([Data(x = data.x.cuda(), edge_index = data.edge_index.cuda().t(), y = y.cuda(), edge_attr = edges_attrs.cuda().unsqueeze(-1))], batch_size=1, shuffle = True)    

    optimizer = torch.optim.Adam([{'params': encoder.parameters(), 'lr': args.lr}])
    optimizer.zero_grad()

    logger.info(f"Warm up schedular: {args.epochs}")
    scheduler = WarmupLinearLR(optimizer, args.epochs, min_lr=[args.min_lr])

    encoder.train()
    epoch_num = 0
    max_map = -1
    max_auc = -1
    max_epoch = -1
    for epoch_num in range(args.epochs):
        batch_loss = []
        batch_contras_loss = []
        batch_lp_loss = []
        batch_edge_score = []
        batch_labels = []
        batch_index = 0
        for batch_data in tqdm(data_set):
            batch_index += 1
            node_outputs, adj_matrix, adj_weight, labels, batch_item = batch_data.x, batch_data.edge_index, batch_data.edge_attr.squeeze(-1), batch_data.y, batch_data.batch
            node_outputs, adj_weight, centroid, output_loss, centroid_loss, edge_prob = encoder(node_outputs, adj_matrix, adj_weight, batch_item, 1)

            # training index
            node_outputs = node_outputs[train_mask][train_label_index]
            output_loss = output_loss[train_mask][train_label_index]
            edge_prob = edge_prob[train_edge_ids]
            edge_labels = train_edge_labels.cuda()
            labels = labels[train_mask][train_label_index]
            

            overall_loss, _, contras_loss, lp_loss = criterion(output_loss, centroid_loss, edge_prob, edge_labels, adj_matrix, batch_item, labels, node_outputs, centroid)
            # overall_loss.backward()             
            overall_loss = overall_loss / args.bs
            overall_loss.backward()   
            batch_loss.append(overall_loss.item())
            batch_contras_loss.append(contras_loss.item())
            batch_lp_loss.append(lp_loss.item())
            if (batch_index + 1) % args.bs == 0: 
                optimizer.step()
                # scheduler.step()
                optimizer.zero_grad()

        avg_batch_loss = np.mean(np.array(batch_loss))
        avg_batch_contras_loss = np.mean(np.array(batch_contras_loss))
        avg_batch_lp_loss = np.mean(np.array(batch_lp_loss))
        logger.info("Epoch:{} Overall loss: {:.6f} Contrastive loss: {:.6f} LP_loss: {:.6f}".format(epoch_num, avg_batch_loss, avg_batch_contras_loss, avg_batch_lp_loss))        
        
        if (epoch_num + 1) % args.verbose == 0:
            encoder.eval()
            test_loss = []
            test_contras_loss = []
            test_lp_loss = []
            test_gt = []

            labels_list = []
            scores_list = []
            with torch.no_grad():
                for batch_test in tqdm(data_set):
                    node_outputs, adj_matrix, adj_weight, labels, batch_item = batch_test.x, batch_test.edge_index, batch_test.edge_attr.squeeze(-1), batch_test.y, batch_test.batch
                    node_outputs, adj_weight, centroid, output_loss, centroid_loss, edge_prob = encoder(node_outputs, adj_matrix, adj_weight, batch_item, 1)
                    centroid = centroid.squeeze(0)
                    centroid_loss = centroid_loss.squeeze(0)

                    # test index
                    node_outputs = node_outputs[test_mask][test_label_index]
                    output_loss = output_loss[test_mask][test_label_index]
                    edge_prob = edge_prob[test_edge_ids]
                    edge_labels = test_edge_labels.cuda()
                    labels = labels[test_mask][test_label_index]
                    
                    test_each_overall_loss, scores, test_each_contras_loss, test_each_lp_loss = criterion(output_loss, centroid_loss, edge_prob, edge_labels, adj_matrix, batch_item, labels, node_outputs, centroid)

                    scores = scores.detach().cpu().numpy()
                    scores_list.append(scores)
                    labels = labels.detach().cpu().numpy()
                    test_gt.append(labels)

                    test_loss.append(test_each_overall_loss.item())
                    test_contras_loss.append(test_each_contras_loss.item())
                    test_lp_loss.append(test_each_lp_loss.item())
            avg_test_loss = np.mean(np.array(test_loss))
            avg_test_contras_loss = np.mean(np.array(test_contras_loss))
            avg_test_lp_loss = np.mean(np.array(test_lp_loss)) 
        
            auc, maps = MAPs(test_gt, scores_list)
            logger.info("Epoch: {} Auc: {:.6f} Maps: {:.6f} Max-Auc: {:.6f} Max-Maps: {:.6f}".format(epoch_num, auc, maps, max_auc, max_map))
            if maps > max_map or auc > max_auc:
                max_epoch = epoch_num
                max_map = maps if maps > max_map else max_map
                max_auc = auc if auc > max_auc else max_auc
                # state = {'encoder': encoder.state_dict()}
                # torch.save(state, saved_file + "model_" + str(epoch_num))
                logger.info("***************** Epoch: {} Max Auc: {:.6f} Maps: {:.6f} *******************".format(epoch_num, max_auc, max_map))
            encoder.train()
            optimizer.zero_grad()
    logger.info("***************** Max_Epoch: {} Max Auc: {:.6f} Maps: {:.6f}*******************".format(max_epoch, max_auc, max_map))