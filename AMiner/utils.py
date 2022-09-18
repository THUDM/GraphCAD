import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import multiprocessing
from sklearn.metrics import roc_auc_score, auc, roc_curve
from torch_geometric.utils import add_self_loops, degree, softmax, to_dense_adj, dense_to_sparse
from operator import itemgetter
from scipy import sparse
import random



def MAPs(label_lists, score_lists):
    assert len(label_lists) == len(score_lists)
    maps = []
    mean_auc = []
    total_count = 0
    # print(np.array(score_lists).shape)
    total_nan = 0
    for sub_labels, sub_scores in zip(label_lists, score_lists):
        assert len(sub_labels) == len(sub_scores)
        combine = [each for each in zip(sub_scores, sub_labels)]
        sorted_combine = sorted(combine, key=itemgetter(0))
        # print(sorted_combine)
        rights = 0
        ps = []
        tmp_scores = []
        tmp_labels = []
        for index in range(len(sorted_combine)):
            ins_scores, ins_labels = sorted_combine[index]
            tmp_scores.append(ins_scores)
            tmp_labels.append(ins_labels)
            if(ins_labels == 0):
                rights += 1
                ps.append(rights/(index+1))

        tmp_scores = np.array(tmp_scores)
        

        nan_num = len(tmp_scores[np.isnan(tmp_scores)])
        total_nan += nan_num
        tmp_scores = np.nan_to_num(tmp_scores)



        tmp_labels = np.array(tmp_labels)
        auc = roc_auc_score(1-tmp_labels, -1 * tmp_scores)

        ap = np.mean(np.array(ps))

        maps.append((ap, len(sub_labels)))
        mean_auc.append(auc)
        total_count += len(sub_labels)
    assert len(maps) == len(mean_auc) == len(label_lists)
    maps_scores = 0
    maps_weight = 0
    for each in maps:
        ap, count = each
        each_w = total_count / count
        
        maps_scores += ap * each_w 
        maps_weight += each_w
    norm_maps = maps_scores/maps_weight
    mean_auc = np.mean(np.array(mean_auc))

    return mean_auc, norm_maps



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
