from random import sample
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
import pickle

from torch_geometric.nn import GINConv_w as GINConv
from torch_geometric.utils import add_self_loops, degree, softmax, to_dense_adj, dense_to_sparse
from torch_scatter import scatter_add
import math
import numpy as np
import pyro

class outlierLoss(nn.Module):
    def __init__(self, args, logger, is_lp = True, lp_weight = 0.1):
        super(outlierLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss().cuda()
        self.dim = args.out_dim
        self.temp = 0.1
        self.margin_1 = 0.5
        self.lamb = 0.5
        self.thres = torch.tensor(0.0).cuda()
        self.lp_weight = lp_weight
        self.is_lp = is_lp
        logger.info("is_lp: %s, lp_weight: %f"%(self.is_lp, self.lp_weight))

    def cal_lp_loss(self, edge_prob, edge_labels):
        # feat_all_trans: [(b x N) x C]
        non_zero = torch.nonzero(edge_labels)
        zero = torch.nonzero(edge_labels == 0)
        if non_zero.size(0) != 0 and zero.size(0) != 0:
            pos_prob = edge_prob[non_zero][:8192]
            neg_prob = edge_prob[zero]
            logits = torch.cat((pos_prob, neg_prob.view(1, neg_prob.size(0)).repeat(pos_prob.size(), 1)), dim = 1)
            logits_labels = torch.zeros([pos_prob.size(0)]).cuda().long()
            lp_loss = self.cross_entropy(logits/self.temp, logits_labels)
        else:
            lp_loss = torch.tensor(0.0).cuda()
        return lp_loss

    def forward(self, feat_all, q, edge_prob, edge_labels, adj_mat, batch_item, labels, raw_feat_all, raw_centroid):
        
        # q2all: [N, 1]
        q = F.normalize(q.view(self.dim, 1), p=2, dim=0)
        feat_all = F.normalize(feat_all, p=2, dim=1)
        q2all = torch.mm(feat_all, q.view(self.dim, 1)).squeeze(-1)
        pos_len = torch.sum(labels, dim = 0)
        neg_len = q2all.size(0) - pos_len

        # pos [P]; neg [Neg]
        q2all_pos, q2all_neg = torch.split(q2all, [pos_len, neg_len], dim = 0)
        

        q2all_each_logits = torch.cat([q2all_pos.unsqueeze(-1), q2all_neg.view(1, neg_len).repeat(pos_len, 1)], dim = -1)
        # print(q2all_each_logits.size())
        q2all_each_logits = q2all_each_logits.view(pos_len, neg_len + 1)

        # pos: [b x P, 1]
        # neg: [b x p, Neg + (b - 1) * N]ß

        logits_labels = torch.zeros([pos_len]).cuda().long()
        
        contras_loss = self.cross_entropy(q2all_each_logits/self.temp, logits_labels)
        if self.is_lp:
            lp_loss = self.cal_lp_loss(edge_prob, edge_labels)
        else:
            lp_loss = torch.tensor(0.0).cuda()

        outlier_loss = contras_loss + self.lp_weight * lp_loss
        scores = torch.mm(feat_all, q.view(self.dim, 1)).squeeze(-1)
        return outlier_loss, scores, contras_loss, lp_loss

class GraphCAD(nn.Module):
    def __init__(self, logger, args,in_dim, out_dim, total_layer_num, ins_layer_num, is_norm = True, is_edge = True, is_node = True, is_system = True, is_global = True, pooling = "memory"):
        super(GraphCAD, self).__init__()  
        self.total_layer_num = total_layer_num
        self.is_edge = is_edge
        self.is_node = is_node
        self.is_system = is_system
        self.in_dim = in_dim
        # edge_model
        # self.edgemodel = None
        if is_edge:
            logger.info("EdgeUpdate")
            self.edgemodel = EdgeUpdate(is_global, out_dim, 1)

        # conv_model
        if is_node:
            logger.info("NodeUpdate")
            self.node_updates = nn.ModuleList()
            for _ in range(self.total_layer_num):
                self.node_updates.append(NodeUpdate(out_dim, out_dim, is_norm, ins_layer_num))

        # sys_model    
        if is_system:
            logger.info("SystemUpdate")
            self.sys_updates = nn.ModuleList()    
            for _ in range(self.total_layer_num):
                self.sys_updates.append(SystemUpdate(out_dim, out_dim, pooling))


        self.mlp_head = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )   

        self.drop_layer = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        logger.info("is_edge: %r, is_global: %r pooling: %r"%(is_edge, is_global, pooling))

    def forward(self, x, edge_index, edge_weight, batch_item, bs):
        init_lens = edge_index
        # x_pos, x_neg = torch.split(x.view(b, N, C), [P, Neg], dim = 1)
        centroid = torch.mean(x.view(bs, -1, self.in_dim), dim = 1)
        edge_prob = edge_index
        x_trans_loss = x
        for index in range(self.total_layer_num):
            # edge update
            if self.is_edge:
                edge_index, edge_weight, edge_prob, x_trans_loss = self.edgemodel(x, edge_index, edge_weight, centroid, batch_item, bs)
            
            # node update
            if self.is_node:
                x, saved_x = self.node_updates[index](x, edge_index, edge_weight)
            
            # system update
            if self.is_system:
                centroid = self.sys_updates[index](saved_x, centroid, bs)


        x_loss = self.mlp_head(x)
        centroid_loss = self.mlp_head(centroid)
        final_len = edge_index
        return x, edge_weight, centroid, x_loss, centroid_loss, edge_prob





class edgePredictor(nn.Module):
    def __init__(self, dim, is_global):
        super(edgePredictor, self).__init__()
        self.is_global = is_global
        self.dim = dim 
        if is_global:
            self.l2r = nn.Sequential(
                nn.Linear(3 * dim, dim),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(dim, 1)
                # nn.Sigmoid()
            )  
        else:
            self.l2r = nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(dim, 1)
                # nn.Sigmoid()
            )              

    def forward(self, node_features, edge_index, centroid, bs):
        node_features = node_features.view(-1, self.dim)
        node_j = node_features[edge_index[0]]
        node_i = node_features[edge_index[1]]
        if self.is_global:
            residual_node_features = (node_features.view(bs, -1, self.dim) - centroid.view(bs, 1, self.dim)).view(-1, self.dim)
            residual_node_j = residual_node_features[edge_index[0]]
            residual_node_i = residual_node_features[edge_index[1]]

            sim_vec = torch.cat((torch.abs(node_i - node_j), residual_node_i, residual_node_j), dim = 1)
        else:
            sim_vec = torch.abs(node_i - node_j)
        prob_score = self.l2r(sim_vec)
        
        return prob_score



class EdgeUpdate(nn.Module):
    def __init__(self, is_global, feature_dim, edge_dim, load_dir = None):
        super(EdgeUpdate, self).__init__()
        self.feature_dim = feature_dim
        self.edge_dim = edge_dim
        self.temp = 0.6
        self.thres_1 = torch.nn.Threshold(0.5, 0)
        self.thres_2 = torch.nn.Threshold(-0.49, 1)
        self.mins = torch.tensor(1e-10).cuda()
        self.relu_fuc = nn.ReLU()
        self.edge_skip_alpha = nn.Parameter(torch.rand(1))
        self.ep_net = edgePredictor(feature_dim, is_global)
            

    def forward(self, x, edge_index, edge_weight, centroid, batch_item, bs):
        pre_prob = self.ep_net(x, edge_index, centroid, bs).squeeze(-1)
        pre_adj = torch.sigmoid(pre_prob)
        sampled_edge = torch.ones([pre_adj.size(0)]).cuda()
        sampled_edge = pyro.distributions.RelaxedBernoulliStraightThrough(temperature=self.temp, probs = pre_adj).rsample()
        combine_weight = self.edge_skip_alpha * (sampled_edge * edge_weight) + (1-self.edge_skip_alpha) * (sampled_edge * pre_adj)

        return edge_index, combine_weight, pre_adj, x



class NodeUpdate(torch.nn.Module):
    def __init__(self, in_channel, out_channel, is_norm, layer_num):
        super(NodeUpdate, self).__init__()
        self.bn1 = torch.nn.BatchNorm1d(in_channel)
        self.conv1 = GINConv(
            nn.Sequential(nn.Linear(out_channel, out_channel), nn.ReLU()))
        self.drop_layer = nn.Dropout(0.5)


    def forward(self, x, edge_index, edge_weight = None):
        his_x = []
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = self.bn1(x)
        x = self.drop_layer(x)
        his_x.append(x)
        
        return x, his_x


class SystemUpdate(nn.Module):
    def __init__(self, in_channel, out_channel, pooling):
        super(SystemUpdate, self).__init__()
        self.in_dim = in_channel
        self.out_dim = out_channel
        self.pooling = pooling

    def forward(self, his_x, init_c, bs):
        mem_lens = len(his_x)
        if self.pooling == "memory":
            for index in range(mem_lens):
                tmp_mem = his_x[index].view(bs, -1, self.in_dim)
                tmp_score = torch.bmm(tmp_mem, init_c.view(bs, self.in_dim, 1)).view(bs, -1)
                tmp_att = F.softmax(tmp_score, dim = 1)
                tmp_read = torch.sum(tmp_att.view(bs, -1, 1) * tmp_mem, dim = 1)
                init_c = tmp_read
        elif self.pooling == "avg":
            x = his_x[-1].view(bs, -1, self.in_dim)
            init_c = torch.mean(x, dim = 1)
        elif self.pooling == "sum":
            x = his_x[-1].view(bs, -1, self.in_dim)
            init_c = torch.sum(x, dim = 1)
        elif self.pooling == "max":
            x = his_x[-1].view(bs, -1, self.in_dim)
            init_c = torch.max(x, dim = 1).values
        else:
            raise ValueError("No such pooling type!")
        return init_c

