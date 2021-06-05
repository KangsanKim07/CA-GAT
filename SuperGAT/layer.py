import random
from multiprocessing import Process, Manager
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score, average_precision_score
from termcolor import cprint
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, dropout_adj, \
    is_undirected, accuracy, negative_sampling, batched_negative_sampling, to_undirected, to_dense_adj
import torch_geometric.nn.inits as tgi
from visualize import plot_graph_layout
from sknetwork.ranking import PageRank
from sknetwork.visualization import svg_graph

from typing import List

from utils import np_sigmoid
import os


def is_pretraining(current_epoch, pretraining_epoch):
    return current_epoch is not None and pretraining_epoch is not None and current_epoch < pretraining_epoch

edge_index_idx = []
ppr_score_list = []
ppr_score_dense = []
edge_index_list = []
ppr = []


class SuperGAT(MessagePassing):

    def __init__(self, y, in_channels, out_channels, heads=1, concat=True, negative_slope=0.2, dropout=0, bias=True,
                 is_super_gat=True, attention_type="basic", super_gat_criterion=None,
                 neg_sample_ratio=0.0, edge_sample_ratio=1.0,
                 pretraining_noise_ratio=0.0, use_pretraining=False,
                 to_undirected_at_neg=False, scaling_factor=None,
                 cache_label=False, cache_attention=False, **kwargs):
        super(SuperGAT, self).__init__(aggr='add', node_dim=0, **kwargs)
        self.y = y
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.is_super_gat = is_super_gat
        self.attention_type = attention_type
        self.super_gat_criterion = super_gat_criterion
        self.neg_sample_ratio = neg_sample_ratio
        self.edge_sample_ratio = edge_sample_ratio
        self.pretraining_noise_ratio = pretraining_noise_ratio
        self.pretraining = None if not use_pretraining else True
        self.to_undirected_at_neg = to_undirected_at_neg
        self.cache_label = cache_label
        self.cache_attention = cache_attention

        self.weight = Parameter(torch.Tensor(in_channels, heads * out_channels))

        if self.is_super_gat:

            if self.attention_type == "gat_originated":  # GO
                self.att_mh_1 = Parameter(torch.Tensor(1, heads, 2 * out_channels))

            elif self.attention_type == "dot_product":  # DP
                pass

            elif self.attention_type == "scaled_dot_product":  # SD
                self.scaling_factor = scaling_factor or np.sqrt(self.out_channels)

            elif self.attention_type.endswith("mask_only"):  # MX
                self.att_mh_1 = Parameter(torch.Tensor(1, heads, 2 * out_channels))

            else:
                raise ValueError

        else:
            if self.attention_type.endswith("gat_originated") or self.attention_type == "basic":
                self.att_mh_1 = Parameter(torch.Tensor(1, heads, 2 * out_channels))

            elif self.attention_type.endswith("dot_product"):
                pass

            else:
                raise ValueError

        self.cache = {
            "num_updated": 0,
            "att": None,  # Use only when self.cache_attention == True for task_type == "Attention_Dist"
            "att_with_negatives": None,  # Use as X for supervision.
            "att_label": None,  # Use as Y for supervision.
        }

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        tgi.glorot(self.weight)
        tgi.zeros(self.bias)
        for name, param in self.named_parameters():
            if name.startswith("att_scaling"):
                tgi.ones(param)
            elif name.startswith("att_bias"):
                tgi.zeros(param)
            elif name.startswith("att_mh"):
                tgi.glorot(param)

    def forward(self, x, edge_index, size=None, batch=None,
                neg_edge_index=None, attention_edge_index=None):
        """
        :param x: [N, F]
        :param edge_index: [2, E]
        :param size:
        :param batch: None or [B]
        :param neg_edge_index: When using explicitly given negative edges.
        :param attention_edge_index: [2, E'], Use for link prediction
        :return:
        """
        # x1 = x.clone().detach()
        if self.pretraining and self.pretraining_noise_ratio > 0.0:
            edge_index, _ = dropout_adj(edge_index, p=self.pretraining_noise_ratio,
                                        force_undirected=is_undirected(edge_index),
                                        num_nodes=x.size(0), training=self.training)

        if size is None and torch.is_tensor(x):
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # [N, F0] * [F0, heads * F] = [N, heads * F]
        x = torch.matmul(x, self.weight)
        x = x.view(-1, self.heads, self.out_channels)

        #Personalized PageRank
        global edge_index_list, ppr_score_list, edge_index_idx, ppr_score_dense
        if torch.sum(edge_index).item() in edge_index_idx:
          index = edge_index_idx.index(torch.sum(edge_index).item())
          # ppr_scores = ppr_score_list[index]
          edge_index = edge_index_list[index]
          
        else:
          adj = to_dense_adj(edge_index).squeeze()
          adj = adj.cpu().numpy()
          ppr_scores = np.zeros(adj.shape)

          def work(id, return_dict, start, end, adj):
            pagerank = PageRank()
            for i in range(start, end):
              if (i%500 == 0):
                print(i,"th node")
              seeds = {i: 1}
              scores = pagerank.fit_transform(adj, seeds)
              return_dict[i] = scores 
          
          manager = Manager()
          return_dict = manager.dict()
          jobs = []
          
          for i in range (10):
            p = Process(target=work, args=(i, return_dict, i * adj.shape[0]//10, (i+1) * adj.shape[0]//10, adj))
            jobs.append(p)
            p.start()
            if i == 9:
              i = 10
              p = Process(target=work, args=(i, return_dict, i * adj.shape[0]//10, adj.shape[0], adj))
          
          for proc in jobs:
            proc.join()

          return_dict = dict(sorted(return_dict.items()))
          return_dict = list(return_dict.values())
          ppr_scores = np.array(return_dict)
          # ppr_scores = np.reshape(ppr_scores, adj.shape)
          ppr_scores = torch.from_numpy(ppr_scores)
          ppr_score_dense.append(ppr_scores)
          ppr_scores = (ppr_scores>0.05).int()
          ppr_scores = sp.coo_matrix(ppr_scores)
          edge_index_idx.append(torch.sum(edge_index).item())
          ppr_score_list.append(ppr_scores)
          del adj
        
          cuda = torch.device('cuda:0')
          for i in range(ppr_scores.row.shape[0]):
            p1 = ppr_scores.row[i]
            p2 = ppr_scores.col[i]
            
            idx = (edge_index[0] == p1).nonzero()

            for j in range(idx.shape[0]):
              if (edge_index[1][idx[j]] == p2):
                break
              elif j == idx.shape[0] - 1:
                idx2 = (edge_index[1] == p1).nonzero()
                for k in range(idx2.shape[0]):
                  if (edge_index[0][idx2[k]] == p2):
                    break
                  elif k == idx2.shape[0] - 1:
                    edge_index = torch.cat((edge_index, torch.Tensor([[p1], [p2]]).long().cuda()), 1)
          edge_index_list.append(edge_index)
        propagated = self.propagate(edge_index, size=size, x=x)

        if (self.is_super_gat and self.training) or (attention_edge_index is not None) or (neg_edge_index is not None):

            device = next(self.parameters()).device
            num_pos_samples = int(self.edge_sample_ratio * edge_index.size(1))
            num_neg_samples = int(self.neg_sample_ratio * self.edge_sample_ratio * edge_index.size(1))

            if attention_edge_index is not None:
                neg_edge_index = None

            elif neg_edge_index is not None:
                pass

            elif batch is None:
                if self.to_undirected_at_neg:
                    edge_index_for_ns = to_undirected(edge_index, num_nodes=x.size(0))
                else:
                    edge_index_for_ns = edge_index
                neg_edge_index = negative_sampling(
                    edge_index=edge_index_for_ns,
                    num_nodes=x.size(0),
                    num_neg_samples=num_neg_samples,
                )
            else:
                neg_edge_index = batched_negative_sampling(
                    edge_index=edge_index,
                    batch=batch,
                    num_neg_samples=num_neg_samples,
                )

            if self.edge_sample_ratio < 1.0:
                pos_indices = random.sample(range(edge_index.size(1)), num_pos_samples)
                pos_indices = torch.tensor(pos_indices).long().to(device)
                pos_edge_index = edge_index[:, pos_indices]
            else:
                pos_edge_index = edge_index

            att_with_negatives = self._get_attention_with_negatives(
                x=x,
                edge_index=pos_edge_index,
                neg_edge_index=neg_edge_index,
                total_edge_index=attention_edge_index,
            )  # [E + neg_E, heads]

            # Labels
            if self.training and (self.cache["att_label"] is None or not self.cache_label):
                att_label = torch.zeros(att_with_negatives.size(0)).float().to(device)
                att_label[:pos_edge_index.size(1)] = 1.
            elif self.training and self.cache["att_label"] is not None:
                att_label = self.cache["att_label"]
            else:
                att_label = None
            self._update_cache("att_label", att_label)
            self._update_cache("att_with_negatives", att_with_negatives)
        
        # plot_graph_layout(x1.cpu().detach().numpy(), self.y.cpu().detach().numpy(), edge_index.cpu().detach().numpy(), edge_to_attention=None)
        return propagated

    def message(self, edge_index_i, x_i, x_j, size_i):
      """
      :param edge_index_i: [E]
      :param x_i: [E, heads * F]
      :param x_j: [E, heads * F]
      :param size_i: N
      :return: [E, heads, F]
      """
      x_j = x_j.view(-1, self.heads, self.out_channels)  # [E, heads, F]
      if x_i is not None:
          x_i = x_i.view(-1, self.heads, self.out_channels)  # [E, heads, F]

      # Compute attention coefficients. [E, heads]
      alpha = self._get_attention(edge_index_i, x_i, x_j, size_i)
      if self.cache_attention:
          self._update_cache("att", alpha)

      # Sample attention coefficients stochastically.
      alpha = F.dropout(alpha, p=self.dropout, training=self.training)

      # [E, heads, F] * [E, heads, 1] = [E, heads, F]
      return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        """
        :param aggr_out: [N, heads, F]
        :return: [N, heads * F]
        """
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def _get_attention(self, edge_index_i, x_i, x_j, size_i, normalize=True, with_negatives=False,
                       **kwargs) -> torch.Tensor:
        """
        :param edge_index_i: [E]
        :param x_i: [E, heads, F]
        :param x_j: [E, heads, F]
        :param size_i: N
        :return: [E, heads]
        """
        # Compute attention coefficients.
        if self.attention_type == "basic" or self.attention_type.endswith("gat_originated"):
            # [E, heads, 2F] * [1, heads, 2F] -> [E, heads]
            alpha = torch.einsum("ehf,xhf->eh",
                                 torch.cat([x_i, x_j], dim=-1),
                                 self.att_mh_1)

        elif self.attention_type == "scaled_dot_product":
            alpha = torch.einsum("ehf,ehf->eh", x_i, x_j) / self.scaling_factor

        elif self.attention_type == "dot_product":
            # [E, heads, F] * [E, heads, F] -> [E, heads]
            alpha = torch.einsum("ehf,ehf->eh", x_i, x_j)

        elif "mask" in self.attention_type:

            # [E, heads, F] * [E, heads, F] -> [E, heads]
            logits = torch.einsum("ehf,ehf->eh", x_i, x_j)

            if self.attention_type.endswith("scaling"):
                logits = logits / self.att_scaling

            if with_negatives:
                return logits

            # [E, heads, 2F] * [1, heads, 2F] -> [E, heads]
            alpha = torch.einsum("ehf,xhf->eh",
                                 torch.cat([x_i, x_j], dim=-1),
                                 self.att_mh_1)
            alpha = torch.einsum("eh,eh->eh", alpha, torch.sigmoid(logits))

        else:
            raise ValueError

        if normalize:
            # global ppr_score_dense, edge_index_list, ppr
            # if len(ppr) != 0:
            #   ppr_scores = ppr[0]
            # else:
            #   ppr_scores = ppr_score_dense[0].cuda()
            #   edge_index = edge_index_list[0].cuda()
            #   adj = to_dense_adj(edge_index).squeeze().cuda()
            #   ppr_scores = torch.mul(adj, ppr_scores)
            #   ppr_scores = sp.coo_matrix(ppr_scores.cpu()).data
            #   ppr_scores = torch.Tensor(ppr_scores).cuda()
            #   ppr.append(ppr_scores)
            alpha = F.leaky_relu(alpha, self.negative_slope)
            # alpha = torch.t(torch.add(torch.t(alpha), ppr_scores))
            alpha = softmax(alpha, edge_index_i, num_nodes=size_i)

        return alpha

    def _get_attention_with_negatives(self, x, edge_index, neg_edge_index, total_edge_index=None):
        """
        :param x: [N, heads * F]
        :param edge_index: [2, E]
        :param neg_edge_index: [2, neg_E]
        :param total_edge_index: [2, E + neg_E], if total_edge_index is given, use it.
        :return: [E + neg_E, heads]
        """

        if neg_edge_index is not None and neg_edge_index.size(1) <= 0:
            neg_edge_index = torch.zeros((2, 0, self.heads))

        if total_edge_index is None:
            total_edge_index = torch.cat([edge_index, neg_edge_index], dim=-1)  # [2, E + neg_E]

        total_edge_index_j, total_edge_index_i = total_edge_index  # [E + neg_E]
        x_i = torch.index_select(x, 0, total_edge_index_i)  # [E + neg_E, heads * F]
        x_j = torch.index_select(x, 0, total_edge_index_j)  # [E + neg_E, heads * F]
        size_i = x.size(0)  # N

        x_j = x_j.view(-1, self.heads, self.out_channels)  # [E + neg_E, heads, F]
        if x_i is not None:
            x_i = x_i.view(-1, self.heads, self.out_channels)  # [E + neg_E, heads, F]

        alpha = self._get_attention(total_edge_index_i, x_i, x_j, size_i,
                                    normalize=False, with_negatives=True)
        return alpha

    def __repr__(self):
        return '{}({}, {}, heads={}, concat={}, att_type={}, nsr={}, pnr={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.heads, self.concat, self.attention_type,
            self.neg_sample_ratio, self.pretraining_noise_ratio
        )

    def _update_cache(self, key, val):
        self.cache[key] = val
        self.cache["num_updated"] += 1

    @staticmethod
    def get_supervised_attention_loss(model, criterion=None):

        loss_list = []
        cache_list = [(m, m.cache) for m in model.modules() if m.__class__.__name__ == SuperGAT.__name__]

        criterion = nn.BCEWithLogitsLoss() if criterion is None else eval(criterion)
        for i, (module, cache) in enumerate(cache_list):
            # Attention (X)
            att = cache["att_with_negatives"]  # [E + neg_E, heads]
            # Labels (Y)
            label = cache["att_label"]  # [E + neg_E]

            att = att.mean(dim=-1)  # [E + neg_E]
            loss = criterion(att, label)
            loss_list.append(loss)

        return sum(loss_list)

    @staticmethod
    def mix_supervised_attention_loss_with_pretraining(loss, model, mixing_weight,
                                                       criterion=None,
                                                       current_epoch=None, pretraining_epoch=None):
        if mixing_weight == 0:
            return loss

        current_pretraining = is_pretraining(current_epoch, pretraining_epoch)
        next_pretraining = is_pretraining(current_epoch + 1, pretraining_epoch)

        for m in model.modules():
            if m.__class__.__name__ == SuperGAT.__name__:
                current_pretraining = current_pretraining if m.pretraining is not None else None
                m.pretraining = next_pretraining if m.pretraining is not None else None

        if (current_pretraining is None) or (not current_pretraining):
            w1, w2 = 1.0, mixing_weight  # Forbid pre-training or normal-training
        else:
            w1, w2 = 0.0, 1.0  # Pre-training

        loss = w1 * loss + w2 * SuperGAT.get_supervised_attention_loss(
            model=model,
            criterion=criterion,
        )
        return loss

    @staticmethod
    def get_link_pred_perfs_by_attention(model, edge_y, layer_idx=-1, metric="roc_auc"):
        """
        :param model: GNN model (nn.Module)
        :param edge_y: [E_pred] tensor
        :param layer_idx: layer idx of GNN models
        :param metric: metric for perfs
        :return:
        """
        cache_list = [m.cache for m in model.modules() if m.__class__.__name__ == SuperGAT.__name__]
        cache_of_layer_idx = cache_list[layer_idx]

        att = cache_of_layer_idx["att_with_negatives"]  # [E + neg_E, heads]
        att = att.mean(dim=-1)  # [E + neg_E]

        edge_probs, edge_y = np_sigmoid(att.cpu().numpy()), edge_y.cpu().numpy()

        perfs = None
        if metric == "roc_auc":
            perfs = roc_auc_score(edge_y, edge_probs)
        elif metric == "average_precision":
            perfs = average_precision_score(edge_y, edge_probs)
        elif metric == "accuracy":
            perfs = accuracy(edge_probs, edge_y)
        else:
            ValueError("Inappropriate metric: {}".format(metric))
        return perfs

    def get_attention_dist(self, edge_index: torch.Tensor, num_nodes: int) -> List[torch.Tensor]:
        """
        :param edge_index: tensor the shape of which is [2, E]
        :param num_nodes: number of nodes
        :return: Tensor list L the length of which is N.
            L[i] = a_ji for e_{ji} \in {E}
                - a_ji = normalized attention coefficient of e_{ji} (shape: [heads, #neighbors])
        """
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)  # [2, E]

        att = self.cache["att"]  # [E, heads]

        att_dist_list = []
        for node_idx in range(num_nodes):
            att_neighbors = att[edge_index[1] == node_idx, :].t()  # [heads, #neighbors]
            att_dist_list.append(att_neighbors)

        return att_dist_list
