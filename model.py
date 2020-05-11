
import torch.utils.data as data
import glob

from random import randrange
from dgl.nn.pytorch import GATConv, GraphConv
import matplotlib.pyplot as plt
import dgl.function as fn

from torch import nn
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys

import numpy as np
import xml.etree.ElementTree as ET
import networkx as nx
import pdb
import pagexml
import re
import torch 
import dgl
from torch.optim.lr_scheduler import StepLR
from utils import *

class Net(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(Net, self).__init__()
        #self.conv1 = GraphConv(in_dim, hidden_dim)
        #self.conv2 = GraphConv(hidden_dim, hidden_dim)
        #self.conv3 = GraphConv(hidden_dim, hidden_dim)

        self.conv1 = GATConv(in_dim, hidden_dim, 4, residual=True, activation=F.relu)
        self.conv2 = GATConv(4*hidden_dim, hidden_dim, 4, residual=True, activation=F.relu)
        self.conv3 = GATConv(4*hidden_dim, hidden_dim, 4, residual=True)

        self.w_group_mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(True), nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.entity_linear = nn.Linear(hidden_dim,hidden_dim)
        self.entity_link_mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(True), nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.training = True
        self.thresh = 0.5
        self.entity_classify =nn.Sequential(nn.Linear(hidden_dim,4),nn.Sigmoid())



    def get_complete_graph_and_pairs(self,num_nodes):
        G = nx.complete_graph(num_nodes)
        
        g = dgl.DGLGraph()
        g.from_networkx(G)
        pairs = torch.t(torch.stack([g.edges()[0],g.edges()[1]]))
        return g,pairs

    def calc_score(self,g):
        #pairs = torch.t(torch.stack([g.edges()[0],g.edges()[1]]))

        # get hidden state of all source destination pairs and calculate their edge score
        s = g.ndata['h'][g.edges()[0]]
        d = g.ndata['h'][g.edges()[1]]
        #s = g.ndata['h'][pairs[:,0]]
        #d = g.ndata['h'][pairs[:,1]]
        score = self.w_group_mlp((s-d).abs()).squeeze()
        #score = F.sigmoid(torch.bmm(s.unsqueeze(1),d.unsqueeze(2)).squeeze())
        #score = self.score(s,d)

        return score

    def entity_link_score(self,pairs,entities):
        s = entities[pairs[:,0]]
        d = entities[pairs[:,1]]
        score = self.entity_link_mlp((s-d).abs()).squeeze()
        return score


    def forward(self, g,target=None):
        # For undirected graphs, in_degree is the same as
        # out_degree.
        #h = g.ndata['position']
        h = torch.cat([g.ndata['position'],g.ndata['w_embed']],dim=1)
        if torch.cuda.is_available():
          h = h.cuda() 
        #h = F.relu(self.conv1(g, h))
        #h = F.relu(self.conv2(g, h))
        #h = self.conv3(g, h)
        h = self.conv1(g,h)
        h = h.view(h.shape[0], -1)

        h = self.conv2(g,h)
        h = h.view(h.shape[0], -1)

        h = self.conv3(g,h)
        h = h.mean(1)
        g.ndata['h'] = h
        groups_score = self.calc_score(g)
        
        entity_states = []
        entity_positions = []
        if self.training: 
            for entity in range(int(torch.max(g.ndata['entity']))+1):
                entity_node_states = g.ndata['h'][g.ndata['entity']==entity]
                entity_state = self.entity_linear(torch.sum(entity_node_states,dim=0))

                entity_states.append(entity_state)
                entity_position = torch.mean(g.ndata['position'][g.ndata['entity']==entity],dim=0)
                entity_positions.append(entity_position)
            entity_states = torch.stack(entity_states)
            entity_positions = torch.stack(entity_positions)
            
            entity_graph = dgl.transform.knn_graph(entity_positions, 10)
        else:
            
            pred_edges = torch.t(torch.stack([g.edges()[0][groups_score>0.5],g.edges()[1][groups_score>0.5]]))
            predg = edges_list_to_dgl_graph(pred_edges)
            predg.ndata['position']=g.ndata['position']
        
            components = nx.connected_components(predg.to_networkx().to_undirected())
       
            entity_states=[]
            for component in components:
                component_node_indices = [node for node in component]
                entity_node_states= g.ndata['h'][component_node_indices]
                entity_state = self.entity_linear(torch.sum(entity_node_states,dim=0))
                entity_states.append(entity_state)
                
                entity_position = torch.mean(g.ndata['position'][component_node_indices],dim=0)
                entity_positions.append(entity_position)
        
            entity_states = torch.stack(entity_states)
            entity_positions = torch.stack(entity_positions)
            k = min(entity_positions.shape[0],10)
            entity_graph = dgl.transform.knn_graph(entity_positions, k)
        
        #_,entity_pairs = self.get_complete_graph_and_pairs(len(entity_states))
        entity_pairs = torch.t(torch.stack([entity_graph.edges()[0],entity_graph.edges()[1]]))

        entity_link_score = self.entity_link_score(entity_pairs,entity_states)
        
        entity_class = self.entity_classify(entity_states)
        return groups_score,entity_class,entity_positions,entity_link_score
