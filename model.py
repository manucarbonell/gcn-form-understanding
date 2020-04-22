
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

class Net(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(Net, self).__init__()
        #self.conv1 = GraphConv(in_dim, hidden_dim)
        #self.conv2 = GraphConv(hidden_dim, hidden_dim)
        #self.conv3 = GraphConv(hidden_dim, hidden_dim)

        self.conv1 = GATConv(in_dim, hidden_dim, 4, residual=True, activation=F.relu)
        self.conv2 = GATConv(4*hidden_dim, hidden_dim, 4, residual=True, activation=F.relu)
        self.conv3 = GATConv(4*hidden_dim, hidden_dim, 4, residual=True)

        self.mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(True), nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.score = nn.CosineSimilarity()

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
        score = self.mlp((s-d).abs()).squeeze()
        #score = F.sigmoid(torch.bmm(s.unsqueeze(1),d.unsqueeze(2)).squeeze())
        #score = self.score(s,d)

        return score

    def forward(self, g):
        # For undirected graphs, in_degree is the same as
        # out_degree.
        h = g.ndata['position']

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
        score = self.calc_score(g)
        #hg = dgl.mean_nodes(g, 'h')
        #return self.classify(hg)
        return score
