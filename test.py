
import torch.utils.data as data
import glob

from utils import *
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
import sklearn

import numpy as np
import xml.etree.ElementTree as ET
import networkx as nx
import pdb
"""## Import libraries"""
import pagexml
import re
import torch 
import dgl
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from model import Net
from datasets import FUNSD,collate

testset = FUNSD('testing_data/annotations','')
test_loader = DataLoader(testset, batch_size=1, collate_fn=collate)
model = torch.load('model.pt')
recalls = []
precisions =[]
print ("Validation on test set")
aris = []

def test_grouping(bg,blab,model):
    print(float(iter)/len(test_loader),'\r\r')
    pdb.set_trace()
    prediction_groups,prediction_links = model(bg)
    prediction = prediction_groups
    prediction[prediction>0.8] = 1
    prediction[prediction<=0.8] = 0
    
    target_edges = blab[0]

    # convert target edges dict from complete graph to input graph edges 0s and 1s
    input_edges = torch.stack(bg.edges()).t().tolist()            
    edges = list(map(tuple, input_edges))
    target = torch.tensor([target_edges[e] for e in edges])

    rec = float(((prediction == target)[target.bool()].float().sum()/target.sum()).item())
    prec = float(((prediction == target)[prediction.bool()].float().sum()/prediction.sum()).item())


    # GT AND PRED COMPONENTS

    pred_edges = torch.t(torch.stack([bg.edges()[0][prediction.bool()],bg.edges()[1][prediction.bool()]]))
    predg = edges_list_to_dgl_graph(pred_edges)
    predg.ndata['position']=bg.ndata['position']
    
    pred_components = nx.connected_components(predg.to_networkx().to_undirected())
    
    target_edges = torch.t(torch.stack([bg.edges()[0][target.bool()],bg.edges()[1][target.bool()]]))
    yg = edges_list_to_dgl_graph(target_edges)
    yg.ndata['position']=bg.ndata['position']
    
    gt_components = nx.connected_components(yg.to_networkx().to_undirected())
    cluster_idx=0
    pred_node_labels=np.zeros(bg.number_of_nodes())
    all_nodes = []
    for node_cluster in pred_components:
        for node in node_cluster:
            all_nodes.append(node)
            pred_node_labels[node]=cluster_idx
        cluster_idx+=1
    cluster_idx=0
    gt_node_labels=np.zeros(bg.number_of_nodes())

    for node_cluster in gt_components:
        for node in node_cluster:
            gt_node_labels[node]=cluster_idx
        cluster_idx+=1
    
    ari = sklearn.metrics.adjusted_rand_score(gt_node_labels,pred_node_labels)


    return prec,rec,ari



for iter,(bg,blab,elab) in enumerate(test_loader):
    precision,recall,ari = test_grouping(bg,blab,model)
    recalls.append(recall)
    precisions.append(precision)
    aris.append(ari)

epoch_prec = np.mean(precisions)
epoch_rec = np.mean(recalls)

ari = np.mean(aris)

print('Precision, recall:',epoch_prec,epoch_rec)

print('F1',2*(epoch_prec*epoch_rec)/(epoch_prec+epoch_rec))


print('ARI',np.mean(aris))
