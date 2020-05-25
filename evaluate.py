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

def test_grouping(bg,prediction,target,thres=0.4):
    prediction[prediction>thres]=1
    prediction[prediction<thres]=0
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


def test_linking(link_scores,link_labels,threshold=0.5):
    link_labels = link_labels[0]
    link_scores[link_scores>threshold]=1
    link_scores[link_scores<threshold]=0
    if link_scores.sum()>0:
        precision  = (link_scores[link_labels==1]).sum()/link_scores.sum()
    else:
        precision = 0
    if link_labels.sum()>0:

        recall  = (link_scores[link_labels==1]).sum()/link_labels.sum()
    else:
        recall= 0
    return float(precision), float(recall)

def test_labeling(entity_class,entity_labels,threshold=0.5):
    labels = entity_labels[0][:,0]
    entity_class = torch.argmax(entity_class,dim=-1)
    true_positives = float((entity_class==labels).sum())
    
    total = int(labels.numel())
    acc =(true_positives/total)
    return acc,acc
