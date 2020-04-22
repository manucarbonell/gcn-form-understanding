
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
"""## Import libraries"""
import pagexml
import re
import torch 
import dgl
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from model import Net
from datasets import FUNSD,collate,pxml2graph

testset = FUNSD('funsd_test','')
test_loader = DataLoader(testset, batch_size=1, collate_fn=collate)
model = torch.load('model.pt')
accuracies = []
print ("Validation on test set")

for iter,(bg,blab) in enumerate(test_loader):
    print(float(iter)/len(test_loader),'\r\r')
    prediction = model(bg)
    prediction[prediction>0.5] = 1
    prediction[prediction<=0.5] = 0
    
    target_edges = blab[0]

    # convert target edges dict from complete graph to input graph edges 0s and 1s
    input_edges = torch.stack(bg.edges()).t().tolist()            
    edges = list(map(tuple, input_edges))
    target = torch.tensor([target_edges[e] for e in edges])

    acc = float(((prediction == target)[target.bool()].float().sum()/target.sum()).item())
    accuracies.append(acc)

epoch_acc = np.mean(accuracies)
print('TOTAL ACC',epoch_acc)
