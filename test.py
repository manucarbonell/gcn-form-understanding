
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
from evaluate import *

testset = FUNSD('testing_data/annotations','')
test_loader = DataLoader(testset, batch_size=1, collate_fn=collate)
model = torch.load('model.pt')
recalls = []
precisions =[]
print ("Validation on test set")
aris = []
labeling_f1 = []
for iter, (input_graph, group_labels,entity_labels,link_labels) in enumerate(test_loader):
    group_prediction,entity_class,entity_position,entity_link_score = model(input_graph,group_labels)

    group_prediction[group_prediction>model.thres]=1.
    precision,recall,ari = test_grouping(input_graph,group_prediction,group_labels[0])

    label_prec,label_rec =test_labeling(input_graph,entity_class,entity_position,entity_labels)
    recalls.append(recall)
    precisions.append(precision)
    aris.append(ari)

    labeling_f1.append(2*(label_prec*label_rec)/(label_prec+label_rec))

epoch_prec = np.mean(precisions)
epoch_rec = np.mean(recalls)

ari = np.mean(aris)

print('Grouping precision, recall & F1:',epoch_prec,epoch_rec,2*(epoch_prec*epoch_rec)/(epoch_prec+epoch_rec))

print('ARI',np.mean(aris))
print('Labeling F1',np.mean(labeling_f1))
