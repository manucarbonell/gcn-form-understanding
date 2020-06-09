
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
def test(test_data_dir,model):
    testset = FUNSD(test_data_dir,'')
    test_loader = DataLoader(testset, batch_size=1, collate_fn=collate)
    recalls = []
    precisions =[]
    print ("Validation on test set")
    aris = []
    labeling_f1 = []
    linking_f1 = []
    precisions=[]
    recalls=[]
    for iter, (input_graph, link_labels) in enumerate(test_loader):
        print('Testing... ',100*float(iter)/len(test_loader),'%',end='\r')
        entity_link_score = model(input_graph)

        link_prec,link_rec= test_linking(entity_link_score,link_labels,threshold= float(entity_link_score.mean()))
        print ("prec ",link_prec," recall",link_rec)
        if (link_prec>0 and link_rec>0):
            linking_f1.append(2*(link_prec*link_rec)/(link_prec+link_rec))
            precisions.append(link_prec)
            recalls.append(link_rec)
        else:
            linking_f1.append(0)

    print('Linking F1',np.mean(linking_f1))
    print('Precision',np.mean(precisions))
    print('Recall',np.mean(recalls))
    linking_f1 = np.mean(linking_f1)
    return  linking_f1 
def main():
    model = torch.load('model.pt')

    test_data_dir = 'testing_data/annotations'
    linking_f1 = test(test_data_dir,model)


if __name__=='__main__':
    main()
