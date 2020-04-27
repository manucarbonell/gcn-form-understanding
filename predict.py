
import torch.utils.data as data
import glob
import random
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
from utils import visualize_graph,edges_list_to_dgl_graph
import utils
from model import Net
from datasets import FUNSD,collate,pxml2graph

testset = FUNSD('funsd_test','')
test_loader = DataLoader(testset, batch_size=1, collate_fn=collate)
model = torch.load('model.pt')
accuracies = []
precisions =[]
number_of_colors = 10
color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                     for i in range(number_of_colors)]
out_dir = 'predictions_out'

print ("Validation on test set")
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
for iter,(bg,blab) in enumerate(test_loader):
    print(float(iter)/len(test_loader),'\r\r')
    prediction = model(bg)
    prediction[prediction>0.5] = 1
    prediction[prediction<=0.5] = 0
    page_id = testset.files[iter].split('/')[-1].split('.')[-2]
    target_edges = blab[0]
    # convert target edges dict from complete graph, to input graph edges 0s and 1s
    input_edges = torch.stack(bg.edges()).t().tolist()            
    edges = list(map(tuple, input_edges))
    target = torch.tensor([target_edges[e] for e in edges])
    # X
    #visualize_graph(bg,os.path.join(out_dir,page_id+'_in.png'))
    
    #VISUALIZE FP  EDGES 
    fp_edges = torch.t(torch.stack([bg.edges()[0][prediction>target],bg.edges()[1][prediction>target]]))
    fpg = edges_list_to_dgl_graph(fp_edges,bg.number_of_nodes())
    fpg.ndata['position']=bg.ndata['position']
    fp_edges = {k for k in dict(fpg.to_networkx().edges()).keys()}
    nx.draw_networkx_edges(fpg.to_networkx(),pos=np.array(fpg.ndata['position']),arrows = False,width = 3.0,edgelist = fp_edges,edge_color='r')
    #visualize_graph(fpg,os.path.join(out_dir,page_id+'_fp.png'))
    
    #VISUALIZE FN  EDGES 
    fn_edges = torch.t(torch.stack([bg.edges()[0][prediction<target],bg.edges()[1][prediction<target]]))
    fng = edges_list_to_dgl_graph(fn_edges,bg.number_of_nodes())
    fng.ndata['position']=bg.ndata['position']
    fn_edges = {k for k in dict(fng.to_networkx().edges()).keys()}
    nx.draw_networkx_edges(fng.to_networkx(),pos=np.array(fng.ndata['position']),arrows = False,width = 3.0,edgelist = fn_edges,edge_color='b')
    #visualize_graph(fpg,os.path.join(out_dir,page_id+'_fp.png'))


    # Y'
    pred_edges = torch.t(torch.stack([bg.edges()[0][prediction.bool()],bg.edges()[1][prediction.bool()]]))
    predg = edges_list_to_dgl_graph(pred_edges)
    predg.ndata['position']=bg.ndata['position']
    visualize_graph(predg,os.path.join(out_dir,page_id+'_pred.png'))
    plt.clf()
    
    # Y
    target_edges = torch.t(torch.stack([bg.edges()[0][target.bool()],bg.edges()[1][target.bool()]]))
    yg = edges_list_to_dgl_graph(target_edges)
    yg.ndata['position']=bg.ndata['position']
    print('GT connected components',nx.number_connected_components(yg.to_networkx().to_undirected()))
    visualize_graph(yg,os.path.join(out_dir,page_id+'_gt.png'))
    plt.clf()

    print('Pred strongly connected components',nx.number_strongly_connected_components(predg.to_networkx()))
    print('Pred connected components',nx.number_connected_components(predg.to_networkx().to_undirected()))
    print('Pred strongly connected components recursive',len([c for c in nx.strongly_connected_components_recursive(predg.to_networkx())]))
    
    #Spectral clustering on Y'
    '''clusters = utils.spectral_clustering(np.array(predg.adjacency_matrix().to_dense()))
    pdb.set_trace() 
    nx.draw_networkx_nodes(predg.to_networkx(),pos = np.array(predg.ndata['position']),node_color=[color[i] for i in clusters])
    plt.show()
    '''
