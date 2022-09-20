
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
import re
import torch 
import dgl
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from utils import visualize_graph,edges_list_to_dgl_graph
import utils
from model import Net
from datasets import FUNSD,collate

testset = FUNSD('testing_data/annotations','')
page_images_dir = 'testing_data/images'
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
    target =target_edges# torch.tensor([target_edges[e] for e in edges])
    # X
    #visualize_graph(bg,os.path.join(out_dir,page_id+'_in.png'))

    # VISUALIZE PAGE
    bkg_im_path = os.path.join(page_images_dir,page_id+'.png')
    bkg_img=plt.imread(bkg_im_path)
    plt.imshow(1-bkg_img,cmap='Greys')
    img_h = bkg_img.shape[0]

    img_w = bkg_img.shape[1]

    #VISUALIZE FP  EDGES 
    fp_edges = torch.t(torch.stack([bg.edges()[0][(prediction==1)*(target==0)],bg.edges()[1][(prediction==1)*(target==0)]]))
    fpg = edges_list_to_dgl_graph(fp_edges,bg.number_of_nodes())
    fpg.ndata['position']=bg.ndata['position']
    fp_edges = {k for k in dict(fpg.to_networkx().edges()).keys()}
    nx.draw_networkx_edges(fpg.to_networkx(),pos=np.array(fpg.ndata['position']),arrows = False,width = 3.0,edgelist = fp_edges,edge_color='r')
    visualize_graph(g=fpg,scale_x = img_w,scale_y=img_h,edge_color = 'r')
    
    #VISUALIZE FN  EDGES 
    fn_edges = torch.t(torch.stack([bg.edges()[0][(prediction==0)*(target==1)],bg.edges()[1][(prediction==0)*(target==1)]]))
    fng = edges_list_to_dgl_graph(fn_edges,bg.number_of_nodes())
    fng.ndata['position']=bg.ndata['position']
    fn_edges = {k for k in dict(fng.to_networkx().edges()).keys()}
    nx.draw_networkx_edges(fng.to_networkx(),pos=np.array(fng.ndata['position']),arrows = False,width = 3.0,edgelist = fn_edges,edge_color='b')
    visualize_graph(fng,scale_x=img_w,scale_y=img_h,edge_color='b')
    
    #VISUALIZE TP  EDGES 
    tp_edges = torch.t(torch.stack([bg.edges()[0][(prediction==1)*(target==1)],bg.edges()[1][(prediction==1)*(target==1)]]))
    tpg = edges_list_to_dgl_graph(tp_edges,bg.number_of_nodes())
    tpg.ndata['position']=bg.ndata['position']
    tp_edges = {k for k in dict(tpg.to_networkx().edges()).keys()}
    nx.draw_networkx_edges(tpg.to_networkx(),pos=np.array(fpg.ndata['position']),arrows = False,width = 3.0,edgelist = tp_edges,edge_color='g')
    visualize_graph(tpg,scale_x=img_w,scale_y=img_h,edge_color='g',im_out_path = os.path.join(out_dir, page_id+'_pred.png'))


    # Y'
    pred_edges = torch.t(torch.stack([bg.edges()[0][prediction.bool()],bg.edges()[1][prediction.bool()]]))
    predg = edges_list_to_dgl_graph(pred_edges,bg.number_of_nodes())
    predg.ndata['position']=bg.ndata['position']
    #visualize_graph(g=predg,im_out_path = os.path.join(out_dir, page_id+'_pred.png'))
    plt.clf()
    ''' 
    # Y
    target_edges = torch.t(torch.stack([bg.edges()[0][target.bool()],bg.edges()[1][target.bool()]]))
    yg = edges_list_to_dgl_graph(target_edges,bg.number_of_nodes())
    yg.ndata['position']=bg.ndata['position']
    print('GT connected components',nx.number_connected_components(yg.to_networkx().to_undirected()))
    visualize_graph(yg,os.path.join(out_dir,page_id+'_gt.png'))
    plt.clf()
    pred_components = nx.connected_components(predg.to_networkx().to_undirected())
    gt_components = nx.connected_components(yg.to_networkx().to_undirected())
    print('Pred strongly connected components',nx.number_strongly_connected_components(predg.to_networkx()))
    #print('GT strongly connected components',nx.number_strongly_connected_components(yg.to_networkx()))
    #Spectral clustering on Y'
    clusters = utils.spectral_clustering(np.array(predg.adjacency_matrix().to_dense()))
    nx.draw_networkx_nodes(predg.to_networkx(),pos = np.array(predg.ndata['position']),node_color=[color[i] for i in clusters])
    plt.show()
    '''
