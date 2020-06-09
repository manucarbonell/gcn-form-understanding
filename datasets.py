import json
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
import fasttext

from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from utils import *



def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs,entity_links = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    #labels = dgl.batch(labels)
    return batched_graph,entity_links#torch.tensor(labels)
    #return batched_graph, group_labels,entity_labels,entity_links#torch.tensor(labels)


class FUNSD(data.Dataset):
    """PagesDataset
    """
  
    def __init__(self, root_path, file_list):
        self.root = root_path
        self.file_list = file_list

        # List of files and corresponding labels
        self.files =glob.glob(root_path+'/*.json')
                    #os.listdir(root_path)
        
        self.unique_labels = ['question','answer','header','other']
        if not os.path.exists('embeddings.bin'):
            self.embeddings = fasttext.train_unsupervised('text_data.txt', model='skipgram')
            self.embeddings.save_model("embeddings.bin")
        else:
            self.embeddings =fasttext.load_model("embeddings.bin")

    def __getitem__(self, index):
        # Read the graph and label
        G,entity_links =self.read_annotations(self.files[index])
        #G,group_labels,entity_labels,entity_links =self.read_annotations(self.files[index])

        # Convert to DGL format
        #node_label = torch.stack([torch.tensor(v['position']) for k,v in G.nodes.items()]).float()
        
        #node_word = torch.stack([torch.tensor(v['w_embed']) for k,v in G.nodes.items()]).float()
        #node_entity = torch.stack([torch.tensor(v['entity']) for k,v in G.nodes.items()]).float()
        # ENSURE BIDIRECTED
        g_in=G
        #g_in = dgl.to_bidirected(G)

        #g_in.ndata['w_embed'] =node_word.float()
        #g_in.ndata['entity'] = node_entity
   
        return g_in,entity_links
        #return g_in,group_labels,entity_labels,entity_links

    def label2class(self, label):
        # Converts the numeric label to the corresponding string
        return self.unique_labels[label]

    def class2label(self,c):
        label = self.unique_labels.index(c)
        return label

    def __len__(self):
        # Subset length
        return len(self.files)
    
    def read_annotations(self,json_file):
        # Input: json file path with page ground truth
        # Output:   - Graph to be given as input to the network
        #           - Dictionary with target edge predictions over input graph
        #           - List of entity links
        form_id = json_file.split('/')[-1].split('.')[0]
        partition = json_file.split('/')[-3]
        image_file = os.path.join(partition,'images',form_id+'.png')
        im = plt.imread(image_file)
        
        image_h,image_w= im.shape
        with open(json_file) as f:
            data = json.load(f)
        form_data = data['form']
        
        node_position={}
        node_text ={}
        node_shape = {}
        node_entity = {}
        word_idx=0
        entity_idx = 0
        entity_shapes = []
        entity_links=[]
        entity_positions=[]
        entity_labels = []
        entity_embeddings =[]
        # Get total amount of words in the form and their attr to create am.
        for entity in form_data:
            for link in entity['linking']:
                if link not in entity_links and [link[1],link[0]] not in entity_links:
                    entity_links.append(link) 
            for word in entity['words']:
                node_position[word_idx]=word['box'][:2]
                node_text[word_idx]=self.embeddings[word['text']]
                node_entity[word_idx]=entity_idx
                word_idx+=1
            
            entity_embeddings.append(self.embeddings[entity['text']])
            
            entity_position = np.array(entity['box'][:2])
            entity_positions.append(entity_position)
           
            entity_shape  = np.array([entity['box'][2] - entity['box'][0],entity['box'][3] - entity['box'][1]])
            entity_shapes.append(entity_shape)

            entity_labels.append((self.class2label(entity['label'])))
            
            entity_idx+=1
        
        entity_embeddings = torch.tensor(entity_embeddings)
        entity_positions = torch.tensor(entity_positions).float()
        entity_shapes = torch.tensor(entity_shapes).float()
        #entity_positions = ( (entity_positions.float() - entity_positions.float().mean(0))/ entity_positions.float().std(0))
        #normalize positions with respect to page
        entity_positions.float()
        entity_positions[:,1]=entity_positions[:,1]/float(image_h)
        entity_positions[:,0]=entity_positions[:,0]/float(image_w)
        entity_positions-=0.5

        entity_shapes[:,1]=entity_shapes[:,1]/float(image_h)
        entity_shapes[:,0]=entity_shapes[:,0]/float(image_w)
        
        entity_labels = torch.tensor(entity_labels)
        entity_labels=torch.cat([entity_labels.view([-1,1]).float(),entity_positions],dim=1)
        
        k = min(100,int(entity_positions.shape[0]))
        #entity_graph = dgl.transform.knn_graph(entity_positions,k)
        entity_graph_nx = nx.complete_graph(len(form_data))
        entity_graph = dgl.DGLGraph()
        entity_graph.from_networkx(entity_graph_nx)
        entity_graph = dgl.to_bidirected(entity_graph)
        entity_graph_edges = torch.t(torch.stack([entity_graph.edges()[0],entity_graph.edges()[1]]))
        
        entity_graph.ndata['position']=entity_positions
        entity_graph.ndata['w_embed']=entity_embeddings
        entity_graph.ndata['shape']=entity_shapes

        entity_link_labels = []
        for edge in entity_graph_edges.tolist():
            if edge in entity_links or [edge[1],edge[0]] in entity_links:
                entity_link_labels.append(1)
            else:
                entity_link_labels.append(0)
        entity_link_labels=torch.tensor(entity_link_labels)
        
        target_am = np.zeros((len(node_position),len(node_position)))
        am=np.ones((len(node_position),len(node_position)))
        word_idx=0
        # Fill target am for word grouping
        for entity in form_data:
            first_region_word=word_idx
            for word in entity['words']:
                word_idx+=1
            last_region_word = word_idx

            target_am[first_region_word:last_region_word,first_region_word:last_region_word]=1
        
        
        G = nx.from_numpy_matrix(am)
        nx.set_node_attributes(G, node_position, 'position')
        nx.set_node_attributes(G, node_text, 'w_embed')
        nx.set_node_attributes(G, node_entity, 'entity')
        #nx.set_node_attributes(G, node_shape, 'shape')
        
        pairs,labels= adjacency_to_pairs_and_labels(target_am)
        label_dict={}
        for k in range(len(pairs)):
            label_dict[pairs[k]]=labels[k]
        
        return entity_graph,entity_link_labels
        #return G,label_dict,entity_labels,entity_link_labels

