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
    graphs, group_labels,entity_labels,entity_links = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    #labels = dgl.batch(labels)
    return batched_graph, group_labels,entity_labels,entity_links#torch.tensor(labels)


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
        G,group_labels,entity_labels,entity_links =self.read_annotations(self.files[index])

        # Convert to DGL format
        node_label = torch.stack([torch.tensor(v['position']) for k,v in G.nodes.items()]).float()
        node_label = (node_label - node_label.mean(0))/ node_label.std(0)

        node_word = torch.stack([torch.tensor(v['w_embed']) for k,v in G.nodes.items()]).float()
        node_entity = torch.stack([torch.tensor(v['entity']) for k,v in G.nodes.items()]).float()
        # ENSURE BIDIRECTED
        g_in = dgl.transform.knn_graph(node_label,10)
        g_in = dgl.to_bidirected(g_in)

        g_in.ndata['position'] =node_label.float()
        g_in.ndata['w_embed'] =node_word.float()
        g_in.ndata['entity'] = node_entity
   


        input_edges = torch.stack(g_in.edges()).t().tolist()            
        edges = list(map(tuple, input_edges))
        target_edges = group_labels
        group_labels = torch.tensor([target_edges[e] for e in edges])


        return g_in,group_labels,entity_labels,entity_links

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
       
        with open(json_file) as f:
            data = json.load(f)
        form_data = data['form']
        
        node_position={}
        node_text ={}
        node_shape = {}
        node_entity = {}
        word_idx=0
        entity_idx = 0
        entity_links=[]
        entity_positions=[]
        entity_labels = []
        # Get total amount of words in the form and their attr to create am.
        for entity in form_data:
            for link in entity['linking']:
                entity_links.append(link) 
            for word in entity['words']:
                node_position[word_idx]=word['box'][:2]
                node_text[word_idx]=self.embeddings[word['text']]
                node_entity[word_idx]=entity_idx
                word_idx+=1
            entity_position = np.array([word['box'][:2] for word in entity['words']]).mean(axis=0)
            entity_positions.append(entity_position)
            entity_labels.append((self.class2label(entity['label'])))
            entity_idx+=1
        entity_positions = torch.tensor(entity_positions)
        entity_positions = ( (entity_positions.float() - entity_positions.float().mean(0))/ entity_positions.float().std(0))
        entity_labels = torch.tensor(entity_labels)
        entity_labels=torch.cat([entity_labels.view([-1,1]).float(),entity_positions],dim=1)
        entity_graph = dgl.transform.knn_graph(entity_positions,10)
        entity_graph_edges = torch.t(torch.stack([entity_graph.edges()[0],entity_graph.edges()[1]]))
        entity_link_labels = []
        for edge in entity_graph_edges.tolist():
            if edge in entity_links:
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
        
        return G,label_dict,entity_labels,entity_link_labels

