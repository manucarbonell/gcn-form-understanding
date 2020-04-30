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
from utils import adjacency_to_pairs_and_labels
def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    #labels = dgl.batch(labels)
    return batched_graph, labels#torch.tensor(labels)


class FUNSD(data.Dataset):
    """PagesDataset
    """
  
    def __init__(self, root_path, file_list):
        self.root = root_path
        self.file_list = file_list

        # List of files and corresponding labels
        self.files =glob.glob(root_path+'/*.xml')
                    #os.listdir(root_path)
        
        if not os.path.exists('embeddings.bin'):
            self.embeddings = fasttext.train_unsupervised('text_data.txt', model='skipgram')
            self.embeddings.save_model("embeddings.bin")
        else:
            self.embeddings =fasttext.load_model("embeddings.bin")

    def __getitem__(self, index):
        # Read the graph and label
        G,target =self.pxml2graph(self.files[index])

        # Convert to DGL format
        node_label = torch.stack([torch.tensor(v['position']) for k,v in G.nodes.items()]).float()
        node_label = (node_label - node_label.mean(0))/ node_label.std(0)

        node_word = torch.stack([torch.tensor(v['w_embed']) for k,v in G.nodes.items()]).float()
        
        g_in = dgl.transform.knn_graph(node_label,10)
        g_in = dgl.to_bidirected(g_in)

        g_in.ndata['position'] =node_label.float()
        g_in.ndata['w_embed'] =node_word.float()

        return g_in, target

    def label2class(self, label):
        # Converts the numeric label to the corresponding string
        return self.unique_labels[label]

    def __len__(self):
        # Subset length
        return len(self.files)
    
    def pxml2graph(self,xml_file):
        
        def get_coords_and_transcript(pxml,textobject,key):
            coords = pxml.getPoints(textobject)
            if len(coords)==4: 
                arg_max_coord=2
            else:
                arg_max_coord=1
            x0=int(coords[0].x)
            y0=int(coords[0].y)
            
            x1=int(coords[arg_max_coord].x)
            y1=int(coords[arg_max_coord].y)

            transcription = pxml.getTextEquiv(textobject)
            tag = pxml.getPropertyValue(textobject,key=key)
            transcription = transcription.lower()
            transcription=re.sub('[0-9]','N',transcription)
                    
            return x0,y0,x1,y1,transcription,tag
        # parse a pagexml file and return a dgl input graph and an edge target dictionary
        pagexml.set_omnius_schema()
        pxml = pagexml.PageXML()
        pxml.loadXml(xml_file)
        pages = pxml.select('_:Page')
        node_id=[]
        node_label={}
        # Parse nodes (text boxes)
        for page in pages:
            pagenum = pxml.getPageNumber(page)
            regions = pxml.select('_:TextRegion',page)
            word_idx=0
            for region in regions:
                words=pxml.select('_:Word',region)
                for Word in words :
                    node_id.append(word_idx)
                    data_f = open('text_data.txt','a')
                    x0,y0,x1,y1,transcription,tag=get_coords_and_transcript(pxml,Word,'label')
                    data_f.write(transcription+' ')
                    data_f.close()
                    node_label[word_idx] = np.array([x0, y0])
                    word_idx+=1

        # node adjacency matrix
                            
        am=np.ones((len(node_label),len(node_label)))

        target_am=np.zeros((len(node_id),len(node_id)))
        node_words = {}
        node_shape = {}
        # nodes corresponding words in same groups are connected:
        for page in pages:
            pagenum = pxml.getPageNumber(page)
            regions = pxml.select('_:TextRegion',page)
            word_idx=0
            for region in regions:
                words=pxml.select('_:Word',region)
                first_region_word = word_idx
                for Word in words :
                    node_id.append(word_idx)
                    x0,y0,x1,y1,transcription,tag=get_coords_and_transcript(pxml,Word,'label')
                    node_label[word_idx] = np.array([x0, y0])
                    node_words[word_idx] = self.embeddings[transcription] 
                    node_shape[word_idx] = np.array([x1-x0,y1-y0])
                    word_idx+=1
                last_region_word = word_idx

                target_am[first_region_word:last_region_word,first_region_word:last_region_word]=1

        G = nx.from_numpy_matrix(am)
        nx.set_node_attributes(G, node_label, 'position')
        nx.set_node_attributes(G, node_words, 'w_embed')
        nx.set_node_attributes(G, node_shape, 'shape')
        
        pairs,labels= adjacency_to_pairs_and_labels(target_am)
        label_dict={}
        for k in range(len(pairs)):
            label_dict[pairs[k]]=labels[k]
                
        return G,label_dict

