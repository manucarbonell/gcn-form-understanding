# -*- coding: utf-8 -*-
"""Copy of gcn-form-understanding.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1rBtuZ32pWcS9-U6QcLTgPzjj8yW6eu72

# Form Understanding

## Dependencies
"""

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
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from model import Net
from datasets import FUNSD,collate,pxml2graph



"""## Get ground truth"""
if not os.path.exists('funsd.tar.gz'):
    os.system('wget --no-check-certificate '+ "'https://docs.google.com/uc?export=download&id=1Y8UQe4-2uLti_YbnyZbwSWpUnC6kUZVj'"+' -O funsd.tar.gz')
    os.system('tar -zxvf funsd.tar.gz')

"""## Auxiliary functions"""




def adjacency_to_pairs_and_labels(am):
    pairs=[]
    labels=[]
    for i in range(am.shape[0]):
        for j in range(am.shape[1]):
            pairs.append((i,j))
            labels.append(am[i,j])

    return pairs,labels



"""## Graph dataloader"""

"""#### Define Dataset Class
Pytorch provides an abstract class representig a dataset, ```torch.utils.data.Dataset```. We need to override two methods:

*   ```__len__``` so that ```len(dataset)``` returns the size of the dataset.
*   ```__getitem__``` to support the indexing such that ```dataset[i]``` can be used to get i-th sample
"""


# Define the corresponding subsets for train, validation and test.
#trainset = Pages(os.path.join(dataset_path, distortion), 'train.cxl')
"""### Prepare DataLoader

```torch.utils.data.DataLoader``` is an iterator which provides:


*   Data batching
*   Shuffling the data
*   Parallel data loading

In our specific case, we need to deal with graphs of many sizes. Hence, we define a new collate function makin guse of the method ```dgl.batch```.
"""



"""## Model"""


"""## Define data loaders"""

trainset = FUNSD('funsd_train','')
validset = FUNSD('funsd_valid','')
testset = FUNSD('funsd_test','')

train_loader = DataLoader(trainset, batch_size=1, shuffle=True,collate_fn=collate)
valid_loader = DataLoader(validset, batch_size=1, collate_fn=collate)
test_loader = DataLoader(testset, batch_size=1, collate_fn=collate)

"""# Train step"""

def accuracy(output, target):
  """Accuacy given a logit vector output and a target class
  """
  _, pred = output.topk(1)
  pred = pred.squeeze()
  correct = pred == target
  correct = correct.float()
  return correct.sum() * 100.0 / correct.shape[0]

"""## Training setup"""

def train(model):
    if torch.cuda.is_available():
        model = model.cuda()
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, 20, gamma = 0.9)
    model.train()
    
    def random_choice(tensor,k=100):
        perm = torch.randperm(tensor.size(0))
        idx = perm[:k]
        samples = tensor[idx]
        return samples

    epoch_losses = []
    train_log = open('train_log.txt','w')
    epoch_loss = 0
    best_acc =0
    patience = 10
    epochs_no_improvement=0
    for epoch in range(100):
        epoch_loss = 0
        for iter, (bg, blab) in enumerate(train_loader):
          
            optimizer.zero_grad()
            prediction = model(bg)
            target_edges = blab[0]

            # convert target edges dict from complete graph to input graph edges 0s and 1s
            input_edges = torch.stack(bg.edges()).t().tolist()            
            edges = list(map(tuple, input_edges))
            target = torch.tensor([target_edges[e] for e in edges])
    
            # class weights
            class_weights = target.shape[0]*torch.ones(target.shape)
            class_weights[target.bool()] /= 2*target.sum()
            class_weights[(1-target).bool()] /= 2*(1-target).sum()
            
            loss = F.binary_cross_entropy(prediction,target,weight=class_weights)

            loss.backward()
            
            epoch_loss+=float(loss)
            
            optimizer.step()
            scheduler.step()

            prediction[prediction>0.5] = 1
            prediction[prediction<=0.5] = 0

            if iter % 20 == 0:
                print('\t* Epoch '+str(epoch)+' '+ str(float(iter/len(train_loader))) +' loss '+str(float(loss)) + ' lr' + str(scheduler.get_lr()[0]))
                print("ACCURACY:",float((prediction == target).sum())/prediction.shape[0])
                print("PRECISION:", ((prediction == target)[target.bool()].float().sum()/prediction.sum()).item())
                print("RECALL:", ((prediction == target)[target.bool()].float().sum()/target.sum()).item())

        print(" Validation \n")
        accuracies = []
        for iter,(bg,blab) in enumerate(valid_loader):
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
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            print('new best acc',epoch_acc)
            torch.save(model,'model.pt')
            epochs_no_improvement=0
        else:
            epochs_no_improvement+=1

        if epochs_no_improvement>patience:
            break
    return model

"""# Main"""

#def main():
model_path = 'model.pt'
if os.path.exists(model_path):
    model = torch.load(model_path)
else:
    
    model = Net(2, 128)

    model = train(model)
    torch.save(model,'model.pt')


#if __name__ == "__main__":
#    main()

def visualize_graph(g):
    position = np.array(g.ndata['position'])
    g = g.to_networkx()
    nx.draw(g, pos=position, arrows=False)
    plt.show()

def edges_list_to_dgl(edges):
    g = dgl.DGLGraph()
    g.add_edges(edges[:,0],edges[:,1])
    return g

