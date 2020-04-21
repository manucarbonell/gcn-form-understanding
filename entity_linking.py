#-*- coding: utf-8 -*-
""## Prepare data reader

#IAM graphs are provided as a GXL file:


'''
<gxl>
  <graph id="GRAPH_ID" edgeids="false" edgemode="undirected">
    <node id="_0">
      <attr name="x">
        <float>0.812867</float>
      </attr>
      <attr name="y">
        <float>0.630453</float>
      </attr>
    </node>
    ...
    <node id="_N">
      ...
    </node>
    <edge from="_0" to="_1"/>
    ...
    <edge from="_M" to="_N"/>
  </graph>
</gxl>

'''
import torch.utils.data as data
import glob

from random import randrange
from dgl.nn.pytorch import GraphConv
import matplotlib.pyplot as plt
import dgl.function as fn

from torch import nn
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

import numpy as np
import xml.etree.ElementTree as ET
import networkx as nx
import pdb
import pagexml
import re
import torch 
import dgl

def get_coords_and_transcript(pxml,textobject,key):
    regex = re.compile('[^a-zA-Z]')
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
    line_transcript=[]
    w=transcription
    '''for w in transcription.split(" "):    
        if '<' in w:
            w = w.split('>')[1].split('<')[0]
    '''
    w= regex.sub('', w)
           
    line_transcript.append(w)
    line_transcript = " ".join(line_transcript)
    line_transcript = line_transcript.strip()
    return x0,y0,x1,y1,line_transcript,tag

def distance_matrix(vectors):
    d =np.zeros((len(vectors),len(vectors)))
    
    for i in range(len(vectors)):
        for j in range(len(vectors)):
            d[i,j]=1-np.dot(vectors[i],vectors[j])/(np.linalg.norm(vectors[i])*np.linalg.norm(vectors[j]))

    return d


def adjacency_to_pairs_and_labels(am):
    pairs=[]
    labels=[]
    for i in range(am.shape[0]):
        for j in range(am.shape[1]):
            pairs.append((i,j))
            labels.append(am[i,j])

    return pairs,labels


def pxml2graph(file):
    # parse a pagexml file and return a networkx graph
    pagexml.set_omnius_schema()
    pxml = pagexml.PageXML()
    pxml.loadXml(file)
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
                x0,y0,x1,y1,transcription,tag=get_coords_and_transcript(pxml,Word,'label')
                node_label[word_idx] = np.array([x0, y0])
                word_idx+=1

    # node adjacency matrix
                        
    am=np.ones((len(node_label),len(node_label)))

    target_am=np.zeros((len(node_id),len(node_id)))
    
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
                word_idx+=1
            last_region_word = word_idx

            target_am[first_region_word:last_region_word,first_region_word:last_region_word]=1

    G = nx.from_numpy_matrix(am)
    nx.set_node_attributes(G, node_label, 'position')
    target_G = nx.from_numpy_matrix(target_am)
    nx.set_node_attributes(target_G, node_label, 'position')
    node_label = torch.stack([torch.tensor(v['position']) for k,v in G.nodes.items()])
    pairs,labels= adjacency_to_pairs_and_labels(target_am)
    label_dict={}
    for k in range(len(pairs)):
        label_dict[pairs[k]]=labels[k]
    #target_G = dgl.transform.knn_graph(node_label,10)
    #target_G.ndata['position'] = node_label
    
    #target_G.edata['groups'] = edge_attr
    #nx.set_edge_attributes(target_G, edge_attr,'groups')
    return G,label_dict



# Read the graph and draw it using networkx tools


"""## Batch processing

### NetworkX to DGL graph
"""


# Create graph object
# Data structure
#print(g)

"""### Define dataset

#### Dataset Division

The dataset is divided by means of CXL files in *train*, *validation* and *test* with the correspondance filename and class:


```
<GraphCollection>
  <fingerprints base="/scratch/mneuhaus/progs/letter-database/automatic/0.1" classmodel="henry5" count="750">
    <print file="AP1_0100.gxl" class="A"/>
    ...
    <print file="ZP1_0149.gxl" class="Z"/>
  </fingerprints>
</GraphCollection>
```
"""

def getFileList(file_path):
  """Parse CXL file and returns the corresponding file list and class
  """
  
  elements, classes = [], []
  tree = ET.parse(file_path)
  root = tree.getroot()
  
  for child in root:
    for sec_child in child:
      if sec_child.tag == 'print':
        elements += [sec_child.attrib['file']]
        classes += sec_child.attrib['class']
        
  return elements, classes

"""#### Define Dataset Class
Pytorch provides an abstract class representig a dataset, ```torch.utils.data.Dataset```. We need to override two methods:

*   ```__len__``` so that ```len(dataset)``` returns the size of the dataset.
*   ```__getitem__``` to support the indexing such that ```dataset[i]``` can be used to get i-th sample
"""


class Pages(data.Dataset):
  """PagesDataset
  """
  
  def __init__(self, root_path, file_list):
    self.root = root_path
    self.file_list = file_list
    
    # List of files and corresponding labels
    self.files =glob.glob(root_path+'/*.xml')
                #os.listdir(root_path)
                
    #self.graphs, self.labels = getFileList(os.path.join(self.root, self.file_list))
    '''
    # Labels to numeric value
    self.unique_labels = np.unique(self.labels)
    self.num_classes = len(self.unique_labels)
    
    self.labels = [np.where(target == self.unique_labels)[0][0] 
                   for target in self.labels]
    '''
    
  def __getitem__(self, index):
    # Read the graph and label
    #G = read_letters(os.path.join(self.root, self.graphs[index]))
    #target = self.labels[index]
    G,target =pxml2graph(os.path.join(self.root,self.files[index]))
    # Convert to DGL format
    '''g = dgl.DGLGraph()
    g.set_n_initializer(dgl.init.zero_initializer)
    g.set_e_initializer(dgl.init.zero_initializer)
    g.from_networkx(G, node_attrs=['position'])'''
    node_label = torch.stack([torch.tensor(v['position']) for k,v in G.nodes.items()])

    g_in = dgl.transform.knn_graph(node_label,20)
    #g_in = dgl.transform.knn_graph(G.ndata['position'], 40)
    
    g_in.ndata['position'] =node_label.float()

    #target_g = dgl.transform.knn_graph(node_label,10)
    return g_in, target
  
  def label2class(self, label):
    # Converts the numeric label to the corresponding string
    return self.unique_labels[label]
  
  def __len__(self):
    # Subset length
    return len(self.files)

# Define the corresponding subsets for train, validation and test.
#trainset = Pages(os.path.join(dataset_path, distortion), 'train.cxl')
"""### Prepare DataLoader

```torch.utils.data.DataLoader``` is an iterator which provides:


*   Data batching
*   Shuffling the data
*   Parallel data loading

In our specific case, we need to deal with graphs of many sizes. Hence, we define a new collate function makin guse of the method ```dgl.batch```.
"""

from torch.utils.data import DataLoader

def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    #labels = dgl.batch(labels)
    return batched_graph, labels#torch.tensor(labels)
  
# Define the three dataloaders. Train data will be shuffled at each epoch

def accuracy(output, target):
  """Accuacy given a logit vector output and a target class
  """
  _, pred = output.topk(1)
  pred = pred.squeeze()
  correct = pred == target
  correct = correct.float()
  return correct.sum() * 100.0 / correct.shape[0]

"""## Define Model

To define a Graph Convolution, three functions have to be defined:




*   Message: Decide which information is sent by a node
*   Reduce: Combine the messages and the current data
*   NodeApply: Update the node features that are recieved from the reduce function
"""

  

class Net(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(Net, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.score = nn.CosineSimilarity()
    def get_complete_graph_and_pairs(self,num_nodes):
        G = nx.complete_graph(num_nodes)
        
        g = dgl.DGLGraph()
        g.from_networkx(G)
        pairs = torch.t(torch.stack([g.edges()[0],g.edges()[1]]))
        return g,pairs

    def calc_score(self,g):
        pairs = torch.t(torch.stack([g.edges()[0],g.edges()[1]]))
        # get hidden state of all source destination pairs and calculate their edge score
        s = g.ndata['h'][pairs[:,0]]
        d = g.ndata['h'][pairs[:,1]]
        
        score = self.score(s,d)
        return score


    def forward(self, g):
        # For undirected graphs, in_degree is the same as
        # out_degree.
        h = g.ndata['position']
        #h = torch.cat((g.ndata['position'], g.in_degrees().view(-1, 1).float()), dim=1)
        
        if torch.cuda.is_available():
          h = h.cuda() 
        h = self.conv1(g, h)
        h = self.conv2(g, h)
        
        g.ndata['h'] = h
        score = self.calc_score(g)
        #hg = dgl.mean_nodes(g, 'h')
        #return self.classify(hg)
        return score


"""## Training setup"""

def train(model):
    trainset = Pages('/home/mcarbonell/DATASETS/FUNSD/pagexml/dataset-funsd-master@9496c3db5f7/data/training','')
    validset = Pages('/home/mcarbonell/DATASETS/FUNSD/pagexml/dataset-funsd-master@9496c3db5f7/data/valid','')
    testset = Pages('/home/mcarbonell/DATASETS/FUNSD/pagexml/dataset-funsd-master@9496c3db5f7/data/testing','')
    
    train_loader = DataLoader(trainset, batch_size=1, shuffle=True,collate_fn=collate)
    valid_loader = DataLoader(validset, batch_size=1, collate_fn=collate)
    test_loader = DataLoader(testset, batch_size=1, collate_fn=collate)
    if torch.cuda.is_available():
        model = model.cuda()
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    
    def random_choice(tensor,k=100):
        perm = torch.randperm(tensor.size(0))
        idx = perm[:k]
        samples = tensor[idx]
        return samples

    epoch_losses = []
    train_log = open('train_log.txt','w')
    epoch_loss = 0

    for iter, (bg, blab) in enumerate(train_loader):
        for epoch in range(500):
            #print(float(iter/len(train_loader)))
            ''' 
            ##### VISUALIZE X Y ###########################3
            # X
            G = bg.to_networkx(node_attrs=['position'])
            position = {k: v.numpy() for k, v in dict(G.nodes(data='position')).items()}
            nx.draw(G, pos=position, arrows=False)
            plt.show()
            ###################### 
            # Y            
            G = label.to_networkx(node_attrs=['position'])
            position = {k: v.numpy() for k, v in dict(G.nodes(data='position')).items()}
            nx.draw(G, pos=position, arrows=False)
            plt.show()
            
            ############################################################## 
            '''
            optimizer.zero_grad()
            prediction = model(bg)
            target_edges = blab[0]
            input_edges = torch.t(torch.stack([bg.edges()[0],bg.edges()[1]])).tolist()
            
            edges = [tuple(input_edges[k]) for k in range(len(input_edges))]
            target = torch.tensor([target_edges[edges[k]] for k in range(len(edges))])
            
            
            loss =  F.binary_cross_entropy_with_logits(prediction,target)
            #loss = F.binary_cross_entropy_with_logits(prediction,target)#,weight=class_weights)

            print('epoch '+str(epoch)+' '+ str(float(iter/len(train_loader))) +' loss '+str(float(loss)))
        
            loss.backward()
            optimizer.step()
            prediction[prediction<0.8] = 0
            prediction[prediction>0.8] = 1
            if epoch % 30 == 0: pdb.set_trace()
            print("ACC:",float((prediction == target).sum())/prediction.shape[0])
            print( model.conv2.weight.grad.norm())
            #print("POSITIVES ACC:",float((prediction[positives_idx] == target[positives_idx]).sum())/prediction.shape[0])
        pdb.set_trace()
def main():
    print('READ GRAPH DATA')

    # DATA SAMPLE 
    G,labels = pxml2graph('/home/mcarbonell/DATASETS/FUNSD/pagexml/dataset-funsd-master@9496c3db5f7/data/training/93455715.xml')

    print('BUILD GCN MODEL')
    model = Net(2, 512)

    print('TRAIN MODEL')
    train(model)
    print('EVALUATE MODEL')

if __name__ == "__main__":
    main()
