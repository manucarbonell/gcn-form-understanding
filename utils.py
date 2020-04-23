import torch
import pdb
import dgl
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def adjacency_to_pairs_and_labels(am):
    pairs=[]
    labels=[]
    for i in range(am.shape[0]):
        for j in range(am.shape[1]):
            pairs.append((i,j))
            labels.append(am[i,j])

    return pairs,labels
    
def visualize_graph(g):
    position = np.array(g.ndata['position'])
    g = g.to_networkx()
    nx.draw(g, pos=position, arrows=False)
    plt.show()

def edges_list_to_dgl_graph(edges):
    g = dgl.DGLGraph()
    g.add_nodes(int(torch.max(edges))+1)
    g.add_edges(edges[:,0],edges[:,1])
    return g

