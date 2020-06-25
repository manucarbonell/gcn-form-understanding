import torch
import pdb
import dgl
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import KMeans

def adjacency_to_pairs_and_labels(am):
    pairs=[]
    labels=[]
    for i in range(am.shape[0]):
        for j in range(am.shape[1]):
            pairs.append((i,j))
            labels.append(am[i,j])

    return pairs,labels
    
def visualize_graph(g,scale_x = 1.,scale_y=1. ,im_out_path='',bkg_im_path = '',edge_color='g'):
    position = np.array(g.ndata['position'])
    
    position+=0.5

    position[:,1] *= scale_y  
    position[:,0] *= scale_x 
    g = g.to_networkx()
    nx.draw_networkx(g, pos=position, arrows=False,node_size=10,with_labels=True,edge_color = edge_color,font_size=5)
    
    if len(im_out_path)>0:
        plt.savefig(im_out_path,dpi=300)
    #plt.show()

def edges_list_to_dgl_graph(edges,num_nodes=0):
    if num_nodes>0:
        nodes = num_nodes
    else:
        nodes = int(torch.max(edges)+1)
    g = dgl.DGLGraph()
    g.add_nodes(nodes)
    g.add_edges(edges[:,0],edges[:,1])
    return g

def spectral_clustering(A):
    # diagonal matrix
    D = np.diag(A.sum(axis=1))

    # graph laplacian
    L = D-A

    # eigenvalues and eigenvectors
    vals, vecs = np.linalg.eig(L)

    # sort these based on the eigenvalues
    vecs = vecs[:,np.argsort(vals)]
    vals = vals[np.argsort(vals)]

    # kmeans on first three vectors with nonzero eigenvalues
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(vecs[:,1:4])
    colors = kmeans.labels_
    return colors
