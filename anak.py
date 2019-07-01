# use node2vec with cora datset
import numpy as np
import scipy.sparse as sp
import torch
import networkx as nx
import matplotlib.pyplot as plt
import parameters as param
import time

from node2vec import Node2Vec
from networkx.algorithms import bipartite
from utils import Cora, Copd, get_subgraph_disconnected
from sys import path

# import sys
# sys.path.insert(0, r'C:\Users\Anak\PycharmProjects\AttentionWalk')
# from src.attentionwalk import AttentionWalkLayer

def pause():
    print("done")
    exit()

# here>> do node2vec emb
def node2vec_emb(G, save_path = './output/node2vec/gene_disease', EMBEDDING_FILENAME = 'nod2vec_emd_gene_disease.txt', log=True):
    # Precompute probabilities and generate walks - **ON WINDOWS ONLY WORKS WITH workers=1**
    node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200,
                        workers=4)  # Use temp_folder for big graphs
    s = time.time()
    # Embed nodes
    model = node2vec.fit(window=10, min_count=1,
                         batch_words=4)  # Any keywords acceptable by gensim.Word2Vec can be passed, `diemnsions` and `workers` are automatically passed (from the Node2Vec constructor)
    f = time.time()
    total = f-s
    print(f'total = {total}')

    # Look for most similar nodes
    print("--Look for most similar nodes")
    model.wv.most_similar('2')  # Output node names are always strings

    output_path = save_path + EMBEDDING_FILENAME

    # Save embeddings for later use
    model.wv.save_word2vec_format(output_path)

    if log:
        with open(f'./log/{EMBEDDING_FILENAME}', 'w') as f:
            f.write(f' --{save_path}{EMBEDDING_FILENAME}\n')
            f.write(f'total running time {total}')



def nx_plot(G, pos=None, node_color=None ):

    if pos is not None and node_color is not None:
        nx.draw(G, pos=pos, node_color=node_color)
        plt.show()
    else:
        nx.draw(G)
        plt.show()

def main():
    cora = Cora()
    copd = Copd()
    # adj, features, labels, idx_train, idx_val, idx_test = cora.load_data()
    adj, labels, idx_train, idx_val, idx_test = copd.load_data()
    G = nx.from_numpy_matrix(np.asmatrix(adj))
    subgraph = get_subgraph_disconnected(G)[0]
    # -- grpah G
    # print(len(G.nodes)) # 2490
    # print(len(G.edges)) # 3687

    # --subgraph
    # print(len(subgraph.nodes)) # 2475
    # print(len(subgraph.edges)) # 3678
    # exit()

    # --color bipartite graph
    if param.plot:
        left, right = bipartite.sets(subgraph)
        bipartite_color = [0 if i < len(left) else 1 for i, _ in enumerate(left.union(right))]
        pos = nx.circular_layout(G)
        nx_plot(subgraph, pos=pos, node_color=bipartite_color)

    # # -- embbedding
    # node2vec_emb(subgraph)

def bine_copd_label():
    # load data in to dataframe
    # add u to gene and add i to item
    # create new columns of weight = 1 ( unweightd)
    pass

if __name__ == "__main__":
    # main()
    bine_copd_label()