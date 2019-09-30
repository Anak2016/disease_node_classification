import networkx as nx
import numpy as np
from arg_parser import *
def jaccard_coeff( edges):
    '''
     import networkx as nx
     G = nx.complete_graph(5)
     preds = nx.jaccard_coefficient(G, [(0, 1), (2, 3)])
     for u, v, p in preds:
         '(%d, %d) -> %.8f' % (u, v, p)

    '(0, 1) -> 0.60000000'
    '(2, 3) -> 0.60000000'

    param: edges numpy array dim = 2 * number of edges
    :return:
        weighted_adj_matrix
            type = numpy matrix

        edges_weight of the same edges sequence
                type = numpy arry dim = 2 * num_edges
    '''
    G = nx.Graph()
    G.add_edges_from(zip(edges[0], edges[1]))
    # original_adj_matrix = nx.to_numpy_matrix(G)

    preds = nx.jaccard_coefficient(G, [tuple(i.tolist()) for i in edges.T])
    edges_weight = []
    max_weight = 0
    for u,v,p in preds:
        if args.verbose:
            print('(%d, %d) -> %.8f' % (u, v, p))
        weight = 1/p if p != 0 else 0
        if weight > max_weight:
            max_weight = weight
        G.add_weighted_edges_from([(u,v,weight)], weight='jaccard_coeff')
        edges_weight.append(weight)


    weighted_adj_matrix = nx.to_numpy_matrix(G, weight='jaccard_coeff')/max_weight  # normalized edges by max_weight

    return weighted_adj_matrix, np.array(edges_weight)

    # return np.array(edges_weight)
    # return np.array([i[2] for i in preds])
