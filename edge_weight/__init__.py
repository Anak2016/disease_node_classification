import networkx as nx
import numpy as np
from arg_parser import *

def jaccard_coeff(dataset,geometric_dataset, original_edges, added_edges, edges, mask_edges=False, weight_limit=1, self_loop=False, edges_percent=1):
    '''
     import networkx as nx
     G = nx.complete_graph(5)
     preds = nx.jaccard_coefficient(G, [(0, 1), (2, 3)])
     for u, v, p in preds:
         '(%d, %d) -> %.8f' % (u, v, p)

    '(0, 1) -> 0.60000000'
    '(2, 3) -> 0.60000000'
    param: dataset = Copd
    param: edges numpy array dim = 2 * number of edges
    param: percentage of edges to be added:
    :return:
        weighted_adj_matrix
            type = numpy matrix

        edges_weight of the same edges sequence
                type = numpy arry dim = 2 * num_edges

        ordered_edges = edges pair of the order of edges_weight (rearrange during the computing process)
    '''
    G = nx.Graph()
    G.add_edges_from(zip(edges[0], edges[1])) # 5254
    # original_adj_matrix = nx.to_numpy_matrix(G)

    edges_weight = []
    max_weight = 1 # weight of orignal edges
    num_disease = dataset.num_diseases
    count_original=0
    count_all_new_edges=0 # all of added edges because select with weight_limit
    count_added_edges = 0
    x = []
    count = 0
    preds = nx.jaccard_coefficient(G, [tuple(i.tolist()) for i in edges.T]) # because added_edges, jaccard between disease and gene are no longer 0
    for u, v, p in preds:
        weight = p
        count+=1
        x.append(p)
        if weight > max_weight:
            max_weight = weight
        #TODO here>> below code needs to be fix ONLY IF I NEED TO.
        #  > as of now, only 'weight' attribute name can be created (this needs to change when there are more than 1 weight attribute)
        if weight != 0:
            # G.add_weighted_edges_from([(u, v, weight)], weight='jaccard_coeff')  # 3334
            # G.add_weighted_edges_from([(u, v, weight)])  # 3334
            '''
            note:
            3334 edges are less than 1 and more than 0
            '''
            #TODO here>> instead of doing this
            # > mask original edges with 1
            if weight> weight_limit:
                G.add_edge(u,v,weight=weight)

            # edges_weight.append(weight)

        # if weight == 0:
        #     # why some weight is not zero
        #     G.add_edge(u, v, weight=1)
        #     edges_weight.append(1)

        # if mask:  # mask original edges with weight = 1
        #     if (u >= num_disease and v < num_disease) or (u < num_disease and v >= num_disease):  # genes and edgse
        #         G.add_weighted_edges_from([(u, v, 1)], weight='jaccard_coeff')  # 3334
        #         edges_weight.append(1)
        #         count_original += 1
        #     else:
        #         # TODO here>> add edges above weight_limit
        #         if p >= weight_limit:
        #             G.add_weighted_edges_from([(u, v, weight)], weight='jaccard_coeff')  # 3334
        #             if weight < 1:
        #                 tmp = u
        #                 tmp1 = v
        #             edges_weight.append(weight)
        #             count_added_edges += 1
        #         count_all_new_edges += 1

    #TODO here>> make it run faster by saving it to numpy folder to be loaded back
    # if mask:
    #     assert count_original == geometric_dataset.edges_index.shape[1], "not all original are mask with weight = 1"
    #
    # assert count_original+count_all_new_edges == edges.shape[1], 'all edges must have weight '

    # weighted_adj_matrix = nx.to_numpy_matrix(G, nodelist=list(range(dataset.num_nodes)), weight='jaccard_coeff')
    weighted_adj_matrix = nx.to_numpy_matrix(G, nodelist=list(range(dataset.num_nodes)), weight='weight')

    # TODO here>> show histogram of edges_weight
    # > select from weight value
    # > select from percentage
    # freq = np.histogram(weighted_adj_matrix, bins=np.linspace(0,1,num=1000))[0]/5254 # get percentage
    import matplotlib.pyplot as plt
    freq = np.histogram(weighted_adj_matrix, bins=np.linspace(0, 1, num=100, endpoint=False))[0]# get freq
    max_freq = np.amax(freq)
    print(f'freq = {freq}')
    plt.bar(np.arange(len(freq)), freq, align='center', alpha=0.5)
    np.set_printoptions(precision=3)
    plt.xticks(np.arange(len(freq)), list(map(str, np.around(np.linspace(0, 1, num=100), 3))), rotation=90)
    plt.ylim(0, 500)
    plt.show()

    #--------select top n percent of the highest value
    print(f"select top {edges_percent} percent  highest val")
    weighted_adj_matrix[weighted_adj_matrix > 0].flatten()[::-1].sort()
    tmp = weighted_adj_matrix
    tmp = tmp[:round(tmp.shape[1] * 0.4)]
    max_val = np.amax(tmp.flatten())
    weighted_adj_matrix =  np.where(weighted_adj_matrix < max_val, 0, weighted_adj_matrix)
    print(f'number of nonzero = {weighted_adj_matrix.nonzero()[0].shape}')


    # weighted_adj_matrix = nx.to_numpy_matrix(G, nodelist=list(range(dataset.num_nodes)), weight='jaccard_coeff')/1  # normalized edges by max_weight
    # weighted_adj_matrix = nx.to_numpy_matrix(G, weight='jaccard_coeff')/1  # normalized edges by max_weight
    if mask_edges:
        #--------mask original edges with weight = 1
        from scipy.sparse import csr_matrix
        # original_adj = csr_matrix((np.arange(original_edges.shape[1]),original_edges[0], original_edges[1]), shape=(dataset.num_nodes, dataset.num_nodes)).toarray()
        weighted_adj_matrix[original_edges[0],original_edges[1]] = 1

        tmp = csr_matrix(weighted_adj_matrix)
        ordered_edges = []
        for i,j in zip(*tmp.nonzero()):
            ordered_edges.append([i,j])
            edges_weight.append(tmp[i,j])

    if self_loop:
        diag = np.arange(weighted_adj_matrix.shape[0])
        weighted_adj_matrix[diag, diag] = 1

    return weighted_adj_matrix, np.array(edges_weight), np.array(original_edges)

    # return np.array(edges_weight)
    # return np.array([i[2] for i in preds])
