import networkx as nx
import numpy as np
from arg_parser import *
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import time
import networkx as nx
import collections
from collections import OrderedDict
from operator import itemgetter



def jaccard_coeff(dataset,geometric_dataset, original_edges, added_edges, edges, mask_edges=False, weight_limit=None, self_loop=False,
                  weight_limit_percent=None,  top_edges_percent=None, bottom_edges_percent=None, all_nodes_random_edges_percent=None, shared_nodes_random_edges_percent=None,top_bottom_percent=None):
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
    param: weight_limit_percent: select edges with weight <= max(20% from lowest value) note: after 50 percent, max value = 1
    param: weight_limit: select edges by weight limit
    param: edges_percent: select edges with highest value n percent of all edges.
    :return:
        weighted_adj_matrix
            type = numpy matrix

        edges_weight of the same edges sequence
                type = numpy arry dim = 2 * num_edges

        ordered_edges = edges pair of the order of edges_weight (rearrange during the computing process)
    '''
    try:
        added_edges = np.array(added_edges)
        if added_edges.shape[0] != 2:
            added_edges = added_edges.T

    except:
        pass
    try:
        original_edges = np.array(original_edges)
        if original_edges.shape[0] != 2:
            original_edges = original_edges.T
    except:
        pass
    assert isinstance(added_edges, np.ndarray) and isinstance(original_edges, np.ndarray), "added_edges and original edges must be conected to 2*n shape numpy array"
    assert original_edges.shape[0] ==2, "shape of added_edges must be 2*n"
    assert added_edges.shape[0] == 2, "shape of original_edges must be 2*n"

    # # make added_edges bidirectional
    # inv_edges = added_edges[[1,0]]
    # added_edges = np.concatenate((added_edges, inv_edges),axis=1)

    #=====================
    #==remove self loop if exists
    #=====================
    #TODO here>>
    num_nodes = dataset.num_nodes
    tmp = csr_matrix((np.ones(edges.shape[1]), (edges[0], edges[1])), shape=(num_nodes, num_nodes))
    # print(tmp.nonzero()[0].shape)
    tmp[np.arange(tmp.shape[0]),np.arange(tmp.shape[1])] = 0
    # print(tmp.nonzero()[0].shape)

    #=====================
    #==create graph
    #=====================
    G = nx.Graph()
    # G.add_edges_from(zip(edges[0], edges[1])) # 5254
    G.add_edges_from(zip(edges[0], edges[1])) # does it has self loop

    # original_adj_matrix = nx.to_numpy_matrix(G)

    edges_weight = []
    max_weight = 1 # weight of orignal edges
    num_disease = dataset.num_diseases
    count_original=0
    count_all_new_edges=0 # all of added edges because select with weight_limit
    count_added_edges = 0
    x = []
    count = 0
    tmp = 0

    preds = nx.jaccard_coefficient(G, [tuple(i.tolist()) for i in edges.T]) # because added_edges, jaccard between disease and gene are no longer 0

    # for u, v, p in preds:
    #     weight = p
    #     count+=1
    #     x.append(p)
    #     if weight > max_weight:
    #         max_weight = weight
    #     #TODO here>> below code needs to be fix ONLY IF I NEED TO.
    #     #  > as of now, only 'weight' attribute name can be created (this needs to change when there are more than 1 weight attribute)
    #     if weight != 0:
    #         # G.add_weighted_edges_from([(u, v, weight)], weight='jaccard_coeff')  # 3334
    #         # G.add_weighted_edges_from([(u, v, weight)])  # 3334
    #         '''
    #         note:
    #         3334 edges are less than 1 and more than 0
    #         '''
    #         G.add_edge(u, v, weight=weight)
    #
    #         # if weight_limit is not None and weight>= weight_limit:
    #         #     # from added_edges, only add edges that are above weight_lmit
    #         #     tmp +=1
    #         #     G.add_edge(u,v,weight=weight)
    #         # else:
    #         #     # remove edges that are
    #         #     #TODO here>> if I remove it here then I cannot plot histogram.
    #         #     # > remove_edges after plottin histogram, collect edges to be removed and remove it with remove_edges_from [(u,v),...]
    # #         #     G.remove_edge(u,v)

    for i, (u, v, p) in enumerate(preds):
        weight = p
        # count+=1
        x.append(p)
        if weight > max_weight:
            max_weight = weight
        # if weight != 0:
        # G.add_weighted_edges_from([(u, v, weight)], weight='jaccard_coeff')  # 3334
        # G.add_weighted_edges_from([(u, v, weight)])  # 3334
        '''
        note:
        3334 edges are less than 1 and more than 0
        '''
        G.add_edge(u, v, weight=weight) # add bidirectional edge
        G.add_edge(v, u, weight=weight)
        if weight == 0:
            count +=1 # todo

        # if weight_limit is not None and weight>= weight_limit:
        #     # from added_edges, only add edges that are above weight_lmit
        #     tmp +=1
        #     G.add_edge(u,v,weight=weight)
        # else:
        #     # remove edges that are
        #     #TODO here>> if I remove it here then I cannot plot histogram.
        #     # > remove_edges after plottin histogram, collect edges to be removed and remove it with remove_edges_from [(u,v),...]
        #     G.remove_edge(u,v)

        # edge  s_weight.append(weight)

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

    # weighted_adj_matrix = nx.to_numpy_matrix(G, nodelist=list(range(dataset.num_nodes)), weight='jaccard_coeff')
    # how many edges have weight = 0 of all the edges.
    weighted_adj_matrix = nx.to_numpy_matrix(G, nodelist=list(range(dataset.num_nodes)), weight='weight') #

    #=====================
    #==masking original edges with weight 1 on to the graph
    #=====================
    # weighted_adj_matrix = nx.to_numpy_matrix(G, nodelist=list(range(dataset.num_nodes)), weight='jaccard_coeff')/1  # normalized edges by max_weight
    # weighted_adj_matrix = nx.to_numpy_matrix(G, weight='jaccard_coeff')/1  # normalized edges by max_weight
    ordered_edges = []
    #TODO here>>
    if mask_edges:
        # original_adj = csr_matrix((np.arange(original_edges.shape[1]),original_edges[0], original_edges[1]), shape=(dataset.num_nodes, dataset.num_nodes)).toarray()
        weighted_adj_matrix[original_edges[0],original_edges[1]] = 1
        weighted_adj_matrix[original_edges[1], original_edges[0]] = 1
    # 9430

    #=====================
    #==plot freq histogram ( for non zero)
    #=====================
    freq = np.histogram(weighted_adj_matrix[weighted_adj_matrix > 0], bins=np.linspace(0, 1, num=50, endpoint=False))[0]  # get freq
    max_freq = np.amax(freq)
    print(f'freq = {freq}')
    print(f'cumsum = {freq.cumsum()}')
    # plt.bar(np.arange(len(freq)), freq, align='center', alpha=0.5)
    # np.set_printoptions(precision=3)
    # # plt.xticks(np.arange(len(freq)), list(map(str, np.around(np.linspace(0, 1, num=100), 3))), rotation=90)
    # plt.ylim(0, 400)
    # plt.title(
    #     "histogram plot by weight_limit")  # plot density=true output weird result # just dont use it if I don't need to
    # plt.show()

    #=====================
    #== apply edges selection here
    #=====================

    #--------apply weight_limit here
    if weight_limit is not None:
        # weighted_adj_matrix = np.where(weighted_adj_matrix > weight_limit, weighted_adj_matrix, 0)
        weighted_adj_matrix = np.where(weighted_adj_matrix >= weight_limit, weighted_adj_matrix, 0) # this should be 9430 not 9530
        # weighted_adj_matrix = weighted_adj_matrix[weighted_adj_matrix>weight_limit]

    if weight_limit_percent is not None:
        print(f'selected edges that has value <= than {weight_limit_percent} percentile. '
              f'eg 20 percent => select all edges that are less than max value of top 20 percent of the lowest value')
        # sorted from low to high
        tmp = np.sort(weighted_adj_matrix[weighted_adj_matrix > 0].flatten()[::-1]) # [::-1] does not do anything; test to confirm
        tmp = np.array([i for i in tmp.tolist()[0]])[:int(tmp.size * weight_limit_percent)]
        # print(tmp)
        if tmp is not None:
            weighted_adj_matrix = np.where(weighted_adj_matrix<=np.amax(tmp), weighted_adj_matrix, 0)

    if top_bottom_percent is not None:
        print(f"select top {top_bottom_percent} percent  highest val")

        tmp = weighted_adj_matrix[added_edges[0], added_edges[1]]  # check if both direction have the same value
        tmp = np.array([i for i in tmp.tolist()[0]])
        tmp = np.sort(tmp[tmp > 0].flatten())
        import math
        num_selected = int(tmp.size * top_bottom_percent)
        top_num_selected =  math.ceil(num_selected/2)#selected edges percentage from top
        bottom_num_selected = math.floor(num_selected/2)

        print(f"amount of added edges = {num_selected}")
        if args.stochastic_edges:
            # TODO here>> use new randomize technique and use it with
            print('stochastic_edges activated')
            # tmp = weighted_adj_matrix[added_edges[0], added_edges[1]]
            # list index of all edges
            # ind = np.array(csr_matrix(weighted_adj_matrix).nonzero())
            ind = added_edges

            # --------picked value == threshold randomly

            # ind = np.array(weighted_adj_matrix[weighted_adj_matrix>0].nonzero())
            top_prob = weighted_adj_matrix[ind[0], ind[1]] / weighted_adj_matrix[ind[0], ind[1]].sum()
            top_prob = np.array([i for i in top_prob.tolist()[0]])

            bottom_prob = weighted_adj_matrix[ind[0], ind[1]]
            bottom_prob = 1 / (bottom_prob * 20000)
            bottom_prob = np.where(bottom_prob == float('inf'), 0, bottom_prob)
            bottom_prob = np.array([i for i in bottom_prob.tolist()[0]])

            # TODO here>> does undirect and directed graph have any effects on node2vec?
            tmp = np.copy(weighted_adj_matrix)
            weighted_adj_matrix[ind[0], ind[1]] = 0  # mask added edges with value 0
            weighted_adj_matrix[ind[1], ind[0]] = 0  # mask added edges with value 0

            # --------randomize v1
            # picked_edges_ind = np.random.choice(np.arange(ind.shape[1]), num_selected, p=prob, replace=False) # does not work with bottom_edges_percent

            # --------randomize v2
            # monte calo randomize
            picked_edges_ind = []
            s = time.time()
            # index = 0
            all_indices = np.arange(ind.shape[1])
            #=====================
            #==select top edges first
            #=====================
            while len(picked_edges_ind) < top_num_selected:
                # index = index % ind.shape[1]
                index = np.random.choice(all_indices)

                random_num = np.random.uniform(0, 1, 1)
                if index not in picked_edges_ind and top_prob[index] > random_num:
                    # higher chance of selecting if prob is higher
                    picked_edges_ind.append(index)
                    all_indices = np.delete(all_indices, np.where(all_indices == index)[0][0])
                    # print(len(picked_edges_ind))
                # index += 1
            for i, j in zip(ind[0][picked_edges_ind], ind[1][picked_edges_ind]):
                # w = list(G.get_edge_data(i, j).values())
                w = tmp[i, j]
                # if len(w) > 0:
                if w > 0:
                    # add bidirectional edges
                    weighted_adj_matrix[i, j] = w
                    weighted_adj_matrix[j, i] = w  # todo
            print(f'picked top edges = {picked_edges_ind}')
            #=====================
            #==select bottom edges
            #=====================

            # --------randomize v2
            # monte calo randomize
            picked_edges_ind = []
            s = time.time()
            # index = 0
            # all_indices = np.arange(ind.shape[1]) # use the
            while len(picked_edges_ind) < bottom_num_selected:
                # index = index % ind.shape[1]
                index = np.random.choice(all_indices)

                random_num = np.random.uniform(0, 1, 1)
                if index not in picked_edges_ind and bottom_prob[index] > random_num:
                    # higher chance of selecting if prob is higher
                    picked_edges_ind.append(index)
                    all_indices = np.delete(all_indices, np.where(all_indices == index)[0][0]) # all_indived after the top percent is selected
                    # print(len(picked_edges_ind))
                # index += 1
            for i, j in zip(ind[0][picked_edges_ind], ind[1][picked_edges_ind]):
                # w = list(G.get_edge_data(i, j).values())
                w = tmp[i, j]
                # if len(w) > 0:
                if w > 0:
                    # add bidirectional edges
                    weighted_adj_matrix[i, j] = w
                    weighted_adj_matrix[j, i] = w  # todo

            print(f'picked bottom edges = {picked_edges_ind}')
        else:
            ind = added_edges

            low_to_high = tmp
            high_to_low = tmp[::-1]

            #=====================
            #==picked bottom_k
            #=====================
            # tmp = np.array([i for i in tmp.tolist()[0]][::-1])[:num_selected]  # 5254;

            # if tmp is not None:
            if len(tmp):
                max_val = np.amax(low_to_high[:bottom_num_selected])
                min_val = np.amin(high_to_low[:top_num_selected])

                # --------select all edges that have value more than threshold
                # bottom_num_left = bottom_num_selected - tmp[tmp < max_val].shape[0]  # num of edges  to be added.
                # top_num_left = top_num_selected - tmp[tmp > min_val].shape[0]  # num of edges  to be added.
                bottom_num_left = bottom_num_selected - tmp[tmp < max_val].shape[0]  # num of edges  to be added.
                max_val_left = tmp[tmp == max_val].shape[0]
                top_num_left = top_num_selected - tmp[tmp > min_val].shape[0]  # num of edges  to be added.
                min_val_left = tmp[tmp == min_val].shape[0]

                # if bottom_num_left > 1 or top_num_left > 1 :
                if max_val_left > 1 or min_val_left > 1 :
                    max_th_edges = [] # edges with value == thresh hold
                    max_th_edges_weight = []
                    min_th_edges = [] # edges with value == thresh hold
                    min_th_edges_weight = []

                    csr = csr_matrix(weighted_adj_matrix)
                    coo = csr.tocoo() # too lazy to import
                    for i,j,data in zip(coo.row, coo.col,coo.data):
                        # if data == min_val:
                        if data == max_val:
                            if j >= i: # garantee to have unique edges (independent of direction; i->j and j<-i is the same edges)
                                max_th_edges.append([i,j])
                                max_th_edges_weight.append(data)
                        if data == min_val:
                            if j >= i: # garantee to have unique edges (independent of direction; i->j and j<-i is the same edges)
                                min_th_edges.append([i,j])
                                min_th_edges_weight.append(data)

                    weighted_adj_matrix = np.where((weighted_adj_matrix < max_val) | (weighted_adj_matrix > min_val), weighted_adj_matrix, 0) # add edges with value > threshold

                    #--------picked value == threshold randomly
                    # if bottom_num_left > 1:
                    if min_val_left > 1:
                        max_picked_edges = np.random.choice(np.arange(len(max_th_edges)), bottom_num_left, p=np.ones(len(max_th_edges))/len(max_th_edges), replace=False)
                        # max_picked_edges = np.random.choice(np.arange(len(max_th_edges)), max_val_left, p=np.ones(len(max_th_edges))/len(max_th_edges), replace=False)
                        max_th_edges = np.array(max_th_edges)
                        max_th_edges_weight = np.array(max_th_edges_weight)
                        # picked_edges = np.random.choice(np.arange(num_left), num_left, p=np.ones(num_left)/num_left)
                        max_th_edges_weight = max_th_edges_weight[max_picked_edges]
                        max_picked_edges = max_th_edges[max_picked_edges, : ] # dim = (number of edges,2)

                        weighted_adj_matrix[max_picked_edges.T[0], max_picked_edges.T[1]] = max_th_edges_weight
                        weighted_adj_matrix[max_picked_edges.T[1], max_picked_edges.T[0]] = max_th_edges_weight

                    # if top_num_left > 1:
                    if min_val_left > 1:
                        min_picked_edges = np.random.choice(np.arange(len(min_th_edges)), top_num_left, p=np.ones(len(min_th_edges))/len(min_th_edges), replace=False)
                        # min_picked_edges = np.random.choice(np.arange(len(min_th_edges)), min_val_left, p=np.ones(len(min_th_edges))/len(min_th_edges), replace=False)
                        min_th_edges = np.array(min_th_edges)
                        min_th_edges_weight = np.array(min_th_edges_weight)
                        # picked_edges = np.random.choice(np.arange(num_left), num_left, p=np.ones(num_left)/num_left)
                        min_th_edges_weight = min_th_edges_weight[min_picked_edges]
                        min_picked_edges = min_th_edges[min_picked_edges, : ] # dim = (number of edges,2)

                        weighted_adj_matrix[min_picked_edges.T[0], min_picked_edges.T[1]] = min_th_edges_weight
                        weighted_adj_matrix[min_picked_edges.T[1], min_picked_edges.T[0]] = min_th_edges_weight



                    # raise ValueError("don't let code pass this point. i haven't check the correctness of this yet")

                # elif bottom_num_left == 1 and top_num_left == 1:
                elif min_val_left == 1 and min_val_left == 1:
                    # some of the original edges is not selected
                    weighted_adj_matrix = np.where((weighted_adj_matrix >= min_val) | (weighted_adj_matrix <= max_val),weighted_adj_matrix, 0)  # add edges with value > threshold
                    weighted_adj_matrix[original_edges[1], original_edges[0]] = 1
                    weighted_adj_matrix[original_edges[0], original_edges[1]] = 1

                else:
                    raise ValueError("edges are not added correctly in jaccard coefficent => top_bottom_percent")


    if shared_nodes_random_edges_percent is not None:
        tmp = weighted_adj_matrix[added_edges[0], added_edges[1]]  # check if both direction have the same value
        tmp = np.array([i for i in tmp.tolist()[0]])
        tmp = np.sort(tmp[tmp > 0].flatten())
        # print(tmp)

        num_selected = int(tmp.size * shared_nodes_random_edges_percent)

        ind = added_edges
        # --------picked value == threshold randomly
        # ind = np.array(weighted_adj_matrix[weighted_adj_matrix>0].nonzero())
        prob = np.array([1/ind.shape[1] for i in range(ind.shape[1])])

        # TODO here>> does undirect and directed graph have any effects on node2vec?
        tmp = np.copy(weighted_adj_matrix)
        weighted_adj_matrix[ind[0], ind[1]] = 0  # mask added edges with value 0
        weighted_adj_matrix[ind[1], ind[0]] = 0  # mask added edges with value 0

        # --------randomize v1

        picked_edges_ind = np.random.choice(np.arange(ind.shape[1]), num_selected, p=prob,
                                            replace=False)  # does not work with bottom_edges_percent
        print(f'picked edges = {picked_edges_ind}')
        for i, j in zip(ind[0][picked_edges_ind], ind[1][picked_edges_ind]):
            # w = list(G.get_edge_data(i, j).values())
            w = tmp[i, j]
            # if len(w) > 0:
            if w > 0:
                # add bidirectional edges
                weighted_adj_matrix[i, j] = w
                weighted_adj_matrix[j, i] = w

    #TODO here>>
    if all_nodes_random_edges_percent is not None:
        all_diseases = np.arange(geometric_dataset.num_disease)
        all_possible_added_edges = []
        for i in all_diseases:
            for j in all_diseases:
                all_possible_added_edges.append([i,j])
        all_possible_added_edges = np.array(all_possible_added_edges).T

        tmp = weighted_adj_matrix[all_possible_added_edges[0],all_possible_added_edges[1]]
        tmp = np.array([i for i in tmp.tolist()[0]])
        # tmp = np.sort(tmp[tmp > 0].flatten())
        # print(tmp)

        num_selected = int(tmp.size * all_nodes_random_edges_percent)

        ind = all_possible_added_edges
        # --------picked value == threshold randomly
        # ind = np.array(weighted_adj_matrix[weighted_adj_matrix>0].nonzero())
        prob = np.array([1/ind.shape[1] for i in range(ind.shape[1])])

        # TODO here>> does undirect and directed graph have any effects on node2vec?
        tmp = np.copy(weighted_adj_matrix)
        weighted_adj_matrix[ind[0], ind[1]] = 0  # mask added edges with value 0
        weighted_adj_matrix[ind[1], ind[0]] = 0  # mask added edges with value 0

        # --------randomize v1

        picked_edges_ind = np.random.choice(np.arange(ind.shape[1]), num_selected, p=prob,
                                            replace=False)  # does not work with bottom_edges_percent
        print(f'picked_edges = {picked_edges_ind}')

        for i, j in zip(ind[0][picked_edges_ind], ind[1][picked_edges_ind]):
            # w = list(G.get_edge_data(i, j).values())
            w = tmp[i, j]
            # if len(w) > 0:
            if w > 0:
                # add bidirectional edges
                weighted_adj_matrix[i, j] = w
                weighted_adj_matrix[j, i] = w

    if top_edges_percent is not None and bottom_edges_percent is not None:
        raise ValueError('either top_egdes_percent or bottom_edges_percent can be specified at a time')

    #--------select top n percent of the highest value
    if top_edges_percent is not None or bottom_edges_percent is not None:
        print(f"select top {top_edges_percent} percent  highest val")

        tmp = weighted_adj_matrix[added_edges[0], added_edges[1]]  # check if both direction have the same value
        tmp = np.array([i for i in tmp.tolist()[0]])
        tmp = np.sort(tmp[tmp > 0].flatten())
        # print(tmp)
        if top_edges_percent is not None:
            num_selected = int(tmp.size * top_edges_percent)
        if bottom_edges_percent is not None:
            num_selected = int(tmp.size * bottom_edges_percent)

        print(f"amount of added edges = {num_selected}")
        if args.stochastic_edges:
            #TODO here>> use new randomize technique and use it with
            print('stochastic_edges activated')
            # tmp = weighted_adj_matrix[added_edges[0], added_edges[1]]
            # list index of all edges
            # ind = np.array(csr_matrix(weighted_adj_matrix).nonzero())
            ind = added_edges

            # --------picked value == threshold randomly
            if top_edges_percent is not None:
                # ind = np.array(weighted_adj_matrix[weighted_adj_matrix>0].nonzero())
                prob = weighted_adj_matrix[ind[0],ind[1]]/weighted_adj_matrix[ind[0],ind[1]].sum()
                prob = np.array([i for i in prob.tolist()[0]])


            if bottom_edges_percent is not None:
                prob = weighted_adj_matrix[ind[0], ind[1]]
                prob = 1 / (prob * 20000)
                prob = np.where(prob == float('inf'), 0, prob)
                prob = np.array([i for i in prob.tolist()[0]])

            #TODO here>> does undirect and directed graph have any effects on node2vec?
            tmp = np.copy(weighted_adj_matrix)
            weighted_adj_matrix[ind[0], ind[1]] = 0 # mask added edges with value 0
            weighted_adj_matrix[ind[1], ind[0]] = 0  # mask added edges with value 0

            #--------randomize v1
            # # choose edge with no replacement
            # if True:
            #     raise ValueError("fixed randomization provess for bottom_k")

            # picked_edges_ind = np.random.choice(np.arange(ind.shape[1]), num_selected, p=prob, replace=False) # does not work with bottom_edges_percent

            #--------randomize v2
            # monte calo randomize
            picked_edges_ind = []
            s = time.time()
            # index = 0
            all_indices = np.arange(ind.shape[1])
            while len(picked_edges_ind) < num_selected:
                # index = index % ind.shape[1]
                index = np.random.choice(all_indices)

                random_num = np.random.uniform(0,1,1)
                if index not in picked_edges_ind and prob[index] > random_num:
                    #higher chance of selecting if prob is higher
                    picked_edges_ind.append(index)
                    all_indices = np.delete(all_indices, np.where(all_indices == index)[0][0])
                    # print(len(picked_edges_ind))
                # index += 1

            print(f'picked edges = {picked_edges_ind}')

            for i,j in zip(ind[0][picked_edges_ind], ind[1][picked_edges_ind]):
                # w = list(G.get_edge_data(i, j).values())
                w = tmp[i,j]
                # if len(w) > 0:
                if w > 0:
                    # add bidirectional edges
                    weighted_adj_matrix[i, j] = w
                    weighted_adj_matrix[j,i] = w # todo
        # if args.stochastic_edges:
        #     print('stochastic_edges activated')
        #     # tmp = weighted_adj_matrix[added_edges[0], added_edges[1]]
        #     # list index of all edges
        #     # ind = np.array(csr_matrix(weighted_adj_matrix).nonzero())
        #     ind = added_edges
        #
        #     # --------picked value == threshold randomly
        #     # ind = np.array(weighted_adj_matrix[weighted_adj_matrix>0].nonzero())
        #     prob = weighted_adj_matrix[ind[0],ind[1]]/weighted_adj_matrix[ind[0],ind[1]].sum()
        #     prob = np.array([i for i in prob.tolist()[0]])
        #
        #     weighted_adj_matrix[ind[0], ind[1]] = 0 # mask added edges with value 0
        #     # choose edge with no replacement
        #     picked_edges = np.random.choice(np.arange(ind.shape[1]), num_selected, p=prob, replace=False)
        #     for i,j in zip(ind[0][picked_edges], ind[1][picked_edges]):
        #         w = list(G.get_edge_data(i, j).values())
        #         if len(w) > 0:
        #             weighted_adj_matrix[i, j] = w[0] # todo
        else:
            # sorted from high to low
            if top_edges_percent is not None:
                tmp = tmp[::-1]

            # tmp = np.array([i for i in tmp.tolist()[0]][::-1])[:num_selected]  # 5254;
            tmp = tmp[:num_selected]
            min_val = None
            max_val = None
            # if tmp is not None:
            if len(tmp):
                if bottom_edges_percent is not None:
                    max_val = np.amax(tmp)
                    # --------select all edges that have value more than threshold
                    num_left = num_selected - tmp[tmp < max_val].shape[0]  # num of edges  to be added.
                    # top_num_left = top_num_selected - tmp[tmp > min_val].shape[0]  # num of edges  to be added.
                    val_left = tmp[tmp == max_val].shape[0]

                if top_edges_percent is not None:
                    min_val = np.amin(tmp)
                    #--------select all edges that have value more than threshold
                    num_left = num_selected - tmp[tmp > min_val].shape[0] # num of edges  to be added.
                    val_left = tmp[tmp == min_val].shape[0]

                if val_left > 1:
                    val = max_val if bottom_edges_percent is not None else min_val
                    th_edges = [] # edges with value == thresh hold
                    th_edges_weight = []
                    csr = csr_matrix(weighted_adj_matrix)
                    coo = csr.tocoo() # too lazy to import
                    for i,j,data in zip(coo.row, coo.col,coo.data):
                        # if data == min_val:
                        if data == val:
                            if j >= i: # garantee to have unique edges (independent of direction; i->j and j<-i is the same edges)
                                th_edges.append([i,j])
                                th_edges_weight.append(data)

                    #--------picked value == threshold randomly
                    picked_edges = np.random.choice(np.arange(len(th_edges)), num_left, p=np.ones(len(th_edges))/len(th_edges), replace=False)
                    th_edges = np.array(th_edges)
                    th_edges_weight = np.array(th_edges_weight)
                    # picked_edges = np.random.choice(np.arange(num_left), num_left, p=np.ones(num_left)/num_left)
                    th_edges_weight = th_edges_weight[picked_edges]
                    picked_edges = th_edges[picked_edges, : ] # dim = (number of edges,2)
                    if min_val is not None:
                        weighted_adj_matrix = np.where(weighted_adj_matrix > min_val, weighted_adj_matrix, 0) # add edges with value > threshold
                    if max_val is not None:
                        weighted_adj_matrix = np.where(weighted_adj_matrix < max_val, weighted_adj_matrix, 0) # add edges with value > threshold
                        weighted_adj_matrix[original_edges[1], original_edges[0]] = 1
                        weighted_adj_matrix[original_edges[0], original_edges[1]] = 1

                    weighted_adj_matrix[picked_edges.T[0], picked_edges.T[1]] = th_edges_weight
                    weighted_adj_matrix[picked_edges.T[1], picked_edges.T[0]] = th_edges_weight

                    # --------picked value == threshold randomly
                    # ind = np.where(weighted_adj_matrix==min_val) # make sure that there are no same edges with different direction
                    # picked_ind = np.random.choice(np.arange(num_left), num_left, p=np.ones(num_left)/num_left) # add these edges to networkx to get undirected edges
                    # weighted_adj_matrix = np.where(weighted_adj_matrix > min_val, weighted_adj_matrix, 0) # add edges with value > threshold
                    # weighted_adj_matrix[ind[0][picked_ind], ind[1][picked_ind]] = min_val # add edges with value == threshold

                elif val_left == 1:
                    # some of the original edges is not selected
                    if top_edges_percent is not None:
                        weighted_adj_matrix = np.where(weighted_adj_matrix >= min_val, weighted_adj_matrix, 0)

                    if bottom_edges_percent is not None:
                        weighted_adj_matrix = np.where(weighted_adj_matrix <= max_val, weighted_adj_matrix, 0) #
                        weighted_adj_matrix[original_edges[1], original_edges[0]] = 1
                        weighted_adj_matrix[original_edges[0], original_edges[1]] = 1
                else:
                    pass

        #TODO here>> create a assert to test number of added edges .
        # > how come weighted_adj_matrix non zero = 9016 is less than original edges * 2 = 9699
        # assert weighted_adj_matrix.nonzero()[0].shape[0] == (original_edges.shape[1] + num_selected) *2  # how should I test this??


    if isinstance(x, np.ndarray):
        print(f'max val = {np.amax(tmp)}')

    print(f'number of added edges {int(weighted_adj_matrix.nonzero()[0].shape[0])}')

    #=====================
    #==add self_loop on to the graph
    #=====================
    if self_loop:
        diag = np.arange(weighted_adj_matrix.shape[0])
        weighted_adj_matrix[diag, diag] = 1

    #=====================
    #==get edges_weight and ordered_edges
    #=====================
    tmp = csr_matrix(weighted_adj_matrix)
    # TODO here>> ordered_edges and edges_weight needs to be fixed
    for i, j in zip(*tmp.nonzero()):
        ordered_edges.append([i, j])
        edges_weight.append(tmp[i, j])
    print(f'number of added edged = {num_selected}')
    return weighted_adj_matrix, np.array(edges_weight), ordered_edges
    # return np.array(edges_weight)
    # return np.array([i[2] for i in preds])
