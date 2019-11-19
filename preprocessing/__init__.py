import sys,os
USER = os.environ['USERPROFILE']
sys.path.insert(1,f'{USER}\\PycharmProjects\\my_utility')

from utility_code.my_utility import *
from utility_code.python_lib_essential import *

from arg_parser import *
import all_datasets
# from my_utils import create_adj_list
# from my_utils import *
import my_utils
import networkx as nx
import collections
from collections import OrderedDict
from operator import itemgetter

def data_preprocessing(dataset = None, name='copd'):
    assert dataset is not None, "In run_preprocessing, dataset must not be none"

    if name=='copd':
        # run preprocessing for copd
        # if args.emb_path is not None:

        # if args.emb_path is not None or args.emb_name != "no_feat":
        if args.emb_path is not None:
            x   = add_features()
        elif args.emb_name in ['gnn', 'no_feat']:
            # --------without features; instead use identity matrix of n*n where n is number of nodes
            x = np.identity(len(dataset.nodes2idx().keys()))
            x = torch.tensor(x, dtype=torch.float)
        else:
            raise ValueError('please specified emb_name or emb_path')

        # else:
        #     x = add_features()

        # --------edge_index
        # tmp = np.unique(copd.edges)
        # todo this is not in order
        edge_index = list(map(dataset.nodes2idx().get, dataset.edges.T.flatten()))
        edge_index = torch.tensor(edge_index, dtype=torch.int64).view(-1,2 ).transpose(0,1)  # torch.Size([2, 4715])

        # --------label gene with 0, 1, 2, 3, 4, 5
        # note that order of node2idx is maintain (you can check in dataset.node2idx())
        y = [dataset.disease2class()[i] if i in dataset.disease2idx().values() else len(dataset.class2disease().keys()) for i in
             dataset.nodes2idx().values()] # todo how can I speed this up? save it somewhere?

        y = torch.tensor(y, dtype=torch.int64)  # torch.Size([2996])

        return x, dataset, edge_index, y


# create funciton as
def create_onehot(edges_dict, geometric_dataset, edges):
    # -- create genes as onehot
    onehot_genes = create_onehot_to_be_merged(edges_dict, edges)

    # -------- merge_onehot
    all_x_input = merge_onehot(onehot_genes, geometric_dataset)

    return all_x_input
    # test_input = normalize_features(csr_matrix(np.array(test_input)))
    # train_input = normalize_features(csr_matrix(np.array(train_input)))
    # test_input = normalize_features(csr_matrix(np.array(test_input)))

    # --------convert numpy array to torch.tensor
    # train_input = torch.tensor(train_input)
    # test_input = torch.tensor(test_input)
    # return train_input, test_input

def plot_shared_nodes_distribution(nodes_with_shared_genes=None, used_nodes=None):
    '''

    :param use:
    :return:
    '''

    shared_gene_dist = []
    shared_disease_dist = []
    for th in range(0, len(nodes_with_shared_genes)):
        disease = []
        gene = []
        for d1, g1 in nodes_with_shared_genes.items():
            if d1 <= 101:  # disease
                # all_diseases = [i for i in nodes_with_shared_genes.items()]
                for i, (d2, g2) in enumerate(nodes_with_shared_genes.items()):
                    if i >= 101:
                        break

                    if len(set(g1).intersection(set(g2))) > th and d1 != d2 and (d1, d2) not in disease and (
                            d2, d1) not in disease:
                        disease.append([d1, d2])

            if d1 > 101:
                for i, (d2, g2) in enumerate(nodes_with_shared_genes.items()):
                    if i >= 101:
                        if len(set(g1).intersection(set(g2))) > th and d1 != d2 and (d1, d2) not in gene and (
                                d2, d1) not in gene:
                            gene.append([d1, d2])

        shared_gene_dist.append({th: len(disease)}) # diseases taht shared genes
        shared_disease_dist.append({th: len(gene)}) # diseases taht shared genes

        # nodes_with_shared_genes = tmp
    disease_x = []
    disease_y = []
    gene_x = []
    gene_y = []
    for i,j in zip(shared_gene_dist, shared_disease_dist): # todo  this is not currently correct
        disease_x.append(list(i.keys())[0])
        disease_y.append(list(i.values())[0])
        gene_x.append(list(j.keys())[0])
        gene_y.append(list(j.values())[0])

    disease_gene = {}
    if len(disease_x) > 0:
        disease_gene['disease'] = [disease_x,disease_y]
    if len(gene_x) > 0:
        disease_gene['gene'] = [gene_x, gene_y]

    config = {
        f'nodes_with_shared_genes_dist {used_nodes}': {
            'x_label': 'number of shared genes',
            'y_label': 'number of nodes',
            'legend': [{"kwargs": {"loc": "lower right"}}],
            'plot': [{'args':j, "kwargs": {'label': f'{i}'}} for i,j in disease_gene.items()]
        }
    }
    plot_figures(config)

    # if used_nodes == 'all':
    #     disease_x = []
    #     disease_y = []
    #     gene_x = []
    #     gene_y = []
    #     for i in shared_gene_dist:
    #         if i <= 101:
    #             disease_x.append(list(i.keys())[0])
    #             disease_y.append(list(i.keys())[1])
    #         else:
    #             gene_x.append(list(i.keys())[0])
    #             gene_y.append(list(i.keys())[1])
    #
    #     config = {
    #         f'nodes_with_shared_genes_dist {used_nodes}': {
    #             'x_label': 'number of shared genes',
    #             'y_label': 'number of nodes',
    #             'legend': [{"kwargs": {"loc": "lower right"}}],
    #             'plot': [
    #                 {"args": [[list(i.keys())[0] for i in shared_gene_dist],
    #                           [list(i.values())[0] for i in shared_gene_dist]]},
    #                 {"args": [[list(i.keys())[0] for i in shared_gene_dist] ,
    #                           [list(i.values())[0] for i in shared_gene_dist]]},
    #             ]
    #         }
    #     }
    # else:
    #     config = {
    #         f'nodes_with_shared_genes_dist {used_nodes}': {
    #             'x_label': 'number of shared genes',
    #             'y_label': 'number of nodes',
    #             'legend': [{"kwargs": {"loc": "lower right"}}],
    #             'plot': [
    #                 {"args": [[list(i.keys())[0] for i in shared_gene_dist],
    #                           [list(i.values())[0] for i in shared_gene_dist]]},
    #                 # {"args": [ list(range(0, len(shared_gene_dist))), [0 for i in range(0,len(shared_gene_dist))]]},
    #             ]
    #         }
    #     }
    # plot_figures(config)

def get_added_edges_from_nodes_with_shared_genes(edges, used_nodes, plot_shared_gene_dist):
    # TODO here>> create_edges_dict support option use=all, gene, disease??
    edges_dict, nodes_with_shared_genes = my_utils.create_edges_dict(edges,
                                                                     used_nodes)  # return {disease: [{gene, weight}, ... ]} where list of genes are genese that connected to disease by an edge.
    # nodes_with_shared_genes = { key: [list(i.keys())[0]  for i in j ] for key, j in edges_dict.items()} # rearrange to {disease: [genes,...]}

    # plot_shared_gene_dist = True
    # --------plot_shread_nodes_distribution
    # plot_shared_gene_dist = True
    if plot_shared_gene_dist:
        # TODO here>> takes too long for gene and disease
        run_time = timer(plot_shared_nodes_distribution, nodes_with_shared_genes,
                         used_nodes)  # plot and choose th for gene and disease)
        print(f"plot_shared_nodes_distribution takes {run_time} ms to run ")

    def get_added_edges(nodes_with_shared_genes, used_nodes):
        '''

        :param nodes_with_shared_genes:
        :param used_nodes:
        :return:
        '''
        # the function is created for readability
        tmp = []
        # selected = []
        nodes_shared_count = {}
        for i, (d1, g1) in enumerate(nodes_with_shared_genes.items()):
            if len(list(nodes_with_shared_genes.items())[i + 1:]) == 0:
                break
            else:
                for d2, g2 in list(nodes_with_shared_genes.items())[i + 1:]:
                    nodes_shared_count.setdefault(d1 * d2, len(
                        set(g1).intersection(set(g2))))  # key = g1 * g2 because it produce unique number for each pair
                    if used_nodes == 'gene':  # add edges between disease
                        if nodes_shared_count.get(d1 * d2) > 0:  # this is slow
                            tmp.append((d1, d2))
                    elif used_nodes == 'disease':  # add edges between genes
                        if nodes_shared_count.get(g1 * g2) > 0:  # this is slow
                            tmp.append((g1, g2))
                    else:
                        # add edges between genes-genes and diseases-diseases
                        if nodes_shared_count.get(d1 * d2) > 0:  # this is slow
                            tmp.append((d1, d2))
                        if nodes_shared_count.get(g1 * g2) > 0:  # this is slow
                            tmp.append((g1, g2))

                    # if nodes_shared_count.get(d1*d2) > 0 and d1 != d2 and d1*d2 not in selected: # this is slow
                    #     tmp.append((d1,d2))
                    # selected.append(d1*d2)

                    # if len(set(g1).intersection(set(g2))) > 0 and d1 != d2 and (d1,d2) not in tmp and (d2,d1) not in tmp: # this is slow
                    #     tmp.append((d1,d2))
        return tmp

    return get_added_edges(nodes_with_shared_genes, used_nodes)

def add_edges_with_shared_nodes(dataset, geometric_dataset, edges, used_nodes, plot_shared_gene_dist, edges_weight_option,save_path):
    # added_edges =
    added_edges = get_added_edges_from_nodes_with_shared_genes(edges, used_nodes, plot_shared_gene_dist)
    # added_edges = get_added_edges(nodes_with_shared_genes, used_nodes)

    max_node = len(list(dataset.nodes2idx()))

    #--------create symmetric adj
    before_added_edges = np.array(edges).T
    before_added_edges_adj = csr_matrix((np.ones_like(before_added_edges)[0], (before_added_edges[0], before_added_edges[1]))).todense() # dim = num_disease * num_nodes
    before_added_edges_adj = np.vstack((before_added_edges_adj, np.zeros((max_node - before_added_edges_adj.shape[0], before_added_edges_adj.shape[1])))) # dim = num_nodes * num_nodes
    before_added_edges_adj = before_added_edges_adj + before_added_edges_adj.transpose() - before_added_edges_adj.diagonal() # symmetric_adj ; dim = num_nodes * num_nodes

    original_edges = edges
    edges = edges + added_edges # (edges= 4715 + added_edges = 539) = 5254
    edges = np.array(edges).T
    #--------get weight edges (networkx function should preserve order)
    edges_weight = None
    weighted_adj = None
    if edges_weight_option == 'jaccard':
        from edge_weight import jaccard_coeff
        # weighted_adj, edges_weight, edges = jaccard_coeff(dataset, geometric_dataset, original_edges, added_edges, edges, mask_edges=args.mask_edges, weight_limit=args.edges_weight_limit, self_loop=args.self_loop, edges_percent=args.edges_percent)
        weighted_adj, edges_weight, edges = jaccard_coeff(dataset, geometric_dataset, original_edges, added_edges, edges, mask_edges=args.mask_edges, weight_limit=args.edges_weight_limit, self_loop=args.self_loop,
                                                          weight_limit_percent=args.edges_weight_percent, top_edges_percent=args.top_percent_edges, bottom_edges_percent=args.bottom_percent_edges,
                                                          shared_nodes_random_edges_percent=args.shared_nodes_random_edges_percent,all_nodes_random_edges_percent=args.all_nodes_random_edges_percent,
                                                          top_bottom_percent=args.top_bottom_percent_edges)
        np.save(f'{save_path}\weighted_adj_option={edges_weight_option}_weight_limit={args.edges_weight_limit}.txt', weighted_adj)
        np.save(f'{save_path}\edges_weight_option={edges_weight_option}_weight_limit={args.edges_weight_limit}.txt', edges_weight)
        print(f'saveing weight_adj and edge_weight (option={edges_weight_option} weight_limit={args.edges_weight_limit}) at {save_path}')

    if edges_weight_option == 'no':
        G = nx.Graph()
        G.add_edges_from(zip(edges[0], edges[1]))  # 5254
        weighted_adj = nx.to_numpy_matrix(G) # all edges have equal weight of 1
        edges_weight = np.ones((weighted_adj.nonzero()[0].shape[0]))
        np.save(f'{save_path}\weighted_adj_option={edges_weight_option}.txt', weighted_adj)
        np.save(f'{save_path}\edges_weight_option={edges_weight_option}.txt', edges_weight)
        print(f'saveing weight_adj and edge_weight( option= {edges_weight_option}) at {save_path}')

    return weighted_adj, edges_weight

def add_edges_with_longest_path(dataset, geometric_dataset, edges, used_nodes, plot_shared_gene_dist,edges_weight_option, save_path, percent):
    weighted_adj, edges_weight = None, None

    def Merlin_pandas(l):
        '''return dict that after apply cumsum to pandas'''
        #     df = pd.DataFrame(l).rename(columns=tmp)
        df = pd.DataFrame(l, index=[0])
        #     print(df)
        df = pd.concat([df] * 2, ignore_index=True)
        df.iloc[1] = df.iloc[0].cumsum()
        #     print(df)
        tmp = df
        return tmp

    G = nx.Graph()
    G.add_edges_from(edges)
    #=====================
    #==get biggest disconnected subgraph, so it consistent with node2vec
    #=====================

    disconnected_graph = list(nx.connected_component_subgraphs(G))
    disconnected_graph = [(disconnected_graph[i], len(g)) for i, g in enumerate(disconnected_graph)]

    from operator import itemgetter

    disconnected_graph = sorted(disconnected_graph, key=itemgetter(1), reverse=True)
    # print(disconnected_graph)

    # disconnected_graph = [subgraph1, subgraph2, ....] #where subgraph is of type networkx
    biggest_disconnected_graph = [graph for graph, length in disconnected_graph][0]
    G = biggest_disconnected_graph

    #=====================
    #==get length of pairwise nodes + sorted in decending order
    #=====================

    # length = nx.all_pairs_shortest_path_length(G)
    length = {}
    for i in range(dataset.num_diseases):
        tmp1 = {}
        for j in range(dataset.num_diseases):
            try:
                tmp1[j] = nx.shortest_path_length(G, source=i, target=j)
            except nx.NodeNotFound:
                pass


        length.setdefault('tmp', []).append([i, tmp1])
    length = length['tmp']
    # print(len(G.edges))
    # print(len(edges))
    def pick_longest_path(length, amount=None):
        tmp = {}
        count_same_val = collections.OrderedDict({})
        for i in length:
            #     print(i)
            for j, val in i[1].items():
                if j >= i[0]:
                    tmp[f'{i[0]}_{j}'] = val
                    if count_same_val.setdefault(val, None) is None:
                        count_same_val[val] = 1
                    else:
                        count_same_val[val] += 1
                        #     tmp = sorted((v,k) for k,v in tmp.items())[::-1]
        tmp = np.array(sorted(tmp.items(), key=lambda tmp: tmp[1], reverse=True))
        new_edges = tmp[:, 0]
        tmp = tmp[:, 1].astype(int)

        count_same_val = OrderedDict(sorted(count_same_val.items(), key=itemgetter(0), reverse=True))
        ind = 0
        print(f'tmp = {tmp}')
        print(f'count_same_val = {count_same_val}')
        # apply cumsum to pandas
        df = Merlin_pandas(count_same_val)  # cumsum with the same key
        print(f'df = {df}')
        picked = None
        cumsum = df.iloc[1].to_dict()
        for i, (k, v) in enumerate(cumsum.items()):
            if i == 0 and v < amount:
                picked = v
            if i == 0 and v > amount:
                picked = amount
                break
            #             print(picked)
            if v <= amount:
                picked = v

        picked = list(range(picked))
        print(f'picked = {picked}')
        left_num = amount - len(picked)
        if left_num > 0:
            tmp = tmp[tmp == tmp[max(picked) + 1]]
            print(tmp)
            prob = [1 / len(tmp) for i in range(len(tmp))]
            more_picked = np.random.choice(range(len(tmp)), left_num, p=prob, replace=False)
            more_picked += len(picked)
            more_picked = list(more_picked)
            print(more_picked)
            picked += more_picked
            print(picked)

        print(f"picked = {picked}")
        # get edgse
        picked_edges = new_edges[picked]
        picked_edges = np.array([i.split('_') for i in picked_edges])
        picked_edges = picked_edges.astype(np.int)

        return picked_edges

    #TODO here>> add same number of edges to 0.05- 0.5
    if percent == 0:
        amount = 0
    if percent == 0.05:
        amount = 24
    if percent == 0.1:
        amount = 48
    if percent == 0.2:
        amount = 97
    if percent == 0.3:
        amount = 146
    if percent == 0.4:
        amount = 195
    if percent == 0.5:
        amount = 244
    print('========before ============')
    print(len(G.edges))
    weighted_adj = nx.to_numpy_matrix(G, nodelist=list(range(dataset.num_nodes)))
    # print(weighted_adj.nonzero()[0].shape[0])
    picked_edges = pick_longest_path(length, amount=amount)
    print('========after============')
    G.add_edges_from(picked_edges)
    print(len(G.edges))
    weighted_adj = nx.to_numpy_matrix(G, nodelist=list(range(dataset.num_nodes)))
    edges_weight = np.array([1 for i in range(weighted_adj.nonzero()[0].shape[0])])
    print(weighted_adj.nonzero()[0].shape[0])
    #TODO here>> what does edges_weight support to look like?

    return weighted_adj, edges_weight
def added_edges_with_same_class(dataset, geometric_dataset, edges, used_nodes, plot_shared_gene_dist,edges_weight_option, save_path, percent):
     x = geometric_dataset.x
     y = geometric_dataset.y

     G = nx.Graph()
     G.add_edges_from(edges)
     # =====================
     # ==get biggest disconnected subgraph, so it consistent with node2vec
     # =====================

     disconnected_graph = list(nx.connected_component_subgraphs(G))
     disconnected_graph = [(disconnected_graph[i], len(g)) for i, g in enumerate(disconnected_graph)]
     #
     from operator import itemgetter

     disconnected_graph = sorted(disconnected_graph, key=itemgetter(1), reverse=True)
     # print(disconnected_graph)

     # disconnected_graph = [subgraph1, subgraph2, ....] #where subgraph is of type networkx
     biggest_disconnected_graph = [graph for graph, length in disconnected_graph][0]
     G = biggest_disconnected_graph

    #=====================
    #==
    #=====================

     added_edges = []
     for i,val1 in enumerate(y):
         if i == geometric_dataset.num_disease:
             break
         for j,val2 in enumerate(y):
             if i > j and val1 == val2 : # no repeated edges and in the same class
                 added_edges.append([i,j]) # 1153


     G.add_edges_from(added_edges)
     weighted_adj = nx.to_numpy_matrix(G, nodelist=list(range(dataset.num_nodes)))
     edges_weight = np.array([1 for i in range(weighted_adj.nonzero()[0].shape[0])])
     print(weighted_adj.nonzero()[0].shape[0])

     return weighted_adj, edges_weight


def add_edges_with_no_shared_nodes(dataset, geometric_dataset, edges, used_nodes, plot_shared_gene_dist,edges_weight_option, save_path, percent):
    if percent == 0:
        amount = 0
    if percent == 0.05:
        amount = 24
    if percent == 0.1:
        amount = 48
    if percent == 0.2:
        amount = 97
    if percent == 0.3:
        amount = 146
    if percent == 0.4:
        amount = 195
    if percent == 0.5:
        amount = 244

    weighted_adj, edges_weight = None, None
    added_edges = get_added_edges_from_nodes_with_shared_genes(edges, used_nodes, plot_shared_gene_dist)
    all_edges = []
    for i in range(geometric_dataset.num_disease):
        for j in range(geometric_dataset.num_disease):
            if i < j:
                all_edges.append((i, j))
    # sorted(added_edges, key=lambda tup: tup[0], reverse=True)
    all_edges = set(all_edges)
    added_edges = set(added_edges)
    no_shared_gene = all_edges.difference(added_edges) # 4511

    prob = [1/len(no_shared_gene) for i in range(len(no_shared_gene))]
    # get number of edges to picked per nodes alpha value
    picked = np.random.choice(range(len(no_shared_gene)), amount, p=prob, replace=False)
    no_shared_gene = np.array(list(no_shared_gene))
    added_edges = no_shared_gene[picked]


    x = geometric_dataset.x
    y = geometric_dataset.y

    G = nx.Graph()
    G.add_edges_from(edges)
    # =====================
    # ==get biggest disconnected subgraph, so it consistent with node2vec
    # =====================

    disconnected_graph = list(nx.connected_component_subgraphs(G))
    disconnected_graph = [(disconnected_graph[i], len(g)) for i, g in enumerate(disconnected_graph)]
    #
    from operator import itemgetter

    disconnected_graph = sorted(disconnected_graph, key=itemgetter(1), reverse=True)
    # print(disconnected_graph)


    # disconnected_graph = [subgraph1, subgraph2, ....] #where subgraph is of type networkx
    biggest_disconnected_graph = [graph for graph, length in disconnected_graph][0]
    G = biggest_disconnected_graph


    G.add_edges_from(added_edges)
    weighted_adj = nx.to_numpy_matrix(G, nodelist=list(range(dataset.num_nodes)))
    edges_weight = np.array([1 for i in range(weighted_adj.nonzero()[0].shape[0])])

    return weighted_adj, edges_weight

def create_common_nodes_as_features(dataset, geometric_dataset, plot_shared_gene_dist = False, used_nodes='gene', edges_weight_option='jaccard', added_edges_option='shared_gene', percent=None):
    '''
        gene is a feat of disease if there exist edges between gene and disease nodes
    :param dataset:
    :param edge_index:
    :param gene: common genes as feature (create edges between disease nodes)
    :param disease: common disease as feature (create edge between genes nodes)
    :return:
    '''
    assert used_nodes != "no", "--common_nodes_feat must be specifed option = ['all', 'gene', 'disease']"
    save_path = r"C:\Users\awannaphasch2016\PycharmProjects\disease_node_classification\data\gene_disease\07_14_19_46\processed\numpy"
    weight_adj_file = f"\weighted_adj_option={args.edges_weight_option}.txt.npy"
    edges_weight_file = f'\edges_weight_option={args.edges_weight_option}.txt.npy'

    # if os.path.isfile(save_path+weight_adj_file) and os.path.isfile(save_path+edges_weight_file):
    #     weight_adj = np.load(save_path+weight_adj_file)
    #     edges_weight = np.load(save_path+weight_adj_file)
    #     print(f'load weight_adj and edge_weight (option={edges_weight_option})from {save_path}')
    #     return weight_adj, edges_weight, None

    #--------convert edges to acceptable format
    edges = [[i, j] if int(i) < len(dataset.disease2idx().values()) else (j, i) for (i, j) in
             zip(geometric_dataset.edge_index[0].numpy(), geometric_dataset.edge_index[1].numpy())]  # [(disease_id, gene_id), ... ]
    edges = list(map(lambda t: (int(t[0]), int(t[1])), edges))
    edges = sorted(edges, reverse=False, key=lambda t: t[0])

    # create edges between disease if it has shared genes
    # get distribution of number of gene shared between mulitple disease nodes
    '''
     graph = {source_node1: [{target_node: weight}, ... ]
                source_node2: [{target_node: weight}, ... ]
                ,....,}
    '''
    #=====================
    #==code below is really really bad and slow.
    #=====================
    if added_edges_option == 'no_shared_gene':
        weighted_adj, edges_weight = add_edges_with_no_shared_nodes(dataset, geometric_dataset, edges, used_nodes,
                                                                 plot_shared_gene_dist, edges_weight_option, save_path, percent)
    if added_edges_option == 'shared_gene':
        weighted_adj, edges_weight= add_edges_with_shared_nodes(dataset, geometric_dataset, edges, used_nodes, plot_shared_gene_dist,edges_weight_option, save_path)

    if added_edges_option == 'longest_path':
        weighted_adj, edges_weight = add_edges_with_longest_path(dataset, geometric_dataset, edges, used_nodes, plot_shared_gene_dist,edges_weight_option, save_path, percent)
    if added_edges_option == 'same_class':
        weighted_adj, edges_weight = added_edges_with_same_class(dataset, geometric_dataset, edges, used_nodes,
                                                                 plot_shared_gene_dist, edges_weight_option, save_path,
                                                                 percent)
    #TODO here>> trying out differnet weight for edges between diseases
    # > create gene to gene nodes and apply the same edge weight criteria

    #--------mask symmetric adj of original edges to weighted_adj where nodes of the same type are connected (there is no need to mask because jaccard value of edges between disease and nodes are no longer zero after edges are added)
    # weighted_adj = csr_matrix((1, (edges[0], edges[1])), shape=(max(max_node), max(max_node))).todense()

    #--------convert to n*p format where n = number of nodes and p = nmber of features.
    # edges_dict, _ = my_utils.create_edges_dict(edges, use_nodes=used_nodes)
    # all_x_input = create_onehot(edges_dict, geometric_dataset, edges)

    # if used_nodes == 'gene': # i am not sure if this is needed
    #     # dim = number of disease * num_all_nodes
    #     weighted_adj = weighted_adj[range(0,geometric_dataset.x.shape[0])]

    #--------update copd_geometric_dataset
    # edges = np.array(edges) # i am not sure why I nee dto convert to array then back to list to convert to tensor
    edges = np.array(weighted_adj.nonzero()).T
    # print(edges.T.shape)
    geometric_dataset.edges_index = torch.tensor(edges.tolist()) # why can't i convert
    geometric_dataset.edge_index = torch.transpose(torch.tensor(edges.tolist()), 0,1 ) # why can't i convert

    geometric_dataset.edges_weight = torch.from_numpy(edges_weight).type(torch.float64)
    geometric_dataset.x = torch.from_numpy(weighted_adj).type(torch.float64)

    return weighted_adj, edges_weight, edges



def merge_onehot(onehot_matrix, geometric_dataset):
    '''
    -- convert onehot input into the following format
    from
      {disease_idx1: [[0,0,0,1,0,0],[0,1,0,0,0,0] ....], disease_idx2: [...],... }
    to
      {disease_idx1: [0,1,0,1,0,0], disease_idx2: [...],... }

    :return:
    '''
    train_input = []
    test_input = []
    #TODO here>> test and train should not be done here. merge_onehot should just merge_onehot
    for key, val in onehot_matrix.items():
        sum = 0
        for v in val:
            sum=np.add(sum,v)
        onehot_matrix[key]= sum
    return np.array([i for i in onehot_matrix.values()])
    # display2screen(onehot_matrix.values(), np.array([i for i in onehot_matrix.values()]))

    #--------convert onehot_matrix to numpy array

    # display2screen(onehot_matrix)
        # if int(key) in geometric_dataset.train_mask:
        #     for v in val:
        #         sum = np.add(sum, v)
        #     onehot_matrix[key] = sum
        #     train_input.append(onehot_matrix[key])
        # sum1 = 0
        # if int(key) in geometric_dataset.test_mask:
        #     for v in val:
        #         sum1 = np.add(sum1, v)
        #     onehot_matrix[key] = sum1
        #     test_input.append(onehot_matrix[key])

    return train_input, test_input

def normalize_features(mx):
    """
        # Row-normalize sparse matrix
        col-normalize sparse matrix
    :param: mx: csr_matrix
    :return mx: numpy array
    """

    # rowsum = np.array(mx.sum(1))
    rowsum = np.array(mx.sum(0))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx.todense()

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt) # D^0-5 * A * D^0.5

def create_onehot_to_be_merged(edges_dict, edges):
    '''
    param: adj_list
        adj_list must be in the followin format:
        adj_list = {source_node1: [{target_node: weight}, ... ]
                source_node2: [{target_node: weight}, ... ]
                ,....,}

    :param edges:
        [(disease1, gene1), (disease2,gene2.....] where disease are sorted in ascending order
        eg. [(0,106),(1,400),(1,203),... ]

    :return: onehot is in the following format
        assuming the followin adj_list is provided as input

            1: [{1: '1'}, {3: '1'}]
        onehot will be
            disease_idx: [[0,1,0,0,0,0], [0,0,0,1,0,0]]
            disease_1 has 2 one hot vector
                :one hot at index 1 and one hot at index 3

    '''
    # -- built-in max() give the wrong result
    max_idx = np.amax(np.array(edges).flatten()) # 2995

    identity_matrix = np.identity(max_idx + 1)

    onehot = {i:[] for i in edges_dict.keys()}
    for key, val in edges_dict.items():
        # print(int(key))
        onehot[int(key)] = np.asarray([identity_matrix[int(list(k.keys())[0]),:] for k in val])

    return onehot

def add_features():
    # ===========================
    # === add embedding as features
    # ===========================
    # -- emb_path
    # if args.emb_path is None:

    if args.emb_name in ['attentionwalk', 'node2vec', 'bine','gat']:
        # emb_path = f"data/gene_disease/{args.time_stamp}/gene_disease/processed/embedding/" + emb_file

        # -- emb_file
        if args.emb_name == 'attentionwalk':
            emb_file = f"{args.emb_name}/{args.emb_name}_emb{args.time_stamp}.txt" # todo name is missing parameters that were used to generate emb

        elif args.emb_name == 'node2vec':
            if args.subgraph:
                emb_file = f"{args.emb_name}/{args.emb_name}_emb_subgraph{args.time_stamp}.txt" # todo name is missing parameters that were used to generate emb
                # emb_file = f"{args.emb_name}/{args.emb_name}_emb_subgraph_common_nodes_feat=True{args.time_stamp}.txt" # todo name is missing parameters that were used to generate emb
            else:
                # emb_file = f"{args.emb_name}/{args.emb_name}_emb_fullgraph{args.time_stamp}.txt" # todo name is missing parameters that were used to generate emb
                # emb_file = r'node2vec/node2vec_emb_fullgraph_common_nodes_feat=gene07_14_19_46_added_edges=disease_jaccard_weight_limit=1_mask=True.txt'
                # emb_file = r'node2vec/node2vec_emb_fullgraph_common_nodes_feat=gene07_14_19_46_added_edges=disease_jaccard_weight_limit=1_mask=F_no_selfloop.txt'
                # emb_file = r'node2vec/node2vec_emb_fullgraph_common_nodes_feat=gene07_14_19_46_added_edges=disease_jaccard_weight_limit=1_mask=True_selfloop.txt'
                # emb_file = f"{args.emb_name}/{args.emb_name}_emb_fullgraph_common_nodes_feat=True{args.time_stamp}.txt"
                # emb_file = r"node2vec/node2vec_emb_fullgraph_common_nodes_feat=gene07_14_19_46_added_edges=disease_jaccard_weight_limit=0.1_mask=True.txt"
                # emb_file = r"node2vec/node2vec_emb_fullgraph_common_nodes_feat=gene07_14_19_46_added_edges=disease_jaccard_weight_limit=0.9_mask=True.txt"
                # emb_file = r"node2vec/node2vec_emb_fullgraph_common_nodes_feat=gene07_14_19_46_added_edges=disease_jaccard_weight_limit=0.5_mask=True.txt"
                # emb_file = r"node2vec/node2vec_emb_fullgraph_common_nodes_feat=gene07_14_19_46_added_edges=disease_jaccard_weight_limit=1.0_mask=True.txt"
                # emb_file = r"node2vec/node2vec_emb_fullgraph_common_nodes_feat=gene07_14_19_46_added_edges=disease_jaccard_weight_limit=None_mask=True_stoch.txt"
                # emb_file = r"node2vec/node2vec_emb_fullgraph_common_nodes_feat=gene07_14_19_46_added_edges=disease_jaccard_top_k=0.25_mask=True_stoch.txt"
                # emb_file = r"node2vec/node2vec_emb_fullgraph_common_nodes_feat=gene07_14_19_46_added_edges=disease_jaccard_top_k=0.05_mask=True_stoch.txt"
                emb_file = r"node2vec/node2vec_emb_fullgraph_common_nodes_feat=gene07_14_19_46_added_edges=disease_jaccard_top_k=0.1_mask=True_stoch.txt"
                # emb_file = r"node2vec/node2vec_emb_fullgraph_common_nodes_feat=gene07_14_19_46_added_edges=disease_jaccard_top_k=0.1_mask=True_stoch.txt"

        elif args.emb_name == 'bine':
            emb_file = f"{args.emb_name}/bine{args.time_stamp}.txt" # todo name is missing parameters that were used to generate emb

        # elif args.emb_name == 'gcn':
        #
        #     #-------- save to file
        #
        #
        #     raise ValueError("Please provide path to gcn_emb file. "
        #                      "Provided emb_name == gcn is not YET supported! (I cannot come but with default name for gcn)")
        elif args.emb_name == 'gat':
            raise ValueError("Please provide path to gat_emb file. "
                             "provided emb_name == gat is not YET supported! (I cannot come but with default name for gat)")
        elif args.emb_name == 'graph_sage':
            raise ValueError("Please provide path to graph_sage_emb file. "
                             "provided emb_name == graph_sage is not YET supported! (I cannot come but with default name for graph_sage)")

        else:
            raise ValueError("provided emb_name is not supported!")

        assert emb_file is not None, f"{args.emb_name} is not available"
        emb_path = f"data/gene_disease/{args.time_stamp}/processed/embedding/" + emb_file

        with open(emb_path, 'r') as f:
            tmp = f.readlines()
            if "bine" not in emb_file:
                tmp = tmp[1:]


        # -- split symbol
        if args.emb_name == "attentionwalk":
            split = ','
        if args.emb_name == "node2vec":
            split = ' '
        if args.emb_name == "bine":
            split = ' '

        #--------{node_idx: emb_value}
        if args.emb_name == "bine":
            emb_dict = {int(float(i.split(split)[0][1:])): list(map(float, i.split(split)[1:])) for i in tmp}
        else:
            emb_dict = {int(float(i.split(split)[0])): list(map(float, i.split(split)[1:])) for i in tmp}

    else:
        # emb_dict = {int(float(i.split(split)[0][1:])): list(map(float, i.split(split)[1:])) for i in tmp}
        # df.to_csv(save_path + 'emb/' + file_gcn_emb, header=True, index=False, sep='\t', mode='w')
        #TODO here>> Is index of emb_path (save by the code) above aranged in order? (0-->2996)
        # > make sure that it is in order.
        #TODO here>> make it compatible with gcn and node2vec
        print(f'run emb from {args.emb_path}')

        emb_name = args.emb_path.split('\\')[-1]
        emb_name = emb_name.split('_')[0]
        # if emb_name == 'node2vec':
        emb_np = pd.read_csv(args.emb_path, sep=' ', skiprows=1, header=None).to_numpy()
        df = pd.DataFrame(emb_np[:, 1:], index=emb_np[:, 0])
        emb_dict = df.to_dict('index')
        for k, v in emb_dict.items():
            x = []
            for k1, v1 in v.items():
                x.append(v1)
            emb_dict[k] = x

        # order of node in emb_path preserved. (check gene2idx() for confirmation)

    # -- make sure that node embs are in ordered
    emb = sorted(emb_dict.items(), key=lambda t: t[0])
    x = np.array([[j for j in i[1]] for i in emb], dtype=np.float)
    x = torch.tensor(x, dtype=torch.float)  # torch.Size([2996, 64])
    # print(x)
    # exit()
    return x
