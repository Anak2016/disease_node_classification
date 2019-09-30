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

def data_preprocessing(dataset = None, name='copd'):
    assert dataset is not None, "In run_preprocessing, dataset must not be none"

    if name=='copd':
        # run preprocessing for copd
        # if args.emb_path is not None:
        if args.emb_path is not None or args.emb_name != "no_feat":
            x   = add_features()
        else:
            # --------without features; instead use identity matrix of n*n where n is number of nodes
            x = np.identity(len(dataset.nodes2idx().keys()))
            x = torch.tensor(x, dtype=torch.float)
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


def create_common_nodes_as_features(dataset, geometric_dataset, plot_shared_gene_dist = False, used_nodes='all', edges_weight_option='jaccard'):
    '''
        gene is a feat of disease if there exist edges between gene and disease nodes
    :param dataset:
    :param edge_index:
    :param gene: common genes as feature (create edges between disease nodes)
    :param disease: common disease as feature (create edge between genes nodes)
    :return:
    '''
    assert used_nodes != "no", "--common_nodes_feat must be specifed option = ['all', 'gene', 'disease']"
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

    #TODO here>> create_edges_dict support option use=all, gene, disease??
    edges_dict, nodes_with_shared_genes = my_utils.create_edges_dict(edges, used_nodes) # return {disease: [{gene, weight}, ... ]} where list of genes are genese that connected to disease by an edge.
    # nodes_with_shared_genes = { key: [list(i.keys())[0]  for i in j ] for key, j in edges_dict.items()} # rearrange to {disease: [genes,...]}

    # plot_shared_gene_dist = True
    #--------plot_shread_nodes_distribution
    # plot_shared_gene_dist = True
    if plot_shared_gene_dist:
        #TODO here>> takes too long for gene and disease
        run_time = timer(plot_shared_nodes_distribution, nodes_with_shared_genes,  used_nodes) # plot and choose th for gene and disease)
        print(f"plot_shared_nodes_distribution takes {run_time} ms to run ")

    def get_added_edges(nodes_with_shared_genes, used_nodes):
        # the function is created for readability
        tmp = []
        # selected = []
        nodes_shared_count = {}
        for i, (d1, g1) in enumerate(nodes_with_shared_genes.items()):
            if len(list(nodes_with_shared_genes.items())[i+1:]) == 0:
                break
            else:
                for d2, g2 in list(nodes_with_shared_genes.items())[i+1:]:
                    nodes_shared_count.setdefault(  d1*d2, len(set(g1).intersection(set(g2)))) # key = g1 * g2 because it produce unique number for each pair
                    if used_nodes == 'gene':
                        if nodes_shared_count.get(d1*d2) > 0 : # this is slow
                            tmp.append((d1,d2))
                    elif used_nodes == 'disease':
                        if nodes_shared_count.get(g1*g2) > 0 : # this is slow
                            tmp.append((g1,g2))
                    else:
                        if nodes_shared_count.get(d1*d2) > 0 : # this is slow
                            tmp.append((d1,d2))
                        if nodes_shared_count.get(g1*g2) > 0 : # this is slow
                            tmp.append((g1,g2))

                    # if nodes_shared_count.get(d1*d2) > 0 and d1 != d2 and d1*d2 not in selected: # this is slow
                    #     tmp.append((d1,d2))
                        # selected.append(d1*d2)

                    # if len(set(g1).intersection(set(g2))) > 0 and d1 != d2 and (d1,d2) not in tmp and (d2,d1) not in tmp: # this is slow
                    #     tmp.append((d1,d2))
        return tmp


    added_edges = get_added_edges(nodes_with_shared_genes, used_nodes)

    max_node = len(list(dataset.nodes2idx()))
    before_added_edges = edges
    # original_adj = csr_matrix((1, (edges[0], edges[1])), shape=(max(max_node), max(max_node)))

    edges = edges + added_edges
    #--------get weight edges (networkx function should preserve order)
    edges_weight = None
    weighted_adj = None
    if edges_weight_option == 'jaccard':
        from edge_weight import jaccard_coeff
        weighted_adj, edges_weight = jaccard_coeff(np.array(edges).T)

    #TODO here>> trying out differnet weight for edges between diseases
    # > create gene to gene nodes and apply the same edge weight criteria
    # >
    weighted_adj[before_added_edges[0], before_added_edges[1]] = np.ones([len(before_added_edges[0]),len(before_added_edges[1])])
    # np.where(weighted_adj)
    weighted_adj
    edges_dict, _ = my_utils.create_edges_dict(edges, use_nodes=used_nodes)

    #--------convert to n*p format where n = number of nodes and p = nmber of features.
    # all_x_input = create_onehot(edges_dict, geometric_dataset, edges) #TODO here>> what do i expect as input and output of this??

    if used_nodes == 'gene':
        # dim = number of disease * num_all_nodes
        weighted_adj = weighted_adj[range(0,geometric_dataset.x.shape[0])]

    # networkx.convert_matrix.to_numpy_matrix
    #TODO here>> weighted_adj when all_x_input  = 101 vs all_x_input = all nodes
    return weighted_adj, edges_weight

    # return all_x_input, edges_weight
    # =====================
    # ==preprocessing
    # =====================
    # train_input, test_input = create_onehot(geometric_dataset,adj_list, edges)

    # return train_input, test_input
    # return  [adj, edges_index, G, g]

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
        Row-normalize sparse matrix
    :param: mx: csr_matrix
    :return mx: numpy array
    """

    rowsum = np.array(mx.sum(1))
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
    if args.emb_path is None:
        # emb_path = f"data/gene_disease/{args.time_stamp}/gene_disease/processed/embedding/" + emb_file

        # -- emb_file
        if args.emb_name == 'attentionwalk':
            emb_file = f"{args.emb_name}/{args.emb_name}_emb{args.time_stamp}.txt" # todo name is missing parameters that were used to generate emb

        elif args.emb_name == 'node2vec':
            if args.subgraph:
                emb_file = f"{args.emb_name}/{args.emb_name}_emb_subgraph{args.time_stamp}.txt" # todo name is missing parameters that were used to generate emb
                # emb_file = f"{args.emb_name}/{args.emb_name}_emb_subgraph_common_nodes_feat=True{args.time_stamp}.txt" # todo name is missing parameters that were used to generate emb
            else:
                emb_file = f"{args.emb_name}/{args.emb_name}_emb_fullgraph{args.time_stamp}.txt" # todo name is missing parameters that were used to generate emb
                # emb_file = f"{args.emb_name}/{args.emb_name}_emb_fullgraph_common_nodes_feat=True{args.time_stamp}.txt" # todo name is missing parameters that were used to generate emb

        elif args.emb_name == 'bine':
            emb_file = f"{args.emb_name}/bine{args.time_stamp}.txt" # todo name is missing parameters that were used to generate emb
            #
        elif args.emb_name == 'gcn':
            raise ValueError("Please provide path to gcn_emb file. "
                             "Provided emb_name == gcn is not YET supported! (I cannot come but with default name for gcn)")
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
        emb_np = pd.read_csv(args.emb_path, sep='\t', header=None).to_numpy()
        node_idx_col = np.arange(emb_np.shape[0])
        emb_dict = dict(zip(node_idx_col, emb_np))

        # order of node in emb_path preserved. (check gene2idx() for confirmation)

    # -- make sure that node embs are in ordered
    emb = sorted(emb_dict.items(), key=lambda t: t[0])
    x = np.array([[j for j in i[1]] for i in emb], dtype=np.float)
    x = torch.tensor(x, dtype=torch.float)  # torch.Size([2996, 64])

    return x
