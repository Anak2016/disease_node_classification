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
        edge_index = list(map(dataset.nodes2idx().get, dataset.edges.T.flatten()))
        edge_index = torch.tensor(edge_index, dtype=torch.int64).view(2, -1)  # torch.Size([2, 4715])

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

def create_common_genes_as_features(dataset, geometric_dataset, plot_shared_gene_dist = False):
    '''
        gene is a feat of disease if there exist edges between gene and disease nodes
    :param dataset:
    :param edge_index:
    :return:
    '''

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
    edges_dict = my_utils.create_edges_dict(edges)
    nodes_with_shared_genes = { key: [list(i.keys())[0]  for i in j ] for key, j in edges_dict.items()}

    tmp = []
    # plot_shared_gene_dist = True

    if plot_shared_gene_dist:
        shared_gene_dist = []
        for th in range(0,101):
            tmp=[]
            for d1, g1 in nodes_with_shared_genes.items():
                for d2, g2 in nodes_with_shared_genes.items():
                    if len(set(g1).intersection(set(g2))) > th and d1 != d2 and (d1,d2) not in tmp and (d2,d1) not in tmp:
                        tmp.append([d1,d2])

            shared_gene_dist.append({ th : len(tmp)})
            # nodes_with_shared_genes = tmp

        config = {
            'nodes_with_shared_genes_dist': {
                'x_label': 'number of shared genes',
                'y_label': 'number of nodes',
                'legend': [{"kwargs": {"loc": "lower right"}}],
                'plot': [
                        {"args": [ [list(i.keys())[0] for i in shared_gene_dist],[list(i.values())[0] for i in shared_gene_dist]] },
                        # {"args": [ list(range(0, len(shared_gene_dist))), [0 for i in range(0,len(shared_gene_dist))]]},
                         ]
            }
        }
        plot_figures(config)

    for d1, g1 in nodes_with_shared_genes.items():
        for d2, g2 in nodes_with_shared_genes.items():
            if len(set(g1).intersection(set(g2))) > 0 and d1 != d2 and (d1,d2) not in tmp and (d2,d1) not in tmp:
                tmp.append((d1,d2))

    nodes_with_shared_genes = tmp
    edges = edges + nodes_with_shared_genes
    edges_dict = my_utils.create_edges_dict(edges)

    all_x_input = create_onehot(edges_dict, geometric_dataset, edges)

    return all_x_input
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
            else:
                emb_file = f"{args.emb_name}/{args.emb_name}_emb_fullgraph{args.time_stamp}.txt" # todo name is missing parameters that were used to generate emb

        elif args.emb_name == 'bine':
            emb_file = f"{args.emb_name}/bine{args.time_stamp}.txt" # todo name is missing parameters that were used to generate emb

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
