import sys,os
USER = os.environ['USERPROFILE']
sys.path.insert(1,f'{USER}\\PycharmProjects\\my_utility')

from utility_code.my_utility import *
from utility_code.python_lib_essential import *

from anak import run_GNN
from arg_parser import *

import os.path
import preprocessing
import all_datasets


def create_features_graph(adj, features, labels):
    '''
        creating a features_graph by add feature_node from node_feature to the graph in which node belongs to .

    :param adj: coo_matrix with dimension n*n where n = number of total nodes (without features node)
    :param features: coo_matrix with dimension n*d where d = feature dim
    :param labels: numpy array
    :return: adj of features_node_graph
    '''
    '''
    step
        >assign index to features with value of 1 (coo_matrix) 
        >add coordinate to adj (coo_matrix)
    '''
    # -- convert node_feature
    adj_dim = adj.shape[0] # use this as a start index of features_row
    features_dim = features.shape[1]
    # display2screen(features_dim)

    # # -- feature to feature edges
    # # --option 1 feature nodes have no self loops
    # option = "feature_no_self_loop"
    # ff_row = [i + adj_dim for i in range(features_dim)]
    # ff_col  = [i + adj_dim for i in range(features_dim)]
    # ff_val = [0 for i in range(features_dim)]

    # # -- option 2 features nodes have self loop
    option = 'features_with_self_loop'
    identity_mx = sp.eye(adj.shape[0]).tocoo()
    ff_row, ff_col, ff_val = identity_mx.row.tolist(), identity_mx.col.tolist(), identity_mx.data.tolist()

    # # -- option 3 all features are connected as clique
    # option = 'features_clique'
    # ones_max = np.ones(adj.shape[0])
    # ones_max = sp.coo_matrix(ones_max)
    # ff_row, ff_col, ff_val = ones_max.row.tolist(), ones_max.col.tolist(), ones_max.data.tolist()


    # ff_row, ff_col, ff_val = identity_mx.row, identity_mx.col, identity_mx.data

    # node to feature edges
    features_ind = [i + adj_dim for i in range(features_dim)] # dim = feature_dim
    nodes_ind = [i for i in range(adj_dim)] # dim = adj_dim

    nf_row = []
    nf_col = []
    nf_val = []

    # todo create unormalized_fetaures_graph.npy
    # file_name = f"adj_features.npy"

    file_name = f"adj_features_graph_normalized_node_option={option}.npy"
    # file_name = "adj_unormalized_features_graph.npy"
    file_path = f'data/preprocessing/{file_name}'
    if os.path.exists(file_path):
        # -- load pre_processed numpy from file_path
        s = time.time()
        adj = np.load(file_path)
        f = time.time()
        total = f-s
        print(f"total time = {total}")
        node_features_row, node_features_col = sp.find(sp.csr_matrix(adj))[0], sp.find(sp.csr_matrix(adj))[1]
    else:

        print("converted node features to features graph...")
        s = time.time()
        for row in nodes_ind:

            '''
            csr_matrix.nonzero()
                eg (array([0, 0, 1, 2, 2]), array([0, 1, 2, 0, 2]))
            '''
            # -- non_zero_feature of the current cow
            non_zero_ind = [ features.nonzero()[1][i] + adj_dim for i,j in enumerate(features.nonzero()[0]) if j == row]
            nf_row += [row for i in range(len(non_zero_ind))]
            nf_col += non_zero_ind

            # -- val = 1 if node has feature in it otherwise 0
            nf_val += [1  for i in features_ind if i in non_zero_ind] # dim = adj_dim
        f = time.time()
        total = f-s
        print(f"time ={total}")

        # --normalized nn_val
        adj = preprocessing.normalize_features(adj)

        nn_row = sp.find(adj)[0].tolist()
        nn_col = sp.find(adj)[1].tolist()
        nn_val = sp.find(adj)[2].tolist()

        # --create adj of node_featurse_graph
        node_features_row = ff_row + nf_row + nf_col + nn_row
        node_features_col = ff_col + nf_col + nf_row + nn_col
        node_features_val = ff_val + nf_val + nf_val + nn_val

        assert len(node_features_val) == len(node_features_row) == len(node_features_col), f"{len(node_features_val)} == {len(node_features_row)} == {len(node_features_col)}"
        node_features_graph = sp.csr_matrix((node_features_val, (node_features_row, node_features_col)))

        # -- edges
        # edges = np.array([[i, j] for i, j in zip(node_features_row, node_features_col)]).T  # dim (2, #edges)
        # -- for readability
        adj = node_features_graph.todense()

        # -- save to file_path
        np.save(file_path, adj)

    # display2screen('line 117')
    max_labels = np.amax(labels)
    non_class_label = max_labels+1

    labels = labels.tolist() + [non_class_label for i in range(adj.shape[0] - labels.flatten().shape[0])]

    # -- convert to numpy array
    edges = np.array([[i, j] for i, j in zip(node_features_row, node_features_col)]).T  # dim (2, #edges)
    edges = edges.astype(int)

    labels = np.array(labels)

    # display2screen(edges.shape)
    return adj, edges, labels

class cora_feature_graph():
    def __init__(self, adj, edge_index, labels, cora_adj):
        '''
            all of the params must be in numpy array
        :param adj: n*n where n = number of nodes in the cora features graph dataset
        :param edge_index: (2, # of edges)
        :param labels:  n where n = number of nodes in the cora features fgraph dataset
        :param cora_adj: m*m where m = number of nodes in cora dataset
        '''
        # -- self initalization
        self.adj = adj
        self.edge_index = edge_index
        self.y = labels
        self.cora_adj = cora_adj

        # -- members
        self.labels   = list(self.labels2class().keys())
        self.unlabels   = list(self.labels2class().keys())
        self.nodes = list(self.nodes2class().keys())
        # -- num len
        self.num_labels = len(self.labels2class().keys())
        self.num_unlabels = len(self.unlabels2class().keys())
        self.num_nodes = len(self.nodes2class().keys())

        # -- num class
        self.num_class = len(self.class2labels().keys())

        # -- this section must not be deleted for compatibility reason with copd_geometric_dataset
    def class2labels(self):
        '''
        :return:
            {class: [list of labels that belongs to this class]}
        '''
        class2labels = {k: [] for k in set(self.y.tolist()[:cora_adj.shape[0]])}
        for k, c in self.labels2class().items():
            class2labels[c].append(k)

        return class2labels

    def labels2class(self):
        '''
        return:
            {cuis: labels}
        '''
        labeled_node = [i for i in range(self.cora_adj.shape[0])]

        return {i:j for i,j in zip(labeled_node, self.y[:len(labeled_node)])}

    def labelnodes2idx(self):
        '''
            in Copd class labels2idx is already taken labels2idx can be replaced with class2idx
        :return: {label: label_rep}
        '''
        return {i:i for i in range(self.cora_adj.shape[0])}

    # -- dictionary
    def nodes2class(self):
        return {**self.labels2class(), **self.unlabels2class()}

    def unlabels2class(self):
        unlabeled_node = [i for i in range(self.adj.shape[0]) if i >= self.cora_adj.shape[0]]

        return {i:j for i,j in zip(unlabeled_node,self.y[:len(unlabeled_node)]) }

if __name__ == "__main__":
    cora = all_datasets.Cora()
    adj, features, labels, idx_train, idx_val, idx_test, unnormalize_adj = cora.load_data()
    # display2screen(labels.shape) #2708

    # -- for readability
    cora_adj = adj
    adj,edge_index, y = create_features_graph(unnormalize_adj, sp.csr_matrix(features.numpy()), labels.numpy())

    cora_feature_graph = cora_feature_graph(adj, edge_index, y, cora_adj)

    # ==================
    # == features options rank by accuracy performance in decending order
    # ==================
    # # -- option 1 => dim = (n+f) * (n+f) ; unormalized adj of node_features
    # #   >> interestin observation:
    # #       :accuracy converge very slowly at 550 to 71 percent accuracy at with option 1.1 normalized adj
    # #       :accuracy converge very quickly at 61-61 percent with option 1.2 unnormalized adj
    # # tmp = adj
    # adj = normalize_features(sp.csr_matrix(adj.astype(float))) # --option 1.1 normalize
    # # adj = adj  # option 1.2 unnormalize
    # # display2screen(tmp[np.nonzero(x)[0][0], np.nonzero(x)[1][0]], x[np.nonzero(x)[0][0], np.nonzero(x)[1][0]])
    # adj = torch.tensor(adj)
    # x = adj
    # edge_index = torch.tensor(edge_index, dtype=torch.int32)
    # y = torch.tensor(y, dtype=torch.long)

    # -- option 2 => dim = (n+f) * (n+f) ;identity matrix
    #   >> test accuracy converge at aroung 61-62 percent
    x = np.identity(adj.shape[0])
    x = preprocessing.normalize_features(sp.csr_matrix(x))
    x = torch.tensor(x, dtype=torch.long)
    edge_index = torch.tensor(edge_index, dtype=torch.int32)
    y = torch.tensor(y, dtype=torch.long)


    # -- option 3 => dim = (n+f) * f;
    #       >>very bad accuracy at epoch 200, but it seems that its performance has not yet fully converge
    # features = np.concatenate((features.numpy(), np.identity(features.shape[1]))) # -- option 3.1
    # # features = np.concatenate((features.numpy(),  0 * np.identity(features.shape[1]))).astype(float) # -- option 3.2
    # # display2screen(np.amax(np.sum(features, axis=1)))
    # # tmp = features
    # features = normalize_features(sp.csr_matrix(features))
    # # display2screen(tmp[np.nonzero(x)[0][0], np.nonzero(x)[1][0]], x[np.nonzero(x)[0][0], np.nonzero(x)[1][0]])
    # features = torch.tensor(features, dtype=torch.long)
    # x = features
    # edge_index = torch.tensor(edge_index, dtype=torch.int32)
    # y = torch.tensor(y, dtype=torch.long)
    # display2screen(np.amax(edge_index.numpy().flatten(), axis=0),np.amin(edge_index.numpy().flatten(), axis=0),x.shape) # 4140

    # # -- option 4: dim = (n+f) * emb_dim; node2vec embeding
    # from node2vec import Node2Vec
    # import networkx as nx
    #
    # # G = nx.Graph()
    # # G.add_nodes_from([i for i in range(4141)])
    # # G.add_edges_from(edge_index.transpose().tolist())
    # # display2screen(len(G.nodes), max(G.nodes), min(G.nodes))
    # # display2screen(3152 in edge_index.flatten())
    #
    # folder = f'data/gene_disease/{args.time_stamp}/processed/embedding/node2vec/'
    # file_name = 'new_feature_node_graph_node2vec.txt' # 4141 nodes
    # # file_name = 'feature_node_graph_node2vec.txt' # 4140 nodes
    # if not os.path.exists(f'{folder}{file_name}'):
    #     G = nx.Graph()
    #     G.add_nodes_from([i for i in range(4141)])
    #     G.add_edges_from(edge_index.transpose().tolist())
    #     print(f'writing to {folder}{file_name}')
    #     save_node2vec_emb(G, save_path=folder, EMBEDDING_FILENAME=file_name, log=True)
    #
    # file_path = f'data/preprocessing/tmp_{file_name}'
    #
    # emb_path = f'{folder}{file_name}'
    # with open(emb_path, 'r') as f:
    #     tmp = f.readlines()
    #     tmp = tmp[1:]
    #     split = ' '
    #     emb_dict = {int(float(i.split(split)[0])): list(map(float, i.split(split)[1:])) for i in tmp}
    #     emb = sorted(emb_dict.items(), key=lambda t: t[0])
    #     x = np.array([[j for j in i[1]] for i in emb], dtype=np.float)
    #
    # x = torch.tensor(x, dtype=torch.float)  # torch.Size([2996, 64])
    # edge_index = torch.tensor(edge_index, dtype=torch.int32)
    # y = torch.tensor(y, dtype=torch.long)
    # # display2screen(x.shape) # expect 4141 not 4140

    # ==============================
    # == features as graph structure rank accuracy performance in decending order
    # ==============================
    cora_geometric_dataset = all_datasets.GeometricDataset(cora_feature_graph, x=x, edges_index=edge_index, y=y, split=args.split)
    # cora_geometric_dataset = GeometricDataset(cora_feature_graph, x=adj, edges_index=edge_index, y=y, split=args.split)
    # cora_geometric_dataset = GeometricDataset(cora_feature_graph, x=features, edges_index=edge_index, y=y, split=args.split)


    # ===============================
    # == run graph
    # ===============================
    # display2screen(adj.shape, edge_index.shape)
    run_GNN(data=cora_geometric_dataset, emb_name=args.emb_name, time_stamp=args.time_stamp, tuning=args.tuning,
            log=args.log, verbose=args.verbose, lr=args.lr, weight_decay=args.weight_decay)
