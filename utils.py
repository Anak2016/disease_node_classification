import numpy as np
import scipy.sparse as sp
import torch
import networkx as nx


class Cora():
    def __init__(self):
        pass

    def load_data(self, path="./data/cora/", dataset="cora"):
        """Load citation network dataset (cora only for now)"""
        print('Loading {} dataset...'.format(dataset))

        idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        labels = self.encode_onehot(idx_features_labels[:, -1])

        # build graph
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(
            edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        features = self.normalize_features(features)
        adj = self.normalize_adj(adj + sp.eye(adj.shape[0])) # add self loop

        idx_train = range(140)
        idx_val = range(200, 500)
        idx_test = range(500, 1500)

        adj = torch.FloatTensor(np.array(adj.todense()))
        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(np.where(labels)[1]) # label = int not one_hot

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        return adj, features, labels, idx_train, idx_val, idx_test

    def normalize_adj(self,mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv_sqrt = np.power(rowsum, -0.5).flatten() # D^0-5 * A * D^0.5
        r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
        r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
        return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

    def encode_onehot(self, labels):
        classes = set(labels)
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
        labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
        return labels_onehot

    def normalize_features(self,mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx


class Copd():
    def __init__(self):
        pass

    def load_data(self, path="./data/gene_disease/", dataset="copd_label"):
        """Load citation network dataset (cora only for now)"""
        print('Loading {} dataset...'.format(dataset))

        idx_labels = np.genfromtxt("{}{}_content.txt".format(path, dataset), dtype=np.dtype(str))
        labels = self.encode_onehot(idx_labels[:, -1] )

        # build graph
        edges_unordered = np.genfromtxt("{}{}_edges.txt".format(path, dataset), dtype=object)

        idx_map = {j:i for i, j in enumerate(np.unique(sorted(edges_unordered.flatten())))}
        idx_map_inverse = {i:j for i, j in enumerate(np.unique(sorted(edges_unordered.flatten())))}
        assert len(idx_map.keys()) == 2490, "uniq_gene + uniq_disease must equal 2490."


        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)

        uniq_node = set(edges_unordered.flatten())
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(len(uniq_node), len(uniq_node)), dtype=np.float32)
        assert adj.shape == (2490,2490), "adj matrix must be a square."
        assert np.unique(edges, axis=0).shape[0] == np.count_nonzero(adj.todense()), "edges are not correctly converted to coo_matrix."

        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        # print(np.count_nonzero(adj.todense())) # 7374

        # -- debugging
        # adj = torch.FloatTensor(np.array(adj.todense()))
        # diag = torch.diag(torch.FloatTensor([adj[i,i] for i in range(adj.size()[0])]))
        # diag = diag.numpy()
        # assert np.count_nonzero(diag) == 0, "self loop exist in adj"
        # -- end debugging

        # adj = self.normalize_adj(adj + sp.eye(adj.shape[0]))  # add self loop
        adj = self.normalize_adj(adj)

        # training, val, test  = 60, 20, 20 split
        dataset_size = edges.shape[0]
        idx_train = range(int(dataset_size * 0.6))
        idx_val = range(int(dataset_size * 0.6) , int(dataset_size * 0.8))
        idx_test = range(int(dataset_size * 0.8) , int(dataset_size))

        adj = torch.FloatTensor(np.array(adj.todense()))
        labels = torch.LongTensor(np.where(labels)[1])  # label = int not one_hot

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        return adj, labels, idx_train, idx_val, idx_test

    def normalize_adj(self,mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv_sqrt = np.power(rowsum, -0.5).flatten() # D^0-5 * A * D^0.5
        r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
        r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
        return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

    def encode_onehot(self, labels):
        classes = set(labels)
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
        labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
        return labels_onehot


def get_subgraph_disconnected( G):
    disconnected_graph = list(nx.connected_component_subgraphs(G))
    disconnected_graph = [(disconnected_graph[i], len(g)) for i, g in enumerate(disconnected_graph)]

    from operator import itemgetter
    disconnected_graph = sorted(disconnected_graph, key=itemgetter(1), reverse=True)
    # print(disconnected_graph)

    # disconnected_graph = [subgraph1, subgraph2, ....] #where subgraph is of type networkx
    disconnected_graph = [graph for graph, length in disconnected_graph]

    return disconnected_graph