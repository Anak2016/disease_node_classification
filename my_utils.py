import numpy as np
import scipy.sparse as sp
import torch
import networkx as nx
import pandas as pd
import collections
import matplotlib.pyplot as plt

from collections import OrderedDict

# -- utils
def pause():
    print("done")
    exit()

def display2screen(*args,**kwargs):
    if args is not None:
        for arg in args:
            print(arg)

    for k, v in kwargs.items():
        print(f"{k}: {v}")

    pause()

# dataset characteristic



# -- files manipulation
def create_copd_label_content(path='./data/gene_disease/',file_name= "copd_label", time_stamp='', **kwargs):
    '''
    use copd_label{time_stamp}.txt to write copd_label_content{time_stamp}.txt

    copd_label_content{time_stamp}.txt contains uniq pair of the following
        <cuis><class_label>

    :return:
    '''

    sep = '\t'
    if kwargs.get('sep'):
        sep = kwargs.get('sep')

    path2file = f"{path}{file_name}{time_stamp}.txt"
    df = pd.read_csv(path2file, sep=sep)
    df = df[["diseaseId", "class"]]
    # -- get unique disease
    np_ = np.unique(df.to_numpy().astype("<U22"), axis=0)
    df = pd.DataFrame(np_)
    # display2screen(df.shape)

    save_file = f"{file_name}_content{time_stamp}.txt"
    write2files(df,path=path,file_name=save_file,type='df')

def create_copd_label_edges(path='./data/gene_disease/',file_name= "copd_label", time_stamp='', **kwargs):
    '''
    use copd_label{time_stamp}.txt to write copd_label_edges{time_stamp}.txt

    copd_label_edges{time_stamp}.txt contains uniq pair the following
        <cuis><class_label>

    :return:
    '''
    sep = '\t'
    if kwargs.get('sep'):
        sep = kwargs.get('sep')

    path2file = f"{path}{file_name}{time_stamp}.txt"
    df = pd.read_csv(path2file, sep=sep)
    df = df[["geneId", "diseaseId"]]
    # -- get unique edges
    np_ = np.unique(df.to_numpy().astype("<U22"), axis=0)
    df = pd.DataFrame(np_)
    # display2screen(df.shape)

    save_file = f"{file_name}_edges{time_stamp}.txt"
    write2files(df,path=path,file_name=save_file,type='df')


def write2files(data,path="./data/gene_disease/", file_name=None, type='df'):
    '''

    :parame data: content to be written in files
    :param path:
    :param dataset:
    :param type: type of content arg;  df, np, dict.
    :return:
    '''
    if file_name is None:
        raise ValueError('In write2files, dataset is not given as an argument')
    if isinstance(data, pd.DataFrame):
        data.to_csv(path+file_name, sep='\t', index=False, header=None)
    elif isinstance(data, np.ndarray):
        pd.DataFrame(data, dtype="U") # convert to type string
        data.to_csv(path + file_name, sep='\t', index=False, header=None)
    elif isinstance(data,dict):
        pd.DataFrame.from_dict(data, dtype="U", columns=None, orient='columns')  # convert to type string
        data.to_csv(path + file_name, sep='\t', index=False, header=None)
    else:
        raise ValueError('type of given data are not accpeted by write2files function')

# -- network x related function
def get_subgraph_disconnected( G):

    disconnected_graph = list(nx.connected_component_subgraphs(G))
    disconnected_graph = [(disconnected_graph[i], len(g)) for i, g in enumerate(disconnected_graph)]

    from operator import itemgetter
    disconnected_graph = sorted(disconnected_graph, key=itemgetter(1), reverse=True)
    # print(disconnected_graph)

    # disconnected_graph = [subgraph1, subgraph2, ....] #where subgraph is of type networkx
    disconnected_graph = [graph for graph, length in disconnected_graph]

    return disconnected_graph


class Cora():
    def __init__(self):
        pass

    def load_data(self, path="./data/cora/", dataset="cora"):
        """Load citation network dataset (cora only for now)"""
        print('Loading {} dataset...'.format(dataset))

        idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype='U')
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        labels = self.encode_onehot(idx_features_labels[:, -1])

        # build graph
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype='U')
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype='U').reshape(
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
    def __init__(self, path=None, data=None, time_stamp=None):

        self.time_stamp = time_stamp
        # --numpy data
        if path is not None and data is not None:
            self.disease, self.labels = GetData.disease_labels(path=path, time_stamp=time_stamp)
        else:
            self.disease, self.labels = GetData.disease_labels(path=path, time_stamp=time_stamp)

        self.gene, self.non_uniq_diseases = GetData.gene_disease(path=path, time_stamp=time_stamp)
        self.edges = GetData.edges(path=path, time_stamp=time_stamp)

        # display2screen(len(self.disease2idx()),len(self.genes2idx().keys()), 'line 186')

    def load_data(self, path="./data/gene_disease/", dataset="copd_label", time_stamp=''):
        """
        load data of cpod for full grpah and largest connected component
        return: adj = adj of subgraph
                labels = lables of subgraph # one diseaseid is not connected to the largest componenet
                g = subgraph of type networkx.Graph()
        """
        print('Loading {} dataset...'.format(dataset))

        edges = self.edges
        # only use largest connected component

        # edges_unordered = pd.DataFrame(edges_unordered.T, columns=['geneid', 'diseaseid'])
        # edges_unordered = np.array(list(map(self.nodes2idx().get, edges_unordered.to_numpy().flatten())), dtype=np.int32).reshape(-1,2).T
        edges = np.array(list(map(self.nodes2idx().get, edges.T.flatten())), dtype=np.int32).reshape(-1,2).T

        edges = pd.DataFrame(edges.T, columns=['geneid', 'diseaseid'])

        G = nx.from_pandas_edgelist(edges, 'geneid', 'diseaseid')
        g = get_subgraph_disconnected(G)[0]

        edges = [np.array((x, y), dtype=np.int32) for x, y in g.edges]
        edges = np.array(edges, dtype=np.int32).T  # shape = (2, 3678)

        # {32, 39, 53, 55, 59, 60} does not exist in disease2class().keys()
        # There are total of 9 keys from 0-8 but only 7 are in the largest subgraph,
        # None = label of geneid ; basically implies that geneid has no labels\

        # label = 8 represent no class; it is used to label geneid
        # todo ???how should I label nodes that has no label? as None
        labels = set(map(self.disease2class().get, g.nodes))# {0, 1, 2, 3, 4, 5, 6, 7, None}

        # labels = {i for i in labels if i is not None}
        labels = self.encode_onehot(labels)

        # len(g.nodes()) = 2489
        adj = sp.coo_matrix((np.ones(edges.T.shape[0]), (edges.T[:, 0], edges.T[:, 1])),
                            shape=(len(G.nodes()), len(G.nodes())), dtype=np.float32) # 7374

        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = self.normalize_adj(adj)

        adj = torch.FloatTensor(np.array(adj.todense()))
        labels = torch.LongTensor(np.where(labels)[1])  # label is of type int NOT type one_hot
        return adj, labels, G, g

    # -- dataset characteristic
    def class_member_dist(self, plot=False, verbose=True):
        '''
        plot class member distribution

        :return: rank dict by value in decending order
            [(class1, # members), (class2, # member), ....]
        '''
        # use orderdict
        dist = {l:len(v) for l, v in self.class2disease().items()}

        # display2screen(dist)
        # sorted by value in ascending order
        dist = sorted(dist.items(), key= lambda t:t[1], reverse=False)
        if plot:
            plt.bar([i for i in range(5)],height=[c[1] for c in dist])
            plt.xticks([i for i in range(5)], [c[0] for c in dist])
            plt.title("class_member_dist")
            plt.show()

        if verbose:
            print(f'ranking class_member_dist in decending order')
            for i, tuple in enumerate(dist):
                print(f"rank {i}: class = {tuple[0]} has {tuple[1]} number of members")

        return dist

    def edges2nodes_ratio(self, verbose=False):
        ratio = self.edges.shape[1]/len(self.nodes2idx().keys())
        if verbose:
            print(f'edges2nodes_ratio = {ratio}')
        return ratio

    def label_rate(self, verbose=False):
        rate = self.disease.shape[0]/len(self.nodes2idx().keys())
        if verbose:
            print(f'label_rate = {rate}')
        return rate

    # todo here>.
    def rank_gene_overlap(self, verbose=False, plot=False):
        '''
        step tp step on how this func works
        1. for each class, create a set of gene that have edges connected to that class
        2. find the overlappi of gene between each classes and rank all of them
            : there will be n*n comparison where n = number of class
        :return:
        '''
        nodes = sorted([i for i in self.nodes2idx().values()])
        edges_flat = self.edges.T.flatten()
        edges_flat = [self.nodes2idx()[i] for i in edges_flat]
        '''
        edges_flat has the following format
            [d1,g1,d2,g2,d3,g3,...] where d = disease and g = gene
        '''
        # idx2disease = {d:i for i, d in self.nodes2idx().items()}
        class_gene = {l:[] for l in self.disease2class().values()}

        for i, n in enumerate(edges_flat[1::2]):
            if n not in self.disease2class().keys(): # gene
                gene_idx = n # int_rep
                if i ==0:
                    disease_idx = edges_flat[0]  # int_rep
                else:
                    disease_idx = edges_flat[(i * 2) - 1]  # int_rep

                #-- add gene to its corresponding class
                class_gene[self.disease2class()[disease_idx]].append(gene_idx)
            else:
                disease_idx = n # int_rep
                if i == 0:
                    gene_idx = edges_flat[0] # int_rep
                else:
                    gene_idx = edges_flat[(i*2) -1] # int_rep

                # -- add gene to its corresponding class
                class_gene[self.disease2class()[disease_idx]].append(gene_idx)

        intersect_gene_member = {}
        # -- find gene overlap between classes
        count = 0
        for k_out, v_out in class_gene.items():
            for k_in, v_in in class_gene.items():
                intersect_gene_member[f"{k_out} & {k_in} "] = len(set(v_out) & set(v_in))
                count +=1
                # print(f"gene intersection between {k_out} and {k_in} = {len(set(v_out) & set(v_in))}")

        assert count == len(class_gene.keys()) * len(class_gene.keys()), "loop is implemented incorreclty"

        intersect_gene_member = sorted(intersect_gene_member.items(), key=lambda t:t[1], reverse=True)

        for k in intersect_gene_member:
            print(f"gene intersection between {k[0]}= {k[1]}")

        if plot:
            plt.bar([i for i in range(len(intersect_gene_member))],height=[c[1] for c in intersect_gene_member])
            plt.xticks([i for i in range(len(intersect_gene_member))], [c[0] for c in intersect_gene_member], rotation='vertical')
            plt.title("ranking gene member overlap between classes")
            plt.show()

        return class_gene, intersect_gene_member

    # -- preprocessing
    def normalize_adj(self,mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))

        # store 0 array of the same size then only change value where rowsum!=0
        r_inv_sqrt =  np.power(rowsum, -0.5, out=np.zeros_like(rowsum), where=rowsum!=0).flatten() # D^0-5 * A * D^0.5
        r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
        r_mat_inv_sqrt = sp.diags(r_inv_sqrt)

        return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

    def encode_onehot(self, labels):
        classes = set(labels)
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
        labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
        return labels_onehot

    # -- conversion related function
    def nodes2idx(self):
        '''
            geneid and diseaseid are nodes in the graph
        :return: {geneid or diseaseid : id }
        '''

        return { **self.disease2idx(), **self.genes2idx()} # concat 2 dictionary

    def disease2class(self):
        '''

        :return:
            {cuis: labels}
        '''

        cuis2labels = {self.disease2idx()[c]: self.labels2idx()[l] for c, l in zip(self.disease, self.labels)}

        return cuis2labels

    def class2disease(self):
        '''

        :return:
            {class_label: [lsit of disease member of the class]}
        '''
        class2disease = {k: [] for k in self.labels2idx().keys()}
        for k,c in self.disease2class().items():
             class2disease[c].append(k)

        return class2disease


    def labels2idx(self):
        '''
        :return: {label: label_rep}
        '''
        labels = self.labels
        return {l: i for i, l in enumerate(list(collections.OrderedDict.fromkeys(labels.tolist()).keys()))}

    def disease2idx(self):
        '''
        :return: {diseaseid: diseaseid_rep}
        '''
        cuis = self.disease

        return {l: i for i, l in enumerate(list(collections.OrderedDict.fromkeys(cuis.tolist()).keys()))}

    def genes2idx(self):
        '''

        :return: {genesid: diseaseid_rep}
        '''
        gene = self.gene

        # 61 is the amount of diseaseid
        return {l:i+int(self.disease.shape[0]) for i, l in enumerate(list(collections.OrderedDict.fromkeys(gene.tolist()).keys()))}

    # -- read, write, convert files
    def create_rep_dataset(self, path='data/gene_disease/'):
        '''
        create files with value representtation of the following files
            >copd_label_content{time_stamp}.txt and copd_label_edges{time_stamp}.txt

        '''
        # -- copd_label_edges.txt
        # disease is not the same as self.disease in that it is not uniq
        # genes, non_uniq_diseases = GetData.gene_disease()
        genes, non_uniq_diseases = self.gene, self.non_uniq_diseases

        # convert gene and non_uniq_disease to  its int_rep
        genes = list(map(self.genes2idx().get, genes))
        non_uniq_diseases = list(map(self.disease2idx().get, non_uniq_diseases))

        # --copd_label_content.txt
        # uniq_diseases, labels = GetData.disease_labels()
        uniq_diseases, labels = self.disease, self.labels

        uniq_diseases = list(map(self.disease2idx().get, uniq_diseases))
        labels = list(map(self.labels2idx().get, labels ))

        gene_disease = pd.DataFrame([genes,non_uniq_diseases], dtype=np.int32).T
        disease_label = pd.DataFrame([uniq_diseases,labels], dtype=np.int32).T
        # display2screen(gene_disease.head(), disease_label.head())

        # write to rep_copd_label_edges.txt
        write2files(gene_disease,path='data/gene_disease/rep/', file_name=f"rep_copd_label_edges{self.time_stamp}.txt")

        # write to rep_copd_content.txt
        write2files(disease_label,path='data/gene_disease/rep/', file_name=f"rep_copd_content{self.time_stamp}.txt")


class GetData():
    def __init__(self):
        pass

    @staticmethod
    def edges(path='data/gene_disease/', data='copd_label_edges', time_stamp=''):
    # def edges(path='data/gene_disease/', data='copd_label_edges.txt'):
        '''

            :param path:
            :param data:
            :return: edges_index; type = numpy(), shape = [2, number of edges ]
            '''
        path2file = f'{path}{data}{time_stamp}.txt'

        edges = pd.read_csv(path2file, sep='\t', names=['geneid', 'diseaseid'], header=None)
        # largest connected componenet

        # -- create graph which have ordered nodes and edges.
        G = nx.OrderedGraph()
        edges_ordered = [(i,j) for i, j in edges.to_numpy()]
        bi0 = { i: None for i in edges.to_numpy()[:, 0]}
        bi1 = { i: None for i in edges.to_numpy()[:, 1]}

        G.add_nodes_from(list(bi0.keys()), bipartite=0)
        G.add_nodes_from(list(bi1.keys()), bipartite=1 )
        G.add_edges_from(edges_ordered)

        edges = [np.array((x, y), dtype='U') for x, y in G.edges]
        edges = np.array(edges, dtype='U').T  # shape = (2, 3687)
        return edges

    @staticmethod
    # def disease_labels(path='data/gene_disease/', data='copd_label_content.txt'):
    def disease_labels(path='data/gene_disease/', data='copd_label_content', time_stamp=''):
        '''

            :param path:
            :param data:
            :return: [cui1, cui2, ... ]; type = numpy ; diseases are uniq
                    [label1, label2,..]; type = numpy ; label of the disease in order
            '''
        path2file = f'{path}{data}{time_stamp}.txt'
        with open(path2file, 'r') as f:
            disease_labels = pd.read_csv(path2file, sep='\t', names=['diseaseid', 'label'], header=None).to_numpy().flatten()
            uniq_disease = np.array(disease_labels.tolist()[0::2])
            labels = np.array(disease_labels.tolist()[1::2])
        return uniq_disease, labels

    @staticmethod
    def gene_disease(path='data/gene_disease/', data='copd_label_edges', time_stamp=''):
        '''

        :return: [gene1, gene2,...]; type = numpy; geneid are not uniq
                [cui1, cui2,....]; type= numpy; diseaseid are not uniq
        '''
        path2file = f'{path}{data}{time_stamp}.txt'

        with open(path2file, 'r') as f:
            gene_disease = pd.read_csv(path2file, sep='\t', names=['geneid', 'diseaseid'], header=None).to_numpy().flatten()
            gene = np.array(gene_disease.tolist()[0::2], dtype="U")
            non_uniq_disease = np.array(gene_disease.tolist()[1::2], dtype="U") # diseaseId that has edges.

        return gene, non_uniq_disease

class Conversion():
    def __init__(self, path=None, data=None):
        pass




