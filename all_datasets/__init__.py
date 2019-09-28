import sys,os
USER = os.environ['USERPROFILE']
sys.path.insert(1,f'{USER}\\PycharmProjects\\my_utility')

from utility_code.my_utility import *
from utility_code.python_lib_essential import *

from arg_parser import *
from torch_geometric.data import Data
import my_utils

class GeometricDataset(Data):

    def __init__(self, data, x=None, edges_index=None, edge_attr=None, y=None, split=0.8, undirected=True):
        '''
            all parameters must have type of torch.tensor expect split and data
        :param data: copd
        :param x: n*d where d = features if no feature d = n
        :param edges_index: (2, # of edges)
        :param edge_attr: (# of edges, # of edge attr) # i am not sure myself
        :param y:  n where n = $ of nodes
        :param split:
        '''
        if undirected:
            df = pd.DataFrame(edges_index.numpy().T, columns=['source','target'])
            edges_index = torch.tensor(list(nx.from_pandas_edgelist(df, "source", "target", create_using=nx.Graph()).edges)).transpose(0,1) # edges are undirected

        super(GeometricDataset, self).__init__(x, edges_index, edge_attr, y)
        self.dataset = data
        self.subgraph = self.dataset.subgraph
        self.graph = self.dataset.graph

        # -- intialization of variable from self.dataset
        self.labeled_nodes  = self.dataset.labelnodes2idx

        self.split = split
        self.y = y
        # -- masking
        self.train_mask_set = []
        self.test_mask_set = []

        ind = 0  # inc everytime nodes are added; check how many nodes are included in training set
        count = 0
        arr_ind = 0 # inc everytimes for loop go through all of the classes; it represent current ind of val_list in each class.

        seed_file = f'data/gene_disease/seed/split={args.split}_seed={args.seed}.txt'

        if os.path.exists(seed_file):
            tmp = seed_file
            with open(tmp, 'r') as f:
                print(f"reading from {tmp}...")
                txt = f.readlines()
                self.train_mask_set = list(itertools.chain(list(map(int, i.split(' '))) for i in txt if len(i) > 0))[0]
        else:
            while True:
                max_class_int_rep = self.num_classes - 1 # max int_rep of all classes
                current_class = count % max_class_int_rep

                #--------create a random split of training and testing dataset
                if ind < int(split * len(self.dataset.labels2class().keys())):  # training set
                    next_val = set(self.dataset.class2labels()[current_class]).difference(set(self.train_mask_set))
                    if len(next_val) > 1:
                        random.seed(args.seed)
                        j = random.randint(1, len(next_val)-1)
                        next_val = list(next_val)[j]
                        # self.train_mask_set.append(copd.class2disease()[current_class][arr_ind])
                        self.train_mask_set.append(next_val)
                        ind += 1

                if ind == int(split * len(self.dataset.labels2class().keys())):
                    break
                # -- debugging
                # if count % 10 == 0:
                #     print(count)

                count += 1
            #TODO here>> figure out why it takes too long
            #=====================
            #==save seed to file
            #=====================
            tmp = f'data/gene_disease/seed/split={args.split}_seed={args.seed}.txt'
            os.makedirs(os.path.dirname(tmp), exist_ok=True)
            if not os.path.exists(tmp):
                with open(tmp, 'w'): pass
            with open(tmp,'w') as f:
                print(f"writing to {tmp}...")
                f.write(' '.join(map(str,self.train_mask)))

        # display2screen(ind, count)
        self.test_mask_set = list(set([i for i in self.dataset.labelnodes2idx().values()]).difference(self.train_mask_set))
        # display2screen(max(self.test_mask_set), max(self.train_mask_set))

        train_class = set([self.dataset.labels2class()[i] for i in self.train_mask_set])
        test_class  = set([self.dataset.labels2class()[i] for i in self.test_mask_set])

        # display2screen(train_class, test_class, train_class.symmetric_difference(test_class))
        assert len(self.test_mask_set) + len(self.train_mask_set) == len(self.dataset.labels2class().keys()), "Some diseases are not included in neither training or test set "
        assert len(set(self.train_mask_set).intersection(set(self.test_mask_set))) == 0, "diseases in both classes must be unique to its dataset either trianing or test dataset"
        assert len(set([self.dataset.labels2class()[i] for i in self.train_mask_set])) == len(self.dataset.class2labels().keys()), f"members of training set does not include all of the class labels.\n classes={train_class}"
        assert len(set([self.dataset.labels2class()[i] for i in self.test_mask_set])) == len(self.dataset.class2labels().keys()), f"members of test set does not include all of the class labels.\n classes={test_class}"

        # display2screen(self.test_mask_set, self.train_mask_set)
        # display2screen(len(self.test_mask_set), len(self.train_mask_set))


        # -- convert to torch.tensor

        self.train_mask_set = torch.LongTensor(self.train_mask_set)
        self.test_mask_set = torch.LongTensor(self.test_mask_set)

        # # -- add gene to train and test dataset; NOPE model always predict gene
        # gene = list(copd.genes2idx().values())
        # import random
        # random.shuffle(gene)
        #
        # self.train_mask_set = torch.LongTensor(self.train_mask_set + gene[:int(0.8 * len(gene))])
        # self.test_mask_set = torch.LongTensor(self.test_mask_set + gene[int(0.8 * len(gene)):])

    @property
    def num_classes(self):
        # return np.unique(y.numpy()).shape[0]
        return np.unique(self.y.numpy()).shape[0]

    # -- masking index for x and y
    @property
    def train_mask(self):
        # make sure that all train set ahve all the classes
        return self.train_mask_set

    @property
    def test_mask(self):
        # make sure that all test set ahve all the classes
        return self.test_mask_set

    @train_mask.setter
    def train_mask(self, value):
        # print('setting value to ')
        self.train_mask_set = value

    @test_mask.setter
    def test_mask(self, value):
        self.test_mask_set = value

class Copd():
    def __init__(self, path=None, data=None, time_stamp=None, undirected=True):

        self.time_stamp = time_stamp
        # todo here>> check which file does GetData read from. and what is the max_int_rep of nodes in these files
        # --numpy data
        if path is not None and data is not None:
            self.disease, self.non_uniq_labels, self.labels = GetData.disease_labels(path=path, time_stamp=time_stamp)
        else:
            self.disease, self.non_uniq_labels, self.labels = GetData.disease_labels(path=path, time_stamp=time_stamp)

        #TODO here>> check gene, non_uniq_disease, edges, labels, disease
        # > what do they suppose to look like in undirected vs in directed.?????
        self.gene, self.non_uniq_diseases = GetData.gene_disease(path=path, time_stamp=time_stamp)
        self.edges = GetData.edges(path=path, time_stamp=time_stamp, undirected=undirected)

        edges = np.array(list(map(self.nodes2idx().get, self.edges.T.flatten())), dtype=np.int32).reshape(-1, 2).T
        edges = pd.DataFrame(edges.T, columns=['geneid', 'diseaseid'])

        #--------add networkx graph and subgraph from edges
        self.graph, self.subgraph = self.get_graph(edges, undirected=undirected)
        #--------add adj
        self.adj =  self.create_adj(edges, undirected=True) #normalized adj undirected

        # display2screen(len(self.disease2idx()),len(self.genes2idx().keys()), 'line 186')
    def get_graph(self, edges, undirected=True):
        if undirected:
            graph = nx.from_pandas_edgelist(edges, 'geneid', 'diseaseid') # always have unqiue edges
        else:
            graph = nx.from_pandas_edgelist(edges, 'geneid', 'diseaseid', create_using=nx.DiGraph()) # edges have direction so to have undirected edges, soruce t0 target and target to source must be input edges.
        subgraph = my_utils.get_subgraph_disconnected(graph)[0]

        return graph, subgraph

    def create_adj(self, edges, undirected=True):
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges.iloc[:, 0], edges.iloc[:, 1])),
                            shape=(len(self.graph.nodes()), len(self.graph.nodes())), dtype=np.float32)  # 7374
        # build symmetric adjacency matrix
        if undirected:
            adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        adj = self.normalize_adj(adj)
        adj = torch.FloatTensor(np.array(adj.todense()))
        return adj

    # def load_data(self, path=f"./data/gene_disease/{args.time_stamp}/raw/", dataset="copd_label", time_stamp=''):
    #     """
    #     load data of cpod for full grpah and largest connected component
    #     return: adj = adj of subgraph
    #             labels = lables of subgraph # one diseaseid is not connected to the largest componenet
    #             g = subgraph of type networkx.Graph()
    #     """
    #     print('Loading {} dataset...'.format(dataset))
    #
    #     edges = self.edges
    #     # only use largest connected component
    #
    #     # edges_unordered = pd.DataFrame(edges_unordered.T, columns=['geneid', 'diseaseid'])
    #     # edges_unordered = np.array(list(map(self.nodes2idx().get, edges_unordered.to_numpy().flatten())), dtype=np.int32).reshape(-1,2).T
    #     edges = np.array(list(map(self.nodes2idx().get, edges.T.flatten())), dtype=np.int32).reshape(-1,2).T
    #
    #     edges = pd.DataFrame(edges.T, columns=['geneid', 'diseaseid'])
    #
    #     G = nx.from_pandas_edgelist(edges, 'geneid', 'diseaseid')
    #     g = my_utils.get_subgraph_disconnected(G)[0]
    #
    #     edges = [np.array((x, y), dtype=np.int32) for x, y in g.edges]
    #     edges = np.array(edges, dtype=np.int32).T  # shape = (2, 3678)
    #
    #     # label = 8 represent no class; it is used to label geneid
    #     # todo ???how should I label nodes that has no label? as None
    #     labels = set(map(self.disease2class().get, g.nodes))# {0, 1, 2, 3, 4, 5, 6, 7, None}
    #
    #     # labels = {i for i in labels if i is not None}
    #     labels = self.encode_onehot(labels)
    #
    #     # len(g.nodes()) = 2489
    #     adj = sp.coo_matrix((np.ones(edges.T.shape[0]), (edges.T[:, 0], edges.T[:, 1])),
    #                         shape=(len(G.nodes()), len(G.nodes())), dtype=np.float32) # 7374
    #
    #     # build symmetric adjacency matrix
    #     adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #     adj = self.normalize_adj(adj)
    #
    #     adj = torch.FloatTensor(np.array(adj.todense()))
    #     labels = torch.LongTensor(np.where(labels)[1])  # label is of type int NOT type one_hot
    #
    #
    #     return adj, labels, G, g

    # -- rename "conversion related function" (can be under this section)
    def labels2class(self):
        '''
            rename for diseases2class

        :return:
            {cuis: labels}
        '''
        cuis2labels = {self.disease2idx()[c]: self.labels2idx()[l] for c, l in zip(self.disease, self.non_uniq_labels)}

        return cuis2labels

    def class2labels(self):
        '''
            rename for class2disease
        :return:
            {class_label: [lsit of disease member of the class]}
        '''
        class2disease = {k: [] for k in self.labels2idx().keys()}
        for k, c in self.disease2class().items():
            class2disease[c].append(k)

        return class2disease

    def labelnodes2idx(self):
        '''
            rename for disease2idx
        :return: {label: label_rep}
        '''
        cuis = self.disease

        return {l: i for i, l in enumerate(list(collections.OrderedDict.fromkeys(cuis.tolist()).keys()))}

    # -- conversion related function
    #   :this must not be deleted for compatibility reason
    def nodes2idx(self):
        '''
            geneid and diseaseid are nodes in the graph
        :return: {geneid or diseaseid : id }
        '''

        return {**self.disease2idx(), **self.genes2idx()}  # concat 2 dictionary

    def disease2class(self):
        '''

        :return:
            {cuis: labels}
        '''

        cuis2labels = {self.disease2idx()[c]: self.labels2idx()[l] for c, l in zip(self.disease, self.non_uniq_labels)}

        return cuis2labels

    def class2disease(self):
        '''

        :return:
            {class_label: [lsit of disease member of the class]}
        '''
        class2disease = {k: [] for k in self.labels2idx().keys()}
        for k, c in self.disease2class().items():
            class2disease[c].append(k)

        return class2disease

    def labels2idx(self):
        '''
        :return: {label: label_rep}
        '''
        labels = self.non_uniq_labels
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
        # display2screen(np.unique(gene).shape)

        # 101 is the amount of diseaseid
        # note:
        #   > gene_idx is sorted by string representation of gene
        #       eg 10 100 1002 101
        #   instead of 10 100 101 1002
        # todo will this order has an effect on
        return {l: i + int(self.disease.shape[0]) for i, l in enumerate(list(collections.OrderedDict.fromkeys(gene.tolist()).keys()))}

    #=====================
    #==report dataset characteristic
    #=====================

    def report_copd_characters(self, plot=False, verbose=True):

        self.edges2nodes_ratio(verbose=verbose)
        self.label_rate(verbose=verbose)
        self.class_member_dist(plot=plot, verbose=verbose)
        self.rank_gene_overlap(plot=plot,verbose=verbose)

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

    def class_gene_dict(self):
        '''
        create class_gene dict where keys = diseases_classes and values = gene that has edges connected to its disease nodes

        :return: {disease_class: gene_connected_to_diseass}
        '''
        nodes = sorted([i for i in self.nodes2idx().values()])
        edges_flat = self.edges.T.flatten()
        edges_flat = [self.nodes2idx()[i] for i in edges_flat]
        '''
        edges_flat has the following format
            [d1,g1,d2,g2,d3,g3,...] where d = disease and g = gene
        '''
        # idx2disease = {d:i for i, d in self.nodes2idx().items()}
        class_gene = {l: [] for l in self.disease2class().values()}

        for i, n in enumerate(edges_flat[1::2]):
            if n not in self.disease2class().keys():  # gene
                gene_idx = n  # int_rep
                if i == 0:
                    disease_idx = edges_flat[0]  # int_rep
                else:
                    disease_idx = edges_flat[(i * 2) - 1]  # int_rep

                # -- add gene to its corresponding class
                class_gene[self.disease2class()[disease_idx]].append(gene_idx)
            else:
                disease_idx = n  # int_rep
                if i == 0:
                    gene_idx = edges_flat[0]  # int_rep
                else:
                    gene_idx = edges_flat[(i * 2) - 1]  # int_rep

                # -- add gene to its corresponding class
                class_gene[self.disease2class()[disease_idx]].append(gene_idx)
        return class_gene

    def rank_gene_overlap(self, verbose=False, plot=False):
        '''
        step tp step on how this func works
        1. for each class, create a set of gene that have edges connected to that class
        2. find the overlappi of gene between each classes and rank all of them
            : there will be n*n comparison where n = number of class
        :return:
        '''

        class_gene = self.class_gene_dict()

        intersect_gene_member = {}
        multi_class_gene = []

        # -- find gene overlap between classes
        count = 0
        for k_out, v_out in class_gene.items():
            for k_in, v_in in class_gene.items():
                intersect_gene_member[f"{k_out} & {k_in} "] = len(set(v_out) & set(v_in))
                multi_class_gene += list(set(v_out) & set(v_in))
                count +=1
                # print(f"gene intersection between {k_out} and {k_in} = {len(set(v_out) & set(v_in))}")

        multi_class_gene = set(multi_class_gene)

        assert count == len(class_gene.keys()) * len(class_gene.keys()), "loop is implemented incorreclty"

        intersect_gene_member = sorted(intersect_gene_member.items(), key=lambda t:t[1], reverse=True)

        for k in intersect_gene_member:
            print(f"gene intersection between {k[0]}= {k[1]}")
        print(f"number of gene that belongs to more than 1 class = {len(list(multi_class_gene))}")

        ratio = float(len(list(multi_class_gene)))/ float((len(self.nodes2idx().keys()) - len(self.disease2class().keys())))
        print(f"multi_class_gene to all_gene ratio = {ratio}")

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



    # -- read, write, convert files
    def create_rep_dataset(self, path=f'data/gene_disease/{args.time_stamp}/processed/rep/'):
        '''
        create files with value representtation of the following files
            >copd_label_content{time_stamp}.txt and copd_label_edges{time_stamp}.txt

        '''
        #TODO here>> check how rep create id or disease and gene (str or int???)

        # -- copd_label_edges.txt
        # disease is not the same as self.disease in that it is not uniq
        # genes, non_uniq_diseases = GetData.gene_disease()
        genes, non_uniq_diseases = self.gene, self.non_uniq_diseases

        # convert gene and non_uniq_disease to  its int_rep
        genes = list(map(self.genes2idx().get, genes))
        non_uniq_diseases = list(map(self.disease2idx().get, non_uniq_diseases))

        # --copd_label_content.txt
        # uniq_diseases, labels = GetData.disease_labels()
        uniq_diseases, labels = self.disease, self.non_uniq_labels

        uniq_diseases = list(map(self.disease2idx().get, uniq_diseases))
        labels = list(map(self.labels2idx().get, labels ))

        gene_disease = pd.DataFrame([genes,non_uniq_diseases], dtype=np.int32).T
        disease_label = pd.DataFrame([uniq_diseases,labels], dtype=np.int32).T

        # display2screen(max(uniq_diseases), max(labels), max(genes))
        # display2screen(gene_disease.head(), disease_label.head())

        # write to rep_copd_label_edges.txt
        write2files(gene_disease,path=f'data/gene_disease/{args.time_stamp}/processed/rep/', file_name=f"rep_copd_label_edges{self.time_stamp}.txt")

        # write to rep_copd_content.txt
        write2files(disease_label,path=f'data/gene_disease/{args.time_stamp}/processed/rep/', file_name=f"rep_copd_content{self.time_stamp}.txt")

class GetData():
    def __init__(self):
        pass

    @staticmethod
    def edges(path=f'data/gene_disease/{args.time_stamp}/raw/', data='copd_label_edges', time_stamp='', undirected=True):
    # def edges(path=f'data/gene_disease/{args.time_stamp}/raw/', data='copd_label_edges.txt'):
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
        if undirected:
            # edges are no longer sorted
            inverse_edge = edges.copy()
            inverse_edge[[0,1]] = inverse_edge[[1,0]]

            edges = np.hstack((edges,inverse_edge))

        return edges

    @staticmethod
    # def disease_labels(path=f'data/gene_disease/{args.time_stamp}/raw/', data='copd_label_content.txt'):
    def disease_labels(path=f'data/gene_disease/{args.time_stamp}/raw/', data='copd_label_content', time_stamp=''):
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
            non_uniq_labels = np.array(disease_labels.tolist()[1::2])
            uniq_labels = np.unique(non_uniq_labels)

        return uniq_disease, non_uniq_labels, uniq_labels

    @staticmethod
    def gene_disease(path=f'data/gene_disease/{args.time_stamp}/raw/', data='copd_label_edges', time_stamp=''):
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


class Cora():
    def __init__(self):
        pass

    # todo here>> check feature in cora why each node only have at most 1 feature???????
    def load_data(self, path="./data/raw/cora/", dataset="cora"):
        """Load citation network dataset (cora only for now)
        return:
            features is unormalized
        """
        print('Loading {} dataset...'.format(dataset))

        idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype='U')
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        labels = self.encode_onehot(idx_features_labels[:, -1])

        # build graph
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        idx_map = {str(j): i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype='U')
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype='U').reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        # features = self.normalize_features(features)
        unnormalize_adj = adj
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


        return adj, features, labels, idx_train, idx_val, idx_test, unnormalize_adj

    def normalize_adj(self,mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv_sqrt = np.power(rowsum, -0.5).flatten()
        r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
        r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
        return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt) # D^0-5 * A * D^0.5

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
