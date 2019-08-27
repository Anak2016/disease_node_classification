from __future__ import print_function, division
from skimage import io,transform
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# use node2vec with cora datset
import torch.nn.functional as F
import os.path as osp
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data as Data
import torch.nn as nn
import torch_geometric.transforms as T
import numpy as np
import scipy.sparse as sp
import torch
import networkx as nx
import matplotlib.pyplot as plt
import parameters as param
import time
import pandas as pd
import os
import random
import collections
import sys

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from datetime import datetime
from torch_geometric.nn import GCNConv, ChebConv, GATConv, SAGEConv  # noqa
from collections import OrderedDict
from node2vec import Node2Vec
from networkx.algorithms import bipartite
from my_utils import Cora, Copd, get_subgraph_disconnected, GetData, Conversion, create_copd_label_content, create_copd_label_edges, display2screen
from my_utils import create_onehot, create_adj_list, add_features
from my_utils import run_mlp, run_logist, run_gcn_on_disease_graph
from sys import path
from visualization import plot_2d
from arg_parser import args

# import sys
# sys.path.insert(0, r'C:\Users\Anak\PycharmProjects\AttentionWalk')
# from src.attentionwalk import AttentionWalkLayer

def pause():
    print("done")
    exit()

def save_node2vec_emb(G, save_path = 'output/gene_disease/embedding/node2vec/', EMBEDDING_FILENAME = 'node2vec_emb.txt', log=True):
    with open(save_path + EMBEDDING_FILENAME, 'w') as f:
        print("path is ok")

    # Precompute probabilities and generate walks - **ON WINDOWS ONLY WORKS WITH workers=1**
    node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200,
                        workers=4)  # Use temp_folder for big graphs
    s = time.time()

    # Embed nodes
    model = node2vec.fit(window=10, min_count=1,
                         batch_words=4)  # Any keywords acceptable by gensim.Word2Vec can be passed, `diemnsions` and `workers` are automatically passed (from the Node2Vec constructor)
    f = time.time()
    total = f-s
    print(f'total = {total}')
    output_path = save_path + EMBEDDING_FILENAME
    # Save embeddings for later use
    model.wv.save_word2vec_format(output_path)

    # # Save model for later use
    # model.save(output_path)

    if log:
        with open(f'./log/{EMBEDDING_FILENAME}', 'w') as f:
            f.write(f' --{save_path}{EMBEDDING_FILENAME}\n')
            f.write(f'total running time {total}')



def nx_plot(G, pos=None, node_color=None ):

    if pos is not None and node_color is not None:
        nx.draw(G, pos=pos, node_color=node_color)
        plt.show()
    else:
        nx.draw(G)
        plt.show()

def run_node2vec(copd, time_stamp=""):
    '''

    :return:
    '''
    # largest connected component -> get adj, label, ... -> get bipartite.sets -> save_node2vec_emb

    # when load data only use data that is
    # copd = Copd()

    # adj, features, labels, idx_train, idx_val, idx_test = cora.load_data()
    adj, labels, G, g = copd.load_data()

    # --color bipartite graph
    if args.plot:
        left, right = bipartite.sets(g)
        bipartite_color = [0 if i < len(left) else 1 for i, _ in enumerate(left.union(right))]
        pos = nx.circular_layout(g)
        nx_plot(g, pos=pos, node_color=bipartite_color)

    #  -- save node2vec embbedding to file
    # display2screen(len(g.nodes)) #2975
    # save_node2vec_emb(g,EMBEDDING_FILENAME=f"node2vec_emb_subgraph{time_stamp}.txt" )
    # display2screen(len(G.nodes)) #2996
    # save_node2vec_emb(G,EMBEDDING_FILENAME=f"node2vec_emb_fullgraph{time_stamp}.txt" )

def bine_copd_label(time_stamp=''):
    # load data in to dataframe
    # add u to gene and add i to item
    # create new columns of weight = 1 ( unweightd)

    # file = f"data/gene_disease/copd_label_edges{time_stamp}.txt"

    file = f"data/gene_disease/rep/rep_copd_label_edges{time_stamp}.txt"

    import pandas as pd
    df = pd.read_csv(file, sep='\t', header=None)
    gene_dict = {'geneid':[]}
    disease_dict = {'diseaseid': []}
    weight_dict = {'weight': []}
    for u,v in zip(df[0],df[1]):
        gene_dict['geneid'].append('u'+ str(u))
        disease_dict['diseaseid'].append('i'+ str(v))
        weight_dict['weight'].append(1)

    gene_dict.update(disease_dict)
    gene_dict.update(weight_dict)
    dict_ = gene_dict
    # print(dict_['diseaseid'][:5])
    df = pd.DataFrame(dict_, columns=['geneid', 'diseaseid', 'weight'] )
    save_path = f'data/gene_disease/bine/cope_label_edges{time_stamp}_.txt'
    df.to_csv(save_path, index=False, header=False, sep='\t')


# -- pytorch dataLoader
class FaceLandmarksDataset(Dataset):
    """Face lLandmarks fdataset."""
    def __init__(self, csv_file, root_dir, transform=None):
        """

        :param csv_file: path to csv
        :param root_dir: Director with all th eimages
        :param transform: optinal transform to be appleid on a smaple
        """
        # to name of image and its landmarks
        self.landmarks_frame = pd.read_csv(csv_file)
        # to be concat with name of imges. This is good because img is not in the memory until it is accessed
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)

        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample



class Rescale(object):
    """Rescale the image in a smaple to a givne size
    Args: output_size(tuple or int): Desired output size.
        If tuple, output is matched to output_size.
        If int, smaller of image edges is matched to output_size keepign aspect ratio the same
    """
    def __init__(self, output_size):
        assert isinstance(output_size,  (int,tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # what is the value of image?
        h,w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h >w:
                new_h, new_w = self.output_size * h/w, self.output_size # preserve proportion of the original image
            else:
                new_h, new_w = self.output_size, self.output_size * w/h
            new_h, new_w = int(new_h), int(new_w)

            img = transform.resize(image, (new_h, new_w))

            landmarks= landmarks * [new_w / w, new_h / h]
            return {'image': img, 'landmarks': landmarks}



class RandomCrop(object):
    """Crop randomly the image in a asmple

    Args: output_size(tuple or int): Desired output size. if int, square crop is made
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int,tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h,w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0,w - new_w)

        image = image[top: top+new_h, left: left +new_w]
        landmarks = landmarks - [left, top ]

        return {'image': image, 'landmarks': landmarks}


class ToTensor(object):
    """Convert ndarrays in smaple to Tensors ]"""
    def __call__(self, sample):
        image, landmarks = sample['image'],  sample['landmarks']
        image = image.transpose((2,0,1))
        return {'image': torch.from_numpy(image), 'landmarks': torch.from_numpy(landmarks)}


def show_landmarks_batch(sample_batched):
    """Show image with landwmarks for a batch of sample"""
    images_batch , landmarks_batch = sample_batched['image'], sample_batched['landmarks']
    batch_size = len(images_batch)
    # (batch, channel, w, h)
    im_size = images_batch.size(2) # w; w == h
    grid_border_size = 2 # for padding

    grid = utils.make_grid(images_batch, padding=2)

    plt.imshow(grid.numpy().transpose((1,2,0))) # (w, h, c)

    for i in range(batch_size):

        # I dont undersatnd the code below?
        # | pic0 | pic1 | pic2 | pic3 | pic4 | pic5 =>> padding is pic_number +1
        plt.scatter(landmarks_batch[i, :, 0].numpy() + i * im_size +(i+1) *grid_border_size,
                    landmarks_batch[i, :, 1].numpy() + grid_border_size,
                    s=10, marker='.', c='r')

        plt.title('Batch from dataloader')

def create_pytorch_dataset(path='data/gene_disease/', files=['copd_label_content','copd_label_edges']):
    '''convert copd_label to pytorch_dataset'''
    import warnings
    warnings.filterwarnings("ignore") # what does thi sdo?
    plt.ion() #set ineteractive mode
    import pandas as pd
    landmarks_frame = pd.read_csv('data/faces/face_landmarks.csv')
    n = 65 #person-7.jpg
    img_name = landmarks_frame.iloc[n,0]
    landmarks = landmarks_frame.iloc[n,1:].as_matrix()
    # print(landmarks)
    landmarks = landmarks.astype('float').reshape(-1,2)
    # print('Image name: {}'.format(img_name))
    # print('Landmarks shape: {}'.format(landmarks.shape))
    # print('First 4 Landmarks: {}'.format(landmarks[:4]))

    def show_landmarks(image, landmarks):
        # plt.figure()
        plt.imshow(image)
        print(landmarks)
        try:
            plt.scatter(landmarks[:, 0], landmarks[:,1], s=10, marker=',', c='r')
        except:
            print(landmarks)
            exit()
            print("here")
        plt.pause(1)
        # plt.show()

    import os
    # show_landmarks(io.imread(os.path.join('data/faces/', img_name)), landmarks)


    face_dataset = FaceLandmarksDataset(csv_file="data/faces/face_landmarks.csv",root_dir='data/faces/')

    # #--plot image and landmarks on matplotlib
    #
    # fig = plt.figure()
    #
    # for i in range(len(face_dataset)):
    #     sample = face_dataset[i]
    #
    #     print(i, sample['image'].shape, sample['landmarks'].shape)
    #
    #     ax = plt.subplot(1, 4, i + 1)
    #     plt.tight_layout()
    #     ax.set_title('Sample #{}'.format(i))
    #     ax.axis('off')
    #     # print(sample.keys())
    #     # exit()
    #     show_landmarks(**sample)
    #
    #     if i == 3:
    #         plt.show()
    #         plt.pause(10)
    #         break

    # # -- compose transforoms
    # scale = Rescale(256)
    # crop = RandomCrop(128)
    #
    # # setting trandform function for torchvision
    # composed = transforms.Compose([Rescale(256), RandomCrop(224)])
    #
    # fig = plt.figure()
    # sample = face_dataset[65]
    # for i, tsfrm in enumerate([scale, crop, composed]):
    #     transformed_sample = tsfrm(sample)
    #     ax = plt.subplot(1, 3, i + 1)
    #     plt.tight_layout()
    #     ax.set_title(type(tsfrm).__name__)
    #     show_landmarks(**transformed_sample)
    #
    # plt.show()

    # -- iterating throguh the dataset
    transformed_dataset = FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv',
                                               root_dir='data/faces/',
                                               transform=transforms.Compose([
                                                   Rescale(256),
                                                   RandomCrop(224),
                                                   ToTensor()
                                               ]))
    # missing out of
    #> batching the data
    #> shuffiling th edata
    #> load the data in parallel using multiprocessing workers
    for i in range(len(transformed_dataset)):
        sample = transformed_dataset[i]

        print(i, sample['image'].size(), sample['landmarks'].size(0))
        if i ==3:
            break

    # DataLoader does batch_size of both keys of transformed_dataset
    # tranformed_dataset has the following form {key1:val1, key2:val2}
    dataloader = DataLoader(transformed_dataset, batch_size=4, shuffle=True, num_workers=4)

    #Helper function to show a batch


    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image'].size(),
              sample_batched['landmarks'].size())
        if i_batch == 3:
            plt.figure()
            show_landmarks_batch(sample_batched) # ??
            plt.axis('off')
            plt.show() # -- I may deleted something here
            break

class geomertric_dataset(Data):

    def __init__(self, data, x=None, edges_index=None, edge_attr=None, y=None, split=0.8):
        '''
            all parameters must have type of torch.tensor expect split and data
        :param data: copd
        :param x: n*d where d = features if no feature d = n
        :param edges_index: (2, # of edges)
        :param edge_attr: (# of edges, # of edge attr) # i am not sure myself
        :param y:  n where n = $ of nodes
        :param split:
        '''
        self.dataset = data

        # -- intialization of variable from self.dataset
        self.labeled_nodes  = self.dataset.labelnodes2idx
        # self.unlabled_nodes = self.dataset.genes2idx().keys()
        # self.nodes          = self.dataset.nodes2idx().keys()

        super(geomertric_dataset, self).__init__(x,edges_index,edge_attr,y)
        self.split = split
        self.y = y
        # -- masking
        self.train_mask_set = []
        self.test_mask_set = []
        ind = 0  # inc everytime nodes are added; check how many nodes are included in training set
        count = 0
        arr_ind = 0 # inc everytimes for loop go through all of the classes; it represent current ind of val_list in each class.

        while True:
            max_class_int_rep = self.num_classes - 1 # max int_rep of all classes
            current_class = count % max_class_int_rep

            # -- create a random split of training and testing dataset
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

        # display2screen(ind, count)
        self.test_mask_set = list(set([i for i in self.dataset.labelnodes2idx().values()]).difference(self.train_mask_set))
        # display2screen(max(self.test_mask_set), max(self.train_mask_set)) # todo here>>

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

def run_GCN(data = None, emb_name=None, time_stamp=None, tuning=False, log=False, plot=False, verbose=False, **kwargs):


    param = {
        # Pseudo-Label
        'T1': int(args.t1_t2_alpha[0]),
        'T2': int(args.t1_t2_alpha[1]),
        'af': float(args.t1_t2_alpha[2])}

    # -- initalization
    if args.dataset == 'cora':
        args.weighted_class = [1,1,1,1,1,1,0]
        weighted_class = args.weighted_class

    class Net(torch.nn.Module):
        def __init__(self, dropout=None):
            super(Net, self).__init__()

            self.dropout = dropout
            #==============
            #== USING MODULE LSIT
            #==============
            # if args.add_features:
            #     data.num_features

            # display2screen("data.num_features",data.num_features)

            if args.arch == 'gcn':
                modules = {
                        # "conv1": GCNConv(64, args.hidden, cached=True),
                        "conv1":GCNConv(data.num_features, args.hidden, cached=True),
                        "conv2":GCNConv(args.hidden , data.num_classes, cached=True)
                }

            if args.arch == 'gat':
                modules = {
                        "conv1": GATConv(data.num_features, args.hidden, heads=args.heads,dropout=0.6),
                        "conv2": GATConv(args.hidden * args.heads, data.num_classes, heads=1, concat=True, dropout=0.6)
                }
            if args.arch == 'sage':
                modules = {
                        "conv1": SAGEConv(data.num_features, args.hidden,aggr=args.aggr),
                        "conv2": SAGEConv(args.hidden , data.num_classes,aggr=args.aggr)
                }

            for name,module in modules.items():
                self.add_module(name,module)

            # [1]-- loss function max out really early on the test dataset
            #   > max out early
            # self.conv1 = GCNConv(data.num_features, args.hidden, cached=True)
            # self.conv2 = GCNConv(args.hidden , data.num_classes, cached=True)

            # [2]-- very smoth. model stop learning at around epoch 90 but accuracy is only 50-mid50s
            #   >max out early
            # self.conv1 = GCNConv(data.num_features, 32, cached=True)
            # self.conv2 = GCNConv(32 , data.num_classes, cached=True)

            # [3]-- loss function max out slowly compare to the [1], and a lot less smooth
            #   >max out quite late. 150
            # self.conv1 = GCNConv(data.num_features, 32, cached=True)
            # self.conv2 = GCNConv(32, 8, cached=True)
            # self.conv3 = GCNConv(8 , data.num_classes, cached=True)

        def forward(self):

            x, edge_index = data.x, data.edge_index
            # display2screen(x[1,:])
            # display2screen(x.shape, edge_index.shape, np.amax(edge_index.numpy()))
            x = x.type(torch.float)
            edge_index = edge_index.type(torch.long)
            dropout = self.dropout

            if args.arch == 'gcn':
                # todo check loss fucntion for gcn
                x = F.relu(self.conv1(x, edge_index))
                x = F.dropout(x, p=dropout, training=self.training)
                x = self.conv2(x, edge_index)
                # x = F.dropout(x, p=dropout, training=self.training)
                # x = self.conv3(x, edge_index)
                return F.log_softmax(x, dim=1)

            if args.arch == 'gat':

                # todo what is the loss function use in gat?
                x = F.dropout(data.x, p=0.6, training=self.training)
                x = F.elu(self.conv1(x, edge_index))
                x = F.dropout(x, p=0.6, training=self.training)
                x = self.conv2(x, edge_index)
                return F.log_softmax(x, dim=1)

            if args.arch == 'sage':

                # todo what is the los func use in sage?
                # graphSage original architecutre has depth = 2.
                x = F.relu(self.conv1(x, edge_index))
                # x = F.dropout(x, p=dropout, training=self.training)
                x = self.conv2(x, edge_index)
                return F.log_softmax(x, dim=1)

    def unlabeled_weight(epoch):
        alpha = 0.0
        if epoch > param['T1']:
            if epoch > param['T2']:
                alpha = param['af']
            else:
                alpha = (epoch - param['T1']) / (param['T2'] - param['T1'] * param['af'])
        return alpha


    def train(epoch, weighted_class,labeled_index, target):
        model.train()
        optimizer.zero_grad()

        if args.pseudo_label_all:
            # -- pseudo_label is differnet from cost-sensitivity
            labeled_loss = F.nll_loss(model()[data.train_mask], data.y[data.train_mask], weight=weighted_class, reduction="mean")

            # for unlabled dataset
            unlabeled_nodes = [ i for i in range(len(data.y)) if i>np.amax(data.train_mask.numpy().tolist() + data.test_mask.numpy().tolist()) ]

            # labeled pseduo_label by class that is predicted with the most confident
            pseudo_label_pred = model()[unlabeled_nodes].max(1)[1]

            unlabeled_loss = F.nll_loss(model()[unlabeled_nodes], pseudo_label_pred, weight=weighted_class, reduction='mean')
            # unlabeled_loss = unlabeled_loss/ len(unlabeled_nodes)

            loss_output = labeled_loss + unlabeled_weight(epoch)*unlabeled_loss


        # todo maybe i can update target and label_index inside of "data" instance
        elif args.pseudo_label_topk:
            th = 0
            if epoch == th:
                # -- pseudo_label is differnet from cost-sensitivity
                if epoch == th and labeled_index is None:
                    labeled_index = data.train_mask
                    target = data.y[labeled_index] # target = labeled_data

                # -- try append all 1 as prediction, nothing happen
                # tmp1 = torch.ones(2000, dtype=torch.long)
                # tmp1 = torch.cat((data.train_mask, tmp1),0)
                # tmp2 = torch.tensor([5 for i in range(2000)], dtype=torch.long)
                # tmp2 = torch.cat((data.y[data.train_mask], tmp2),0)
                # labeled_loss = F.nll_loss(model()[tmp1], tmp2, weight=weighted_class, reduction="mean")

                # -- leak of label of test dataset
                # labeled_loss = F.nll_loss(model()[labeled_index], data.y[labeled_index], weight=weighted_class, reduction="mean")

                # -- no leak of label of test dataset
                labeled_loss = F.nll_loss(model()[labeled_index], target, weight=weighted_class, reduction="mean")

                # -- index of all nodes
                all_nodes = [i for i in range(len(data.y))]

                assert 1 + np.amax(data.train_mask.numpy().tolist() + data.test_mask.numpy().tolist()) == len(data.labeled_nodes().keys()), f"{len(data.labeled_nodes().keys())} != {1 + np.amax(data.train_mask.numpy().tolist() + data.test_mask.numpy().tolist())}"

                #-- labeled top k most confidence node to be pseduo_labels
                pseudo_label_pred = model()[all_nodes].max(1)[1]

                tmp = model()[all_nodes].max(1)[1].detach().flatten().tolist()
                tmp = [(l,i) for i,l in enumerate(tmp)]
                tmp = sorted(tmp, key=lambda x:x[0], reverse=True) # rank label by predicted confidence value

                ranked_labels = [(l,i) for (l,i) in tmp]
                top_k_tuple = []

                for (l,i) in ranked_labels:
                    if len(top_k_tuple) >= int(args.topk):
                        break

                    # if True: # obtian highest accuracy; basically always append i
                    # if i not in  list(copd.labelnodes2idx().values()): # bad accuracy because label is not added along with it
                    if i not in  labeled_index: # this also obtain highest accuracy; every label can be included at most 1 time in loss function
                        top_k_tuple.append((i,l)) # get index of top_k to be masked during loss

                    # todo here>> always add the most confident
                    # top_k_tuple.append((i, l))  # get index of top_k to be masked during loss

                top_k = [t[0] for t in top_k_tuple]
                new_labels = [t[1] for t in top_k_tuple]

                # -- add new label to target
                if len(top_k) != int(args.topk):
                    # -- what is the condision that have to be satisfy when this condition is true.
                    pass
                else:
                    assert len(top_k) == int(args.topk), "len(top_k) != int(args.topk)"

                if len(top_k) >0:
                    target = torch.cat((target, torch.tensor(new_labels)), 0)

                # -- add top_k to labeld_loss
                if epoch > th and len(top_k) > 0:
                    len_before_topk = labeled_index.shape[0]
                    labeled_index = torch.cat((labeled_index, torch.tensor(top_k)),0)
                    assert len_before_topk + len(top_k) == labeled_index.shape[0], "recently added top_k index are already included in labled_index"

                    unlabeled_loss = F.nll_loss(model()[top_k], pseudo_label_pred[top_k], weight=weighted_class, reduction='mean')
                    # unlabeled_loss = F.nll_loss(model()[top_k], torch.tensor(new_labels), weight=weighted_class, reduction='mean')

                    loss_output = labeled_loss + unlabeled_weight(epoch) * unlabeled_loss

                else:
                    if len(top_k)> 0:
                        len_before_topk = labeled_index.shape[0]

                        labeled_index = torch.cat((labeled_index, torch.tensor(top_k)), 0)
                        assert len_before_topk + len(top_k) == labeled_index.shape[0], "recently added top_k index are already included in labled_index"

                        unlabeled_loss = F.nll_loss(model()[top_k], pseudo_label_pred[top_k], weight=weighted_class, reduction='mean')
                        # unlabeled_loss = F.nll_loss(model()[top_k], torch.tensor(new_labels), weight=weighted_class,reduction='mean')

                        loss_output = labeled_loss + unlabeled_weight(epoch) * unlabeled_loss
                    else:
                        # top_k == 0 => unlabeled_loss has no input => output of unlabeled_los s= 0
                        unlabeled_loss = 0
                        loss_output = labeled_loss + unlabeled_weight(epoch) * unlabeled_loss
            else:

                loss_output = F.nll_loss(model()[data.train_mask], data.y[data.train_mask], weight=weighted_class, reduction="mean")

        elif args.pseudo_label_topk_with_replacement:

            # -- index of all nodes
            all_nodes = [i for i in range(len(data.y))]

            assert 1 + np.amax(data.train_mask.numpy().tolist() + data.test_mask.numpy().tolist()) == len(
                data.labelnodes2idx().keys()), f"{len(data.labelnodes2idx().keys())} != {1 + np.amax(data.train_mask.numpy().tolist() + data.test_mask.numpy().tolist())}"

            # -- labeled top k most confidence node to be pseduo_labels
            pseudo_label_pred = model()[all_nodes].max(1)[1]

            tmp = model()[all_nodes].max(1)[1].detach().flatten().tolist()
            tmp = [(l, i) for i, l in enumerate(tmp)]
            tmp = sorted(tmp, key=lambda x: x[0], reverse=True)  # rank label by predicted confidence value

            ranked_labels = [(l, i) for (l, i) in tmp]
            top_k_tuple = []

            for (l, i) in ranked_labels:
                top_k_tuple.append((i, l))  # get index of top_k to be masked during loss

                if len(top_k_tuple) >= int(args.topk):
                    break
            top_k = [t[0] for t in top_k_tuple]
            new_labels = [t[1] for t in top_k_tuple]

            # -- add new label to target
            if len(top_k) != int(args.topk):
                # -- what is the condision that have to be satisfy when this condition is true.
                pass
            else:
                assert len(top_k) == int(args.topk), "len(top_k) != int(args.topk)"

            # -- labled_loss
            labeled_loss = F.nll_loss(model()[data.train_mask], data.y[data.train_mask], weight=weighted_class, reduction="mean")

            # -- add top_k to labeld_loss
            if epoch > 1 and len(top_k) > 0:

                # unlabeled_loss = F.nll_loss(model()[top_k], pseudo_label_pred[top_k], weight=weighted_class, reduction='mean')
                unlabeled_loss = F.nll_loss(model()[top_k], torch.tensor(new_labels), weight=weighted_class,
                                            reduction='mean')

                loss_output = labeled_loss + unlabeled_weight(epoch) * unlabeled_loss

            else:
                if len(top_k) > 0:

                    # unlabeled_loss = F.nll_loss(model()[top_k], pseudo_label_pred[top_k], weight=weighted_class, reduction='mean')
                    unlabeled_loss = F.nll_loss(model()[top_k], torch.tensor(new_labels), weight=weighted_class,
                                                reduction='mean')

                    loss_output = labeled_loss + unlabeled_weight(epoch) * unlabeled_loss
                else:
                    # top_k == 0 => unlabeled_loss has no input => output of unlabeled_los s= 0
                    unlabeled_loss = 0
                    loss_output = labeled_loss + unlabeled_weight(epoch) * unlabeled_loss
        else:
            # todo here>>
            loss_output = F.nll_loss(model()[data.train_mask], data.y[data.train_mask], weight=weighted_class, reduction="mean")

        # untrain_model is returned for just the first iteration
        untrain_model = model()

        try:
            loss_output.backward()
        except UnboundLocalError as e:
            display2screen(f"epoch = {epoch}", e)

        optimizer.step()

        return model(), loss_output.data, untrain_model, labeled_index, target

    def test():
        # add f1 here

        model.eval()
        logits, accs = model(), []

        # for _,mask in data('train_mask', 'test_mask'):
        for mask in [data.train_mask, data.test_mask]:
            pred = logits[mask].max(1)[1]

            acc = pred.eq(data.y[mask]).sum().item()/ mask.shape[0]

            accs.append(acc)
        return accs

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    count = 0
    curr_time = datetime.now().strftime('%Y_%m_%d_%H_%M')
    try:
        if tuning:
            folder = f"log/{time_stamp}/{args.arch}/hp_tuning"
            if not os.path.exists(folder):
                os.makedirs(folder)

            #-- write to file
            save_path = f'{folder}/{emb_name}{curr_time}.txt'


            f = open(save_path, 'w')

            best_hp_config = {0:0}
            write_stat = False
            while True:

                dropout = 0.1 * random.randint(3, 8)
                lr = round(random.uniform(0.01,0.1),2)

                # decay_coeff =  0.1 * torch.FloatTensor([random.randint(1,9)])
                # decay_power = torch.FloatTensor([random.randint(1,4)])
                decay_coeff =  round(random.randint(1,9),2)
                decay_power = random.randint(2,4)

                weight_decay = decay_coeff/10**decay_power

                model = Net(dropout).to(device)

                # optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4) # original before modify
                # best lr = 0.05 weight_decay=5e-4 ====> around 60-70 percent
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

                # model() -> optimizer -> loss -> model.train()-> optimizer.zero_grad() -> loss.backward() -> optimizer.step() -> next epoch
                best_val_acc =test_acc = 0
                best_test_acc = [0]
                log_list = []

                # for epoch in range(1,201):
                for epoch in range(1,args.epochs):
                    gcn_emb, loss_epoch, _ = train(epoch)

                    train_acc, test_acc = test()

                    if verbose:
                        logging = 'Epoch: {:03d}, Train: {:.4f}, Test: {:.4f}'.format(epoch, train_acc, test_acc)
                        print(logging)

                    if test_acc > best_test_acc[0]:
                        best_test_acc.insert(0,test_acc)
                        if len(best_test_acc) > 10:
                            best_test_acc.pop(-1)


                # if sum(best_test_acc)/len(best_test_acc) > list(best_hp_config.values())[0]:
                if best_test_acc[0] > list(best_hp_config.values())[0]:
                    best_hp_config[f"dropout = {dropout}; lr = {lr} ; weight_decay = {weight_decay}"] = best_hp_config.pop(list(best_hp_config.keys())[0])
                    best_hp_config[f"dropout = {dropout}; lr = {lr} ; weight_decay = {weight_decay}"] = best_test_acc[0]
                    write_stat = True
                txt = '\n'.join(["========================",
                                f"loop = {count}",
                                f"dropout = {dropout}; lr = {lr} ; weight_decay = {weight_decay}",
                                f"top 10 best acc = {best_test_acc}",
                                f"average = {sum(best_test_acc)/len(best_test_acc)}",
                                f"!!! current best config is  **{list(best_hp_config.keys())[0]}** with best_acc = {best_hp_config[list(best_hp_config.keys())[0]]} and avg_acc = {sum(best_test_acc)/len(best_test_acc)} !!!"])

                txt = txt + '\n'
                print(txt)

                # -- write to file
                if write_stat:
                    print("writing to file ...")
                    f.write(txt)
                    write_stat = False

                count += 1

        else:
            #==========================
            #==== NOT TUNING HYPER-PARAMETERS
            #==========================
            x, edge_index = data.x, data.edge_index
            x = x.type(torch.float)
            edge_index = edge_index.type(torch.long)

            # dropout = self.dropout
            # # display2screen(dropout)
            # # display2screen(edge_index.shape, x.shape, torch.max(edge_index))
            #
            # x = F.relu(self.conv1(x, edge_index))
            # x = F.dropout(x, p=dropout, training=self.training)
            # x = self.conv2(x, edge_index)
            # # x = F.dropout(x, p=dropout, training=self.training)
            # # x = self.conv3(x, edge_index)
            # return F.log_softmax(x, dim=1)

            # labeled_loss = F.nll_loss(model()[data.train_mask], data.y[data.train_mask], weight=weighted_class)
            # display2screen(*config['gcn'])
            model = Net(args.dropout).to(device) # old style

            # display2screen(verbose)

            lr = 0.01
            weight_decay = 5e-4
            if kwargs.get('lr'):
                lr = kwargs.get('lr')
            if kwargs.get('weight_decay'):
                weight_decay = kwargs.get('weight_decay')

            # original before modify
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

            # model() -> optimizer -> loss -> model.train()-> optimizer.zero_grad() -> loss.backward() -> optimizer.step() -> next epoch
            best_val_acc = test_acc = 0
            log_list = []

            best_epoch = {0: [0]}
            best_test_acc = [0]
            # best_epoch = 0
            loss_hist = []
            train_acc_hist = []
            test_acc_hist = []

            weighted_class = torch.tensor(list(map(int,args.weighted_class))+[0], dtype=torch.float)

            labeled_index = None
            target        = None
            # for epoch in range(1, 201):
            for epoch in range(1, args.epochs):
                if epoch == 1:
                    _, loss_epoch, gcn_emb_no_train,labeled_index, target = train(epoch,weighted_class,labeled_index, target)
                else:
                    gcn_emb, loss_epoch, _ ,labeled_index, target = train(epoch,weighted_class,labeled_index, target)
                # display2screen(gcn_emb.numpy())
                loss_hist.append(loss_epoch.tolist())

                train_acc, test_acc = test()

                logging = 'Epoch: {:03d}, Train: {:.4f}, Test: {:.4f}'.format(epoch, train_acc, test_acc)
                log_list.append(logging)
                train_acc_hist.append(train_acc)
                test_acc_hist.append(test_acc)

                if test_acc > best_test_acc[0]:
                    best_test_acc.insert(0, test_acc)
                    if len(best_test_acc) > 3:
                        best_test_acc.pop(-1)

                    best_epoch.pop(list(best_epoch.keys())[0])
                    best_epoch[f"epoch = {epoch}"] = best_test_acc

                if verbose:
                    print(logging)

            # -- print set of best accuracy and its epoch.
            if verbose:
                print(f"!!!!! {list(best_epoch.keys())[0]} = {best_epoch[list(best_epoch.keys())[0]]} !!!!!!! ")

            # ================
            # == intilize logging naming convention
            # ================

            split = data.split
            # -- create dir for hyperparameter config if not already exists
            # display2screen(args.dataset == 'cora')


            weighted_class = ''.join(list(map(str, args.weighted_class)))

            HP = f'lr={args.lr}_d={args.dropout}_wd={args.weight_decay}'
            folder = f"log/{time_stamp}/{args.arch}/{emb_name}/split={split}/{HP}/"

            if not os.path.exists(folder):
                os.makedirs(folder)

            if args.add_features:
                feat_stat = "YES"
            else:
                feat_stat = "NO"

            if args.pseudo_label_all:
                pseudo_label_stat = "ALL"
            elif args.pseudo_label_topk:
                pseudo_label_stat = "TOP_K"
            elif args.pseudo_label_topk_with_replacement:
                pseudo_label_stat = "TOP_K_WITH_REPLACEMENT"
            else:
                pseudo_label_stat = "NONE"


            T_param = ','.join([str(param['T1']),str(param['T2'])])
            # -- creat directory if not yet created
            save_path = f'{folder}/img/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            if args.plot_all is True:
                args.plot_loss = True
                args.plot_no_train = True
                args.plot_train = True

            # -- plot loss function as epoch increases.
            if args.plot_loss:
                # ======================
                # == plot loss and acc vlaue
                # ======================
                plt.figure(1)
                # -- plot loss hist
                plt.subplot(211)
                plt.plot(range(len(loss_hist)), loss_hist)
                plt.ylabel("loss values")
                plt.title("loss history")

                # -- plot acc hist
                plt.subplot(212)
                plt.plot(range(len(train_acc_hist)), train_acc_hist)
                plt.plot(range(len(test_acc_hist)), test_acc_hist)
                plt.ylabel("accuracy values")
                plt.title("accuracy history")
                print("writing to  "+save_path+f"LOSS_ACC_feat={feat_stat}_pseudo_label={pseudo_label_stat}_wc=[{weighted_class}]_T=[{T_param}]_topk={args.topk}.png")
                plt.savefig(save_path+f'ACC_feat={feat_stat}_pseudo_label={pseudo_label_stat}_wc=[{weighted_class}]_T=[{T_param}]_topk={args.topk}.png')
                plt.show()

            # ==========================
            # === plot 2D output GCN embedding
            # ==========================
            # save_path = f'output/gene_disease/embedding/gcn/'

            if args.plot_no_train:
                file_gcn_emb = f"TRAIN=NO_ACC_feat={feat_stat}_pseudo_label={pseudo_label_stat}_wc=[{weighted_class}]_T=[{T_param}]_topk={args.topk}.txt"
                img_gcn_emb = f"TRAIN=NO_ACC_feat={feat_stat}_pseudo_label={pseudo_label_stat}_wc=[{weighted_class}]_T=[{T_param}]_topk={args.topk}.png"
                # file_gcn_emb_no_train = f'gcn_emb_no_train{lr}_{weight_decay}.txt'

                # display2screen(df)
                # -- df for gcn_emb_no_train
                # gcn_emb_no_train = np.append(gcn_emb_no_train.detach().numpy(), gcn_emb_no_train.max(1)[1]

                df = pd.DataFrame(data.x.numpy()) # output before first epoch
                # df = pd.DataFrame(gcn_emb_no_train.detach().numpy()) # output after first epoch

                df.to_csv(save_path + file_gcn_emb, sep=' ')

                df_pred = pd.DataFrame(pd.DataFrame({"pred": gcn_emb_no_train.max(1)[1].detach().numpy()}))

                # -- gcn emb with no training feedback
                print("--gcn emb with no training feedback")

                # todo what is the format of predicted nodes
                if args.plot_all is True:
                    plot_2d(data.dataset, save_path, file_gcn_emb, emb='gcn', with_gene=True, func=args.plot_2d_func, log=args.log,save_img=img_gcn_emb, pred_label=df_pred)
                    plot_2d(data.dataset, save_path, file_gcn_emb, emb='gcn', with_gene=False, func=args.plot_2d_func, log=args.log,save_img=img_gcn_emb, pred_label=df_pred)
                else:
                    plot_2d(data.dataset, save_path, file_gcn_emb, emb='gcn', with_gene=args.with_gene,
                            func=args.plot_2d_func, log=args.log, save_img=img_gcn_emb, pred_label=df_pred)
            if args.plot_train:
                file_gcn_emb = f"TRAIN=YES_ACC_feat={feat_stat}_pseudo_label={pseudo_label_stat}_wc=[{weighted_class}]_T=[{T_param}]_topk={args.topk}.txt"
                img_gcn_emb = f"TRAIN=YES_ACC_feat={feat_stat}_pseudo_label={pseudo_label_stat}_wc=[{weighted_class}]_T=[{T_param}]_topk={args.topk}.png"
                # file_gcn_emb = f'gcn_emb_{lr}_{weight_decay}.txt'

                # -- df for gcn_emb
                df = pd.DataFrame(gcn_emb.detach().numpy())
                df.to_csv(save_path+file_gcn_emb, sep=' ')

                df_pred = pd.DataFrame(pd.DataFrame({"pred": gcn_emb.max(1)[1].detach().numpy()}))

                # -- gcn emb with training feedback
                print("--gcn emb with training feedback")
                if args.plot_all is True:
                    plot_2d(data.dataset, save_path, file_gcn_emb, emb='gcn', with_gene=True,
                            func=args.plot_2d_func, log=args.log, save_img=img_gcn_emb, pred_label=df_pred)
                    plot_2d(data.dataset, save_path, file_gcn_emb, emb='gcn', with_gene=False,
                            func=args.plot_2d_func, log=args.log, save_img=img_gcn_emb, pred_label=df_pred)
                else:
                    plot_2d(data.dataset, save_path, file_gcn_emb, emb='gcn', with_gene=args.with_gene, func=args.plot_2d_func, log=args.log, save_img=img_gcn_emb, pred_label=df_pred)

            #==========================
            #== logging
            #==========================

            #--train_mask f1,precision,recall
            train_pred = model()[data.train_mask].max(1)[1]
            train_f1 = f1_score(data.y[data.train_mask], train_pred, average='micro')
            train_precision = precision_score(data.y[data.train_mask], train_pred, average='micro')
            train_recall = recall_score(data.y[data.train_mask], train_pred, average='micro')

            #-- test_mask f1,precision,recall
            test_pred = model()[data.test_mask].max(1)[1]
            test_f1 = f1_score(data.y[data.test_mask], test_pred , average='micro')
            test_precision = precision_score(data.y[data.test_mask], test_pred, average='micro')
            test_recall = recall_score(data.y[data.test_mask], test_pred, average='micro')

            if args.log:
                # save_path = f'log/{args.arch}/{HP}/{time_stamp}/feat_stat={feat_stat}_{args.arch}_accuracy_{emb_name}{time_stamp}_split_{split}.txt'
                save_path = f'{folder}ACC_feat={feat_stat}_pseudo_label={pseudo_label_stat}_wc={weighted_class}.txt'
                print(f"writing to {save_path}...")
                with open(save_path, 'w') as f:
                    txt = '\n'.join(log_list)
                    f.write(txt)

            if args.log:
                cm_train = confusion_matrix(model()[data.train_mask].max(1)[1], data.y[data.train_mask])
                cm_test = confusion_matrix(model()[data.test_mask].max(1)[1], data.y[data.test_mask])

                # formatter = {'float_kind': lambda x: "%.2f" % x})
                cm_train = np.array2string(cm_train)
                cm_test = np.array2string(cm_test)

                save_path = f'{folder}CM_feat={feat_stat}_pseudo_label={pseudo_label_stat}_wc={weighted_class}.txt'
                print(f"writing to {save_path}...")

                # txt = 'class int_rep is [' + ','.join(list(map(str, np.unique(data.y.numpy()).tolist()))) + ']'
                txt = 'class int_rep is [' + ','.join([str(i) for i in range(data.num_classes)]) + ']'
                txt = txt + '\n\n' + "training cm" + '\n' + cm_train + '\n' \
                      + f"training_accuracy ={log_list[-1].split(',')[1]}" + '\n'  \
                      + f"training_f1       ={train_f1}" + '\n' \
                      + f"training_precision={train_precision}" + '\n' \
                      + f"training_recall   ={train_recall}" + '\n'

                txt = txt + '\n\n' + "test cm" + '\n' + cm_test + '\n' \
                      + f"test_accuracy ={log_list[-1].split(',')[2]}" + '\n' \
                      + f"test_f1       ={test_f1}" + '\n' \
                      + f"test_precision={test_precision}" + '\n' \
                      + f"test_recall   ={test_recall}" + '\n'

                with open(save_path, 'w') as f:
                    f.write(txt)

    except KeyboardInterrupt:
        f.close()
        sys.exit()


if __name__ == "__main__":

    # ==============================
    # == data manipulation + labeling
    # ==============================
    # create_copd_label_content(time_stamp=time_stamp, sep=',')
    # create_copd_label_edges(time_stamp=time_stamp, sep=',')
    # bine_copd_label(time_stamp=time_stamp)

    # ==========================
    # == cora dataset
    # ==========================
    # create_pytorch_dataset()

    # ========================
    # == copd dataset
    # ========================
    # copd = Copd(path='data/gene_disease/', data="copd_label", time_stamp="")
    copd = Copd(path=args.copd_path, data=args.copd_data, time_stamp=args.time_stamp)
    # copd.create_rep_dataset()

    # ======================
    # == running model
    # ======================
    # run_node2vec(copd=copd, time_stamp=time_stamp)

    # =========================
    # == run_GCN()
    # =========================
    # -- copd report
    # copd.edges2nodes_ratio(verbose=args.verbose)
    # copd.label_rate(verbose=args.verbose)
    '''
    rank 0: class = 2 has 8 number of members
    rank 1: class = 0 has 13 number of members
    rank 2: class = 3 has 22 number of members
    rank 3: class = 1 has 27 number of members
    rank 4: class = 4 has 31 number of members
    '''
    # copd.class_member_dist(plot=args.plot, verbose=args.verbose)
    # display2screen()
    # copd.rank_gene_overlap(verbose=args.verbose, plot=args.plot)

    # -- copd arguments
    # display2screen(args.add_features)
    if args.emb_name != "no_feat":
        x = add_features(args)
    else:
        #===================
        #== without features; instead use identity matrix of n*n where n is number of nodes
        #==================
        x = np.identity(len(copd.nodes2idx().keys()))
        x = torch.tensor(x, dtype=torch.float)

    # -- edge_index
    # tmp = np.unique(copd.edges)
    edge_index = list(map(copd.nodes2idx().get, copd.edges.T.flatten()))
    edge_index = torch.tensor(edge_index, dtype=torch.int64).view(2, -1)  # torch.Size([2, 4715])

    # -- label
    # label gene with 6
    y = [copd.disease2class()[i] if i in copd.disease2idx().values() else len(copd.class2disease().keys()) for i in
         copd.nodes2idx().values()]
    y = torch.tensor(y, dtype=torch.int64)  # torch.Size([2996])


    # -- Copd_geometric_dataset
    copd_geometric_dataset = geomertric_dataset(copd, x=x,edges_index=edge_index,y=y, split=args.split )

    # display2screen( x.shape, edge_index.shape)
    # display2screen(copd_geometric_dataset.edge_index.shape, edge_index.shape)


    param = {
            #Pseudo-Label
            'T1':int(args.t1_t2_alpha[0]),
            'T2':int(args.t1_t2_alpha[1]),
            'af':float(args.t1_t2_alpha[2])}

    # ====================
    # == run models
    # ====================

    if args.run_gcn:
        run_GCN(data=copd_geometric_dataset,emb_name=args.emb_name, time_stamp=args.time_stamp,tuning=args.tuning,
                 log=args.log,verbose=args.verbose, lr=args.lr,weight_decay=args.weight_decay,
                **param)
    if args.run_mlp:
        G = nx.Graph()

        # tmp = [(i,j) for (i,j) in zip(edge_index[0].numpy(), edge_index[1].numpy()) if int(i) == 2995 or int(j) == 2995 ]
        edges = [[i,j] if int(i) < len(copd.disease2idx().values()) else (j,i) for (i,j) in zip(edge_index[0].numpy(), edge_index[1].numpy())]
        edges = list(map(lambda t: (int(t[0]), int(t[1])), edges))

        edges = sorted(edges, reverse=False, key=lambda t:t[0])

        adj_list = create_adj_list(edges)
        # -- create genes as onehot
        onehot_genes =create_onehot(adj_list, edges)

        G.add_edges_from(edges)
        input = nx.adjacency_matrix(G).todense()[:len(copd.disease2idx().keys()),:]
        tmp = nx.adjacency_matrix(G).todense()[:len(copd.disease2idx().keys()),:]

        config = {
            "data": copd,
            "input": onehot_genes, # dictionary
            "label":y.numpy(), # tensor
            "train_mask":copd_geometric_dataset.train_mask,
            "test_mask":copd_geometric_dataset.test_mask,
            # change value of hidden_layers to be used in nn.sequential
            "hidden_layers":[2996, 2996, 128,16,len(copd.labels2idx().keys())],
            "epochs":200,
            "args": args,
            "param": param
        }
        run_mlp(config)
    if args.run_logist:
        G = nx.Graph()

        # tmp = [(i,j) for (i,j) in zip(edge_index[0].numpy(), edge_index[1].numpy()) if int(i) == 2995 or int(j) == 2995 ]
        edges = [[i, j] if int(i) < len(copd.disease2idx().values()) else (j, i) for (i, j) in
                 zip(edge_index[0].numpy(), edge_index[1].numpy())]
        edges = list(map(lambda t: (int(t[0]), int(t[1])), edges))

        edges = sorted(edges, reverse=False, key=lambda t: t[0])

        adj_list = create_adj_list(edges)
        # -- create genes as onehot
        onehot_genes = create_onehot(adj_list, edges)

        G.add_edges_from(edges)
        input = nx.adjacency_matrix(G).todense()[:len(copd.disease2idx().keys()), :]
        tmp = nx.adjacency_matrix(G).todense()[:len(copd.disease2idx().keys()), :]


        config = {
            "data": copd,
            "input": onehot_genes,  # dictionary
            "label": y.numpy(),
            "train_mask": copd_geometric_dataset.train_mask,
            "test_mask": copd_geometric_dataset.test_mask,
            "emb": x.numpy(),
            "args": args
        }

        run_logist(config, emb_name=args.emb_name)
    if args.run_gcn_on_disease_graph:
        G = nx.Graph()

        # tmp = [(i,j) for (i,j) in zip(edge_index[0].numpy(), edge_index[1].numpy()) if int(i) == 2995 or int(j) == 2995 ]
        edges = [[i, j] if int(i) < len(copd.disease2idx().values()) else (j, i) for (i, j) in
                 zip(edge_index[0].numpy(), edge_index[1].numpy())]
        edges = list(map(lambda t: (int(t[0]), int(t[1])), edges))

        edges = sorted(edges, reverse=False, key=lambda t: t[0])

        adj_list = create_adj_list(edges)
        # -- create genes as onehot
        onehot_genes = create_onehot(adj_list, edges)

        G.add_edges_from(edges)
        input = nx.adjacency_matrix(G).todense()[:len(copd.disease2idx().keys()), :]
        tmp = nx.adjacency_matrix(G).todense()[:len(copd.disease2idx().keys()), :]

        config = {
            "data": copd,
            "input": onehot_genes,  # dictionary
            "label": y.numpy(),
            "train_mask": copd_geometric_dataset.train_mask,
            "test_mask": copd_geometric_dataset.test_mask,
            "emb": x.numpy(),
            "hidden_layers": [2996, 2996, 128, 16, len(copd.labels2idx().keys())],
            "epochs": 200,
            "args": args,
            "param": param
        }
        run_gcn_on_disease_graph(config, emb_name=args.emb_name)
