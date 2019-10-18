from __future__ import print_function, division
import sys,os

USER = os.environ['USERPROFILE']
sys.path.insert(1,f'{USER}\\PycharmProjects\\my_utility')
from utility_code.my_utility import *

#--------original import of anak
from skimage import io,transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# use node2vec with cora datset
import torch.nn.functional as F

import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
import time
import pandas as pd
import os
import random
import sys

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from datetime import datetime
from torch_geometric.nn import GCNConv, GATConv, SAGEConv  # noqa
from node2vec import Node2Vec
from networkx.algorithms import bipartite

# from my_utils import create_adj_list, run_gcn_on_disease_graph
from arg_parser import args
from all_models import baseline
import preprocessing
import plotting
import all_datasets
import my_utils
# from plotting import plotting.plot_2d


# display2screen()
# import sys
# sys.path.insert(0, r'C:\Users\Anak\PycharmProjects\AttentionWalk')
# from src.attentionwalk import AttentionWalkLayer

def pause():
    print("done")
    exit()

def save_node2vec_emb(G, save_path = f'data/gene_disease/{args.time_stamp}/processed/embedding/node2vec/', EMBEDDING_FILENAME = 'node2vec_emb.txt', log=True):
    #TODO here>>
    with open(save_path + EMBEDDING_FILENAME, 'w') as f:
        print(f"save node2vec emb to {save_path + EMBEDDING_FILENAME}")

    # Precompute probabilities and generate walks - **ON WINDOWS ONLY WORKS WITH workers=1**
    node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200,
                        workers=4)  # Use temp_folder for big graphs # todo undirected_edges
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
        with open(f'./log/gene_disease/{EMBEDDING_FILENAME}', 'w') as f:
            f.write(f' --{save_path}{EMBEDDING_FILENAME}\n')
            f.write(f'total running time {total}')



def nx_plot(G, pos=None, node_color=None ):

    if pos is not None and node_color is not None:
        nx.draw(G, pos=pos, node_color=node_color)
        plt.show()
    else:
        nx.draw(G)
        plt.show()

def run_node2vec(copd, copd_geometric, time_stamp=args.time_stamp):
    '''

    :return:
    '''
    # todo check if this is correct
    if args.common_nodes_feat == 'gene':
        # adj = np.zeros((len(list(copd.nodes2idx().keys())), (len(list(copd.nodes2idx().keys())))))
        # dataset, geometric_dataset, plot_shared_gene_dist = False, used_nodes = 'all', edges_weight_option = 'jaccard'):
        # all_x_input = preprocessing.create_common_nodes_as_features(copd, copd_geometric, edges_weight_option=args.edges_weight_option,plot_shared_gene_dist=True)
        # adj[:all_x_input.shape[0], :all_x_input.shape[1]] = adj[:all_x_input.shape[0], :all_x_input.shape[1]] + all_x_input
        # adj = np.asmatrix(adj)
        # graph = nx.from_numpy_matrix(adj)
        # subgraph = my_utils.get_subgraph_disconnected(graph)[0]
        weighted_adj, edges_weight, edges = preprocessing.create_common_nodes_as_features(copd, copd_geometric, used_nodes='gene', edges_weight_option=args.edges_weight_option,plot_shared_gene_dist=False)
        graph = nx.from_numpy_matrix(weighted_adj)
        subgraph = my_utils.get_subgraph_disconnected(graph)[0]

        # graph1 = copd_geometric.graph
        # display2screen(list(map(len, [graph1.edges, graph.edges, graph1.nodes, graph1.nodes])))
    elif args.common_nodes_feat != 'disease':
        assert ValueError('not yet support')
    elif args.common_nodes_feat != 'all':
        assert ValueError('not yet support')
    else:
        assert ValueError('not yet support')

        subgraph = copd_geometric.subgraph
        graph = copd_geometric.graph

        # G = nx.from_pandas_edgelist(edges, 'geneid', 'diseaseid')

    # --color bipartite graph
    if args.plot_emb:
        left, right = bipartite.sets(copd_geometric.subgraph)
        bipartite_color = [0 if i < len(left) else 1 for i, _ in enumerate(left.union(right))]
        pos = nx.circular_layout(copd_geometric.subgraph)
        nx_plot(copd_geometric.subgraph, pos=pos, node_color=bipartite_color)

    #  -- save node2vec embbedding to file
    # display2screen(len(g.nodes)) #2975
    # save_node2vec_emb(subgraph,EMBEDDING_FILENAME=f"node2vec_emb_subgraph_common_nodes_feat={args.common_nodes_feat}{time_stamp}.txt" )

    # display2screen(len(G.nodes)) #2996
    # def save_node2vec_emb(G, save_path=f'data/gene_disease/{args.time_stamp}/processed/embedding/node2vec/',
    #                       EMBEDDING_FILENAME='node2vec_emb.txt', log=True):
    # if args.self_loop is True:

    save_node2vec_emb(graph,
                      EMBEDDING_FILENAME=f"node2vec_emb_fullgraph_common_nodes_feat={args.common_nodes_feat}{time_stamp}_added_edges=disease_{args.edges_weight_option}_top_k={args.top_percent_edges}_mask={args.mask_edges}_stoch1.txt")
    #     if args.stochastic_edges:
    #         save_node2vec_emb(graph,EMBEDDING_FILENAME=f"node2vec_emb_fullgraph_common_nodes_feat={args.common_nodes_feat}{time_stamp}_added_edges=disease_{args.edges_weight_option}_top_k={args.top_percent_edges}_mask={args.mask_edges}_stoch_selfloop.txt" )
    #     else:
    #         save_node2vec_emb(graph,EMBEDDING_FILENAME=f"node2vec_emb_fullgraph_common_nodes_feat={args.common_nodes_feat}{time_stamp}_added_edges=disease_{args.edges_weight_option}_top_k={args.top_percent_edges}_mask={args.mask_edges}_selfloop.txt")
    # else:
    #     if args.stochastic_edges:
    #         save_node2vec_emb(graph,EMBEDDING_FILENAME=f"node2vec_emb_fullgraph_common_nodes_feat={args.common_nodes_feat}{time_stamp}_added_edges=disease_{args.edges_weight_option}_top_k={args.top_percent_edges}_mask={args.mask_edges}_stoch1.txt" )
    #     else:
    #         save_node2vec_emb(graph,EMBEDDING_FILENAME=f"node2vec_emb_fullgraph_common_nodes_feat={args.common_nodes_feat}{time_stamp}_added_edges=disease_{args.edges_weight_option}_top_k={args.top_percent_edges}_mask={args.mask_edges}.txt")

def bine_copd_label(time_stamp=''):
    # load data in to dataframe
    # add u to gene and add i to item
    # create new columns of weight = 1 ( unweightd)

    # file = f"data/{args.time_stamp}/gene_disease/raw/copd_label_edges{time_stamp}.txt"

    file = f"data/{args.time_stamp}/gene_disease/processed/rep/rep_copd_label_edges{time_stamp}.txt"

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
    save_path = f'data/gene_disease/{args.time_stamp}/processed/embedding/bine/rep/cope_label_edges{time_stamp}_.txt'
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

def create_pytorch_dataset(path=f'data/gene_disease/{args.time_stamp}/raw/', files=['copd_label_content','copd_label_edges']):
    '''convert copd_label to pytorch_dataset'''
    import warnings
    warnings.filterwarnings("ignore") # what does thi sdo?
    plt.ion() #set ineteractive mode
    import pandas as pd
    landmarks_frame = pd.read_csv('data/raw/faces//face_landmarks.csv')
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

    # show_landmarks(io.imread(os.path.join('data/raw/faces//', img_name)), landmarks)


    face_dataset = FaceLandmarksDataset(csv_file="data/raw/faces//face_landmarks.csv",root_dir='data/raw/faces//')

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
    transformed_dataset = FaceLandmarksDataset(csv_file='data/raw/faces//face_landmarks.csv',
                                               root_dir='data/raw/faces//',
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




