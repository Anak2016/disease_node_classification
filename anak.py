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
import collections


from torch_geometric.nn import GCNConv, ChebConv  # noqa
from node2vec import Node2Vec
from networkx.algorithms import bipartite
from my_utils import Cora, Copd, get_subgraph_disconnected, GetData, Conversion, create_copd_label_content, create_copd_label_edges, display2screen
from sys import path

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
    if param.plot:
        left, right = bipartite.sets(g)
        bipartite_color = [0 if i < len(left) else 1 for i, _ in enumerate(left.union(right))]
        pos = nx.circular_layout(g)
        nx_plot(g, pos=pos, node_color=bipartite_color)

    #  -- save node2vec embbedding to file
    # display2screen(len(g.nodes)) #2975
    save_node2vec_emb(g,EMBEDDING_FILENAME=f"node2vec_emb_subgraph{time_stamp}.txt" )
    # display2screen(len(G.nodes)) #2996
    save_node2vec_emb(G,EMBEDDING_FILENAME=f"node2vec_emb_fullgraph{time_stamp}.txt" )

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
            plt.ioff()
            plt.show()
            break

# read https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html
# use GCN in scatch_paper.py as a template to build new GCN for COPD
class Copd_geomertric_dataset(Data):

    def __init__(self, data, x=None, edges_index=None, edge_attr=None, y=None):
        self.dataset = data
        super(Copd_geomertric_dataset, self).__init__(x,edges_index,edge_attr,y)

        # -- masking
        self.train_mask_set = []
        self.test_mask_set = []
        ind = 0  # inc everytime nodes are added; check how many nodes are included in training set
        count = 0
        arr_ind = 0 # inc everytimes for loop go through all of the classes; it represent current ind of val_list in each class.

        while True:
            # for i,k in enumerate(copd.disease2class().keys()):

            max_class_int_rep = self.num_classes - 1 # max int_rep of all classes
            current_class = count % max_class_int_rep

            if ind < int(0.8 * len(copd.disease2class().keys())):  # training set
                next_val = set(copd.class2disease()[current_class]).difference(set(self.train_mask_set))
                if len(next_val) > 1:
                    next_val = list(next_val)[0]
                    # self.train_mask_set.append(copd.class2disease()[current_class][arr_ind])
                    self.train_mask_set.append(next_val)
                    ind += 1

            if ind == int(0.8 * len(copd.disease2class().keys())):
                break

            count += 1

        # display2screen(ind, count)
        self.test_mask_set = list(set([i for i in copd.disease2idx().values()]).difference(self.train_mask_set))

        train_class = set([copd.disease2class()[i] for i in self.train_mask_set])
        test_class  = set([copd.disease2class()[i] for i in self.test_mask_set])

        # display2screen(train_class, test_class, train_class.symmetric_difference(test_class))
        assert len(self.test_mask_set) + len(self.train_mask_set) == len(copd.disease2class().keys()), "Some diseases are not included in neither training or test set "
        assert len(set(self.train_mask_set).intersection(set(self.test_mask_set))) == 0, "diseases in both classes must be unique to its dataset either trianing or test dataset"
        assert len(set([copd.disease2class()[i] for i in self.train_mask_set])) == len(copd.class2disease().keys()), f"members of training set does not include all of the class labels.\n classes={train_class}"
        assert len(set([copd.disease2class()[i] for i in self.test_mask_set])) == len(copd.class2disease().keys()), f"members of test set does not include all of the class labels.\n classes={test_class}"

        # display2screen(self.test_mask_set, self.train_mask_set)
        # display2screen(len(self.test_mask_set), len(self.train_mask_set))

        # -- convert to torch.tensor
        self.train_mask_set = torch.LongTensor(self.train_mask_set)
        self.test_mask_set = torch.LongTensor(self.test_mask_set)

    @property
    def num_classes(self):
        return np.unique(y.numpy()).shape[0]

    # -- inherited property
    # @num_node_features.setter
    # def num_features(self):

    # -- masking index for x and y
    @property
    def train_mask(self):
        # make sure that all train set ahve all the classes
        return self.train_mask_set

    @property
    def test_mask(self):
        # make sure that all test set ahve all the classes
        return self.test_mask_set

def run_GCN(data = None):

    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()

            self.conv1 = GCNConv(data.num_features, 16, cached=True)
            self.conv2 = GCNConv(16 , data.num_classes, cached=True)

        def forward(self):
            x, edge_index = data.x, data.edge_index
            x = x.type(torch.float)
            edge_index = edge_index.type(torch.long)
            # edge_index = torch.transpose(edge_index, 0,1)

            x = F.relu(self.conv1(x, edge_index))
            x = F.dropout(x, training=self.training)
            x = self.conv2(x, edge_index)
            return F.log_softmax(x, dim=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model= Net().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    # model() -> optimizer -> loss -> model.train()-> optimizer.zero_grad() -> loss.backward() -> optimizer.step() -> next epoch

    def train():
        model.train()
        optimizer.zero_grad()
        # todo create correct train_mask to be used
        F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
        optimizer.step()

    def test():
        model.eval()
        logits, accs = model(), []

        # for _,mask in data('train_mask', 'test_mask'):
        for mask in [data.train_mask, data.test_mask]:
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item()/ mask.sum().item()
            accs.append(acc)
        return accs

    best_val_acc =test_acc = 0
    for epoch in range(1,201):
        train()
        train_acc, test_acc = test()
        log = 'Epoch: {:03d}, Train: {:.4f}, Test: {:.4f}'
        print(log.format(epoch, train_acc, test_acc))


if __name__ == "__main__":
    # --initalization
    time_stamp = '07_14_19_46'


    # -- data manipulation + labeling
    # create_copd_label_content(time_stamp=time_stamp, sep=',')
    # create_copd_label_edges(time_stamp=time_stamp, sep=',')
    # bine_copd_label(time_stamp=time_stamp)

    # -- cora dataset
    # create_pytorch_dataset()

    # -- copd dataset
    # copd = Copd(path='data/gene_disease/', data="copd_label", time_stamp="")
    copd = Copd(path='data/gene_disease/', data="copd_label", time_stamp=time_stamp)
    # copd.create_rep_dataset()

    verbose= True
    plot = True
    # -- copd report
    # copd.edges2nodes_ratio(verbose=verbose)
    # copd.label_rate(verbose=verbose)
    # copd.class_member_dist(plot=plot, verbose=verbose)
    # copd.rank_gene_overlap(verbose=verbose, plot=plot)

    # -- running model
    # run_node2vec(copd=copd, time_stamp=time_stamp)

    # 2996, 101, 2895
    # display2screen(len(list(copd.nodes2idx().values())), len(copd.disease2class().keys()), len(copd.genes2idx()))

    # ==============================
    # ==pre arguments to be fed to Copd_geometric_dataset
    # ==============================
    # -- add node embedding to x
    emb_path = f'output/gene_disease/embedding/node2vec/node2vec_emb_fullgraph{time_stamp}.txt'
    with open(emb_path,'r') as f:
        tmp = f.readlines()
        tmp = tmp[1:]
    emb_dict = {int(i.split(' ')[0]):list(map(float,i.split(' ')[1:]))  for i in tmp }
    emb = sorted(emb_dict.items(), key= lambda t:t[0])

    x = np.array([[j for j in i[1]] for i in emb ], dtype=np.float)
    x = torch.tensor(x, dtype=torch.float) # torch.Size([2996, 64])

    # -- edge_index
    edge_index = list(map(copd.nodes2idx().get, copd.edges.T.flatten()))
    edge_index = torch.tensor(edge_index, dtype=torch.int64).view(2,-1) # torch.Size([2, 4715])

    # -- label
    # label gene with 99
    y = [copd.disease2class()[i] if i in copd.disease2idx().values() else 99 for i in copd.nodes2idx().values()]
    y = torch.tensor(y, dtype=torch.int64) # torch.Size([2996])
    # display2screen(y.shape)

    copd_geometric_dataset = Copd_geomertric_dataset(copd, x=x,edges_index=edge_index,y=y)
    run_GCN(copd_geometric_dataset)
