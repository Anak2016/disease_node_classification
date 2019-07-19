import pandas as pd
import matplotlib.pyplot as plt
import time

from my_utils import *
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def plot_2d(copd, path, file_list, with_gene=True, **kwargs):
    print("=======================")
    print("plotting 2d emb...")
    x = None
    emb = None
    nodes = None # list of all nodes_int representation range = 0-2900-ish

    s = time.time()
    if isinstance(file_list, list):

        for file in file_list:
            x = pd.read_csv(path + file, sep=' ', header=None)

            if kwargs.get('emb'):
                if kwargs.get('emb') == 'bine':
                    nodes = x.iloc[:, 0].to_numpy()
                    nodes = np.array([int(i[1:])for i in nodes])

                    x = x.iloc[:,1:-1].to_numpy() # value of last col is NA

            # apply dimensional reduction to 2 D
            if kwargs.get('func'):
                if kwargs.get('func') == 'tsne':
                    emb = TSNE(n_components=2).fit_transform(x)
                if kwargs.get('func') == 'pca':
                    emb = PCA(n_components=2).fit_transform(x)

            # -- labels nodes in order
            class2nodes = {}
            # display2screen(len(copd.disease2class().keys()))

            # for i, n in enumerate(nodes[:61]):
            for i, n in enumerate(nodes):
                try:
                    condition = True if copd.disease2class()[n] in class2nodes.keys() else False
                except:
                    condition = True if len(copd.disease2class().keys()) in class2nodes.keys() else False

                if condition:
                    if n in copd.disease2class().keys():
                        class2nodes[copd.disease2class()[n]].append([n, emb[i]])
                    else:
                        class2nodes[len(copd.disease2class().keys())].append([n, emb[i]])
                else:
                    if n in copd.disease2class().keys():
                        class2nodes[copd.disease2class()[n]] = [[n, emb[i]]]
                    else:
                        class2nodes[len(copd.disease2class().keys())] = [[n, emb[i]]]

            # emb.shape = (# of keys, -1)
            emb = np.array([np.array([tuple[1] for tuple in class2nodes[k]]) for k in class2nodes.keys()])

            for emb_arr, label in zip(emb, class2nodes.keys()):
                plt.scatter(emb_arr[:, 0], emb_arr[:, 1], label=label)
            plt.legend()

            f = time.time()
            total = f - s
            print(f'running time {total}')
        plt.show()

    else:
        file = file_list

        if kwargs.get('emb'):
            if kwargs.get('emb') == 'attentionwalk':
                x = pd.read_csv(path + file, sep=',', header=None)
                nodes = x.iloc[1:, 0].to_numpy()
                nodes = [int(float(i)) for i in nodes]
                x = x.iloc[1:,1:].to_numpy()

            if kwargs.get('emb') == "node2vec":
                with open(path+file, 'r') as f:
                    x = f.readlines()
                x = x[1:]
                x = pd.DataFrame([i.split(' ') for i in x]) # just to make all of the dtype have type = DataFrame.

                nodes = x.iloc[1:, 0].to_numpy()
                nodes = [int(i) for i in nodes] #This line garantee that value is int
                x = x.iloc[:,1:].to_numpy()

            if kwargs.get('emb') == 'gcn':
                with open(path+file, 'r') as f:
                    x = f.readlines()
                x = x[1:]
                x = pd.DataFrame([i.split(' ') for i in x]) # just to make all of the dtype have type = DataFrame.

                nodes = x.iloc[1:, 0].to_numpy()
                nodes = [int(i) for i in nodes] #This line garantee that value is int
                x = x.iloc[:,1:].to_numpy()

            if kwargs.get('emb') == 'bine':
                x = pd.read_csv(path + file, sep=' ', header=None)
                nodes = x.iloc[:, 0].to_numpy()
                nodes = np.array([int(i[1:]) for i in nodes])

                x = x.iloc[:, 1:-1].to_numpy()  # value of last col is NA

        if kwargs.get('func'):
            if kwargs.get('func') == 'tsne':
                # display2screen(x.shape)
                emb = TSNE(n_components=2).fit_transform(x)
            if kwargs.get('func') == 'pca':
                emb = PCA(n_components=2).fit_transform(x)
            # apply dimensional reduction to 2 D

        # -- labels nodes in order
        class2nodes = {}

        # -- included_nodes is all nodes or just diesease nodes?
        if with_gene:
            included_nodes = max(nodes)
        else:
            included_nodes = len(copd.disease2idx().keys()) - 1

        # -- label nodes in
        for i, n in enumerate(nodes):

            if n <= included_nodes:
                try:
                    condition = True if copd.disease2class()[n] in class2nodes.keys() else False
                except:
                    condition = True if len(copd.disease2class().keys()) in class2nodes.keys() else False

                if condition:
                    if n in copd.disease2class().keys():
                        class2nodes[copd.disease2class()[n]].append([n, emb[i]])
                    else:
                        class2nodes[len(copd.disease2class().keys())].append([n, emb[i]])
                else:
                    if n in copd.disease2class().keys():
                        class2nodes[copd.disease2class()[n]] = [[n, emb[i]]]
                    else:
                        class2nodes[len(copd.disease2class().keys())] = [[n, emb[i]]]

        # emb.shape = (# of keys, -1)
        emb = np.array([np.array([tuple[1] for tuple in class2nodes[k]]) for k in class2nodes.keys() ])
        # display2screen(emb) # sorted by disease and only plot disease

        for emb_arr, label in zip(emb, class2nodes.keys()):
            plt.scatter(emb_arr[:, 0], emb_arr[:, 1], label=label)
        plt.legend()

    f = time.time()
    total = f-s
    print(f'total running time {total}')
    plt.show()



if __name__ == "__main__":
    # =======================
    # ==initialize path and file
    # =======================
    path = r"C:/Users/awannaphasch2016/PycharmProjects/disease_node_classification/output/gene_disease/embedding/"
    # time_stamp = ""
    time_stamp = '07_14_19_46'

    copd = Copd(path='data/gene_disease/', data="copd_label", time_stamp=time_stamp)
    # =========================
    # == plot grpah
    # =========================

    # emb = 'attentionwalk'
    # file = f"{emb}/{emb}_emb{time_stamp}.txt"

    # emb = 'node2vec'
    # file = f"{emb}/{emb}_emb_fullgraph{time_stamp}.txt"
    # file = f"{emb}/{emb}_emb_subgraph{time_stamp}.txt"

    emb = 'bine'
    file = [f"{emb}/vectors_u{time_stamp}.dat", f"{emb}/vectors_v{time_stamp}.dat"]

    with_gene = True

    plot_2d(copd, path, file, emb=emb , func="tsne", with_gene=with_gene)
    # plot_2d(copd, path, file, emb=emb, func="pca", with_gene=with_gene)
