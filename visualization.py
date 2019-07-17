import pandas as pd
import matplotlib.pyplot as plt
import time

from my_utils import *
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def plot_2d(copd, path, file_list, **kwargs):
    x = None
    emb = None
    nodes = None # list of all nodes_int representation range = 0-2900-ish


    s = time.time()
    if isinstance(file_list, list):

        for file in file_list:
            if kwargs.get('sep'): # sep ='\t' is not right
                x = pd.read_csv(path + file, sep=kwargs.get('sep'), header=None)

            if kwargs.get('dtype'):
                if kwargs.get('dtype') == 'bine':
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

        if kwargs.get('dtype'):
            if kwargs.get('dtype') == 'attention_walk':
                if kwargs.get('sep'):  # sep ='\t' is not right
                    x = pd.read_csv(path + file, sep=kwargs.get('sep'), header=None)
                nodes = x.iloc[1:, 0].to_numpy()
                nodes = [int(float(i)) for i in nodes]
                x = x.iloc[1:,1:].to_numpy()

            if kwargs.get('dtype') == "node2vec":
                with open(path+file, 'r') as f:
                    x = f.readlines()
                x = x[1:]
                x = pd.DataFrame([i.split(' ') for i in x]) # just to make all of the dtype have type = DataFrame.

                nodes = x.iloc[1:, 0].to_numpy()
                nodes = [int(i) for i in nodes] #This line garantee that value is int
                x = x.iloc[:,1:].to_numpy()

        if kwargs.get('func'):
            if kwargs.get('func') == 'tsne':
                emb = TSNE(n_components=2).fit_transform(x)
            if kwargs.get('func') == 'pca':
                emb = PCA(n_components=2).fit_transform(x)
            # apply dimensional reduction to 2 D

        # -- labels nodes in order
        class2nodes = {}

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
        emb = np.array([np.array([tuple[1] for tuple in class2nodes[k]]) for k in class2nodes.keys() ])

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
    # --attention_walk
    # file = "attention_walk/attentionwalk_emb.txt"
    # plot_2d(copd, path, file, sep=',', dtype='attention_walk', func="tsne")
    # plot_2d(copd, path, file, sep=',', dtype='attention_walk', func="pca")
    # exit()

    # --bine
    # file = ["bine/vectors_u.dat", "bine/vectors_v.dat"]
    # plot_2d(copd, path, file, sep=' ', dtype="bine", func="tsne")
    # plot_2d(copd, path, file, sep=' ', dtype="bine", func="pca")
    # exit()

    # -- node2vec
    file = f"node2vec/node2vec_emb_subgraph{time_stamp}.txt"
    plot_2d(copd, path, file, sep=' ', dtype="node2vec", func="tsne")
    # plot_2d(copd, path, file, sep=' ', dtype="node2vec", func="pca")
    exit()