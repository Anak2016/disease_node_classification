import sys,os
USER = os.environ['USERPROFILE']
sys.path.insert(1,f'{USER}\\PycharmProjects\\my_utility')

from utility_code.my_utility import *
from utility_code.python_lib_essential import *

from my_utils import *
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import plotting
import all_datasets

def plot_figures(config,save_path=None, file_name=None):
    file_name = file_name.split('.')[:-1]
    file_name = '/'.join(file_name) + ".png"
    print(f"save plot to {save_path}{file_name}...")
    num_fig = len(config.keys())

    if num_fig <= 3 :
        num_col = num_fig
        num_row = 1
    elif num_fig/3 == int(num_fig/3):
        num_col = 3
        num_row = int(num_fig/3)
    else:
        num_col = 3
        num_row = int(num_fig/3) + 1

    assert num_row * num_col >= num_fig, "num_row * num_col must be more than or equal to num_fig"

    fig, axes = plt.subplots(num_row,num_col)

    # num_fig = len(config.keys())
    for i, (c,v) in enumerate(config.items()):
        x_label = v['x_label']
        y_label = v['y_label']
        title = c
        x_lim = v.get('x_lim', None)
        y_lim = v.get('y_lim', None)

        col = i % num_col
        row = int(i / num_col)

        '''there will be 2 sections
        1. config keywords already have to be provided
        2. config keywords may or may not be provided
        
            there are 2 types of config keywords  
        1. config keywords with eithre args or kwargs
        2. config keywords that does not have args or kwargs 
            eg x_lim 
        '''
        if num_row == 1 and num_col ==1:
            plot_args = v.get('plot', [{'args': []}])
            plot_kwargs = v.get('plot', [{'kwargs': []}])
            legend_kwargs = v.get('legend', [{'kwargs': []}])

            for i, j, k in zip(plot_args, plot_kwargs, legend_kwargs):
                axes.plot(*i.get('args',[]), **j.get('kwargs',{}))
                axes.legend(**k.get('kwargs',{}))

            if x_lim is not None:
                axes.set_xlim(*x_lim)
            if y_lim is not None:
                axes.set_ylim(*y_lim)
            axes.set_xlabel(x_label)
            axes.set_ylabel(y_label)
            axes.set_title(title)
        elif num_row == 1 and num_col != 1:
            plot_args = v.get('plot', [{'args': []}])
            plot_kwargs = v.get('plot', [{'kwargs': []}])
            legend_kwargs = v.get('legend', [{'kwargs': []}])

            for i, j, k in zip(plot_args, plot_kwargs, legend_kwargs):
                axes[col].plot(*i.get('args',[]), **j.get('kwargs',{}))
                axes[col].legend(**k.get('kwargs',{}))

            if x_lim is not None:
                axes[col].set_xlim(*x_lim)
            if y_lim is not None:
                axes[col].set_ylim(*y_lim)
            axes[col].set_xlabel(x_label)
            axes[col].set_ylabel(y_label)
            axes[col].set_title(title)
        else:
            plot_args = v.get('plot', [{'args': []}])
            plot_kwargs = v.get('plot', [{'kwargs': []}])
            legend_kwargs = v.get('legend', [{'kwargs': []}])

            for i, j, k in zip(plot_args, plot_kwargs, legend_kwargs):
                axes[row, col].plot(*i.get('args',[]), **j.get('kwargs',{}))
                axes[row, col].legend(**k.get('kwargs',{}))

            if x_lim is not None:
                axes[row, col].set_xlim(x_lim)
            if y_lim is not None:
                axes[row, col].set_ylim(y_lim)
            axes[row, col].set_xlabel(x_label)
            axes[row, col].set_ylabel(y_label)
            axes[row, col].set_title(title)

    os.makedirs(f'{save_path}', exist_ok=True)

    if save_path is not None:
        print(f"writing to {save_path}{file_name}")
        plt.savefig(f'{save_path}{file_name}')
    plt.show()


def plot_2d(copd, path, file_list, with_gene=True, **kwargs):
    '''outdated:
        this need to be merged with plot_figures
    '''
    print("=======================")
    print("plotting 2d emb...")
    x = None
    emb = None
    nodes = None # list of all nodes_int representation range = 0-2900-ish

    s = time.time()
    if isinstance(file_list, list):
        # saving image to path is not yet implemented in here
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
                    condition = True if copd.labels2class()[n] in class2nodes.keys() else False
                except:
                    condition = True if len(copd.labels2class().keys()) in class2nodes.keys() else False

                if condition:
                    if n in copd.labels2class().keys():
                        class2nodes[copd.labels2class()[n]].append([n, emb[i]])
                    else:
                        class2nodes[len(copd.labels2class().keys())].append([n, emb[i]])
                else:
                    if n in copd.labels2class().keys():
                        class2nodes[copd.labels2class()[n]] = [[n, emb[i]]]
                    else:
                        class2nodes[len(copd.labels2class().keys())] = [[n, emb[i]]]

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

        # read noe emb from file
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
                x = pd.DataFrame([i.split('\t') for i in x]) # just to make all of the dtype have type = DataFrame.

                nodes = x.iloc[1:, 0].to_numpy()
                nodes = [int(float(i)) for i in nodes] #This line garantee that value is int
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
            included_nodes = len(copd.labels2class().keys()) - 1

        # -- label nodes in
        for i, n in enumerate(nodes):

            if n <= included_nodes:
                try:
                    condition = True if copd.labels2class()[n] in class2nodes.keys() else False
                except:
                    condition = True if len(copd.labels2class().keys()) in class2nodes.keys() else False

                if condition:
                    if n in copd.labels2class().keys():
                        class2nodes[copd.labels2class()[n]].append([n, emb[i]])
                    else:
                        class2nodes[len(copd.labels2class().keys())].append([n, emb[i]])
                else:
                    if n in copd.labels2class().keys():
                        class2nodes[copd.labels2class()[n]] = [[n, emb[i]]]
                    else:
                        class2nodes[len(copd.labels2class().keys())] = [[n, emb[i]]]

        # emb.shape = (# of class, # of node emb in each class)

        emb_reordered = np.array([np.array([tuple[1] for tuple in class2nodes[k]]) for k in class2nodes.keys() ])
        # display2screen(em_reoredered) # sorted by disease and only plot disease

        if kwargs['pred_label'] is not None:
            # pred_label is expected to have the same order as emb that is read from path+file
            pred_label = kwargs['pred_label'].to_numpy().flatten().tolist()

            if with_gene is False:
                pred_label = pred_label[:included_nodes]
                emb = emb[:included_nodes]

            # todo here>> legends is not correct and figure are not in the same plot
            plt.figure(1)
            plt.subplot(121)
            # for n, p in zip(emb, pred_label):
            # group node up with its label
            label_node = {l:[] for l in set(pred_label)}
            for i,l in enumerate(pred_label):
                label_node[l].append(i)

            for l,n in label_node.items():
                plt.scatter(emb[n,0], emb[n,1], label=l)
            # plt.scatter(emb[:,0], emb[:,1], c=pred_label)
            plt.legend()

            plt.subplot(122)
            for emb_arr, label in zip(emb_reordered, class2nodes.keys()):
                plt.scatter(emb_arr[:, 0], emb_arr[:, 1], label=label)
            plt.legend()
        else:
            for emb_arr, label in zip(emb_reordered, class2nodes.keys()):
                plt.scatter(emb_arr[:, 0], emb_arr[:, 1], label=label)
            plt.legend()

    f = time.time()
    total = f-s
    print(f'total running time {total}')

    if kwargs['log'] is True:
        print(f"writing to {path + kwargs['save_img']}...")
        plt.savefig(path + kwargs["save_img"])

    plt.show()

if __name__ == "__main__":
    # =======================
    # ==initialize path and file
    # =======================
    path = r"C:/Users/awannaphasch2016/PycharmProjects/disease_node_classification/data/{args.time_stamp}/gene_disease/processed/embedding/"
    # time_stamp = ""
    time_stamp = '07_14_19_46'

    copd = all_datasets.Copd(path=f'data/gene_disease/{args.time_stamp}/raw/', data="copd_label", time_stamp=time_stamp)
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

    plotting.plot_2d(copd, path, file, emb=emb , func="tsne", with_gene=with_gene)
    # plotting.plot_2d(copd, path, file, emb=emb, func="pca", with_gene=with_gene)
