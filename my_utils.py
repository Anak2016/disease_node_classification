import numpy as np
import scipy.sparse as sp
import torch
import networkx as nx
import pandas as pd
import collections
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from scipy.sparse import csr_matrix
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
    # display2screen(np_.shape)

    save_file = f"{file_name}_content{time_stamp}.txt"
    # write2files(df,path=path,file_name=save_file,type='df')

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
    # display2screen(np.unique(np_).shape)

    save_file = f"{file_name}_edges{time_stamp}.txt"
    # write2files(df,path=path,file_name=save_file,type='df')


def write2files(data,path="./data/gene_disease/", file_name=None, type='df'):
    '''

    :parame data: content to be written in files
    :param path:
    :param dataset:
    :param type: type of content arg;  df, np, dict.
    :return:
    '''
    print(f'write to {path+file_name}...')

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
def create_adj_list(edges):
    '''

    :param edges:
        [(disease1, gene1), (disease2,gene2.....] where disease are sorted in ascending order
        eg. [(0,106),(1,400),(1,203),... ]

    :return:adj_list
            adj_list has the follwoing format must be in the followin format:

        graph = {source_node1: [{target_node: weight}, ... ]
                source_node2: [{target_node: weight}, ... ]
                ,....,}

    '''
    adj_list = {i:[] for (i,j) in edges}
    for disease, gene in edges:
        adj_list[disease].append({gene:'1'})

    return adj_list

def create_onehot(adj_list, edges):
    '''
    param: adj_list
        adj_list must be in the followin format:
        adj_list = {source_node1: [{target_node: weight}, ... ]
                source_node2: [{target_node: weight}, ... ]
                ,....,}

    :param edges:
        [(disease1, gene1), (disease2,gene2.....] where disease are sorted in ascending order
        eg. [(0,106),(1,400),(1,203),... ]

    :return: onehot is in the following format
        assuming the followin adj_list is provided as input

            1: [{1: '1'}, {3: '1'}]
        onehot will be
            disease_idx: [[0,1,0,0,0,0], [0,0,0,1,0,0]]
            disease_1 has 2 one hot vector
                :one hot at index 1 and one hot at index 3

    '''
    # -- built-in max() give the wrong result
    # max1, max2 = max(edges)[0], max(edges)[1]
    # max_idx = max([max1,max2]) # 2955
    max_idx = np.amax(np.array(edges).flatten()) # 2995
    identity_matrix = np.identity(max_idx + 1)

    onehot = {i:[] for i in adj_list.keys()}
    for key, val in adj_list.items():
        # print(int(key))
        onehot[int(key)] = np.asarray([identity_matrix[int(list(k.keys())[0]),:] for k in val])
    # display2screen(onehot[0].shape)

    return onehot

# =======================
# == BaseLine
# =======================

# -- logistic regression with node embedding
def run_logist(config, emb_name):
    '''
    run logistic regression

    :param config:
    :param use_emb: use node embedding in logistic regression
    :return:
    '''
    copd = config["data"]
    input = config["input"]
    y = config['label']
    train_mask = config['train_mask']
    test_mask = config['test_mask']
    emb = config['emb']
    args = config['args']


    # -- initialization
    train_label = y[train_mask]
    test_label  = y[test_mask]
    train_input = []
    test_input = []

    if emb_name != 'no_feat':
        train_input = emb[train_mask]
        test_input = emb[test_mask]
    else:
        '''
            convert input(aka onehot_genes) into the following 
                given disease 1 has 2 onehot_vector at position 1, 3 with dimention = 5
                disease 1 will vector = [0,1,0,1,0] 
        '''
        for key, val in input.items():
            sum = 0
            if int(key) in train_mask:
                for v in val:
                    sum = np.add(sum,v)
                input[key] = sum
                train_input.append(input[key])
            sum1 = 0
            if int(key) in test_mask:
                for v in val:
                    sum1 = np.add(sum1,v)
                input[key] = sum1
                test_input.append(input[key])

    train_input = normalize_features(csr_matrix(np.array(train_input)))
    test_input = normalize_features(csr_matrix(np.array(test_input)))

    # -- convert to tensor
    train_input = torch.tensor(train_input, dtype=torch.float ).numpy()
    test_input  = torch.tensor(test_input, dtype=torch.float ).numpy()
    train_label = torch.tensor(train_label, dtype=torch.long ).numpy()
    test_label  = torch.tensor(test_label, dtype=torch.long ).numpy()

    from sklearn import metrics
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(solver = 'lbfgs')
    model.fit(train_input, train_label)

    y_pred_train = model.predict(train_input)
    y_pred_test = model.predict(test_input)



    log_list = []
    # -- metrix results
    cm_train = confusion_matrix(y_pred_train, train_label)
    cm_train = np.array2string(cm_train)
    count_misclassified = (train_label != y_pred_train).sum()
    accuracy = metrics.accuracy_score(train_label, y_pred_train)


    txt = ["For training data", 'Misclassified samples: {}'.format(count_misclassified), 'Accuracy: {:.2f}'.format(accuracy)]
    log_list.append('\n'.join(txt))
    print(log_list[-1])

    # -- metrix results
    cm_test = confusion_matrix(y_pred_test, test_label)
    cm_test = np.array2string(cm_test)
    count_misclassified = (test_label != y_pred_test).sum()
    accuracy = metrics.accuracy_score(test_label, y_pred_test)

    txt = ["For test data ", 'Misclassified samples: {}'.format(count_misclassified), 'Accuracy: {:.2f}'.format(accuracy)]
    log_list.append('\n'.join(txt))
    print(log_list[-1])

    # ===================================
    # == logging signature initialization
    # ===================================
    split = args.split
    # -- create dir for hyperparameter config if not already exists
    weighted_class = ''.join(list(map(str, args.weighted_class)))

    folder = f"log/{args.time_stamp}/LogistircRegression/split={split}/"

    import os
    if not os.path.exists(folder):
        os.makedirs(folder)

    # -- creat directory if not yet created
    save_path = f'{folder}img/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if args.log:
        save_path = f'emb_name={emb_name}_LogisticRegression_results.txt'
        print(f"writing to {save_path}...")
        with open(save_path, 'w') as f:
            txt = '\n\n'.join(log_list)
            f.write(txt)


def run_gcn_on_disease_graph(config, emb_name):
    '''
    Frame the problem by connect subgraph that has shared nodes
        ie. diseases that share node will be connected by an edges
    :param config:
    :return:
    '''
    # -- input arguments
    copd = config["data"]
    input = config["input"] # {disease_idx1: [[0,0,0,1,0,0], ....], disease_idx2: [...],... }
    y = config['label']
    train_mask = config['train_mask']
    test_mask = config['test_mask']
    emb = config['emb']
    hidden_sizes = config['hidden_layers']
    epochs = config['epochs']
    args = config['args']
    param = config['param']

    len_nodes = len(input.keys()) # amount of all node
    train_label = y[train_mask]
    test_label = y[test_mask]
    train_onehot = []
    test_onehot = []
    train_key = []
    test_key = []

    # -- convert onehot input into the following format
    # from
    #   {disease_idx1: [[0,0,0,1,0,0],[0,1,0,0,0,0] ....], disease_idx2: [...],... }
    # to
    #   {disease_idx1: [0,1,0,1,0,0], disease_idx2: [...],... }
    for key, val in input.items():
        sum = 0
        if int(key) in train_mask:
            for v in val:
                sum = np.add(sum, v)
            input[key] = sum
            train_onehot.append(input[key])
            train_key.append(key)
        sum1 = 0
        if int(key) in test_mask:
            for v in val:
                sum1 = np.add(sum1, v)
            input[key] = sum1
            test_onehot.append(input[key])
            test_key.append(key)

    # -- normalize feature
    train_input = normalize_features(csr_matrix(np.array(train_onehot)))
    test_input = normalize_features(csr_matrix(np.array(test_onehot)))

    # -- edge_index for disease_graph
    #   1. find overlap value between each disease
    edge_index = []

    # the higher the threshold, the most overfit to training set it is.
    # This is because in there will noly have edges to node that have edge sto more genes.
    th = int(args.th) # default = 100
    for d_out, k_out in zip(test_input, test_key):
        for d_in, k_in in zip(test_input, test_key):
            x = d_out - d_in
            x = x[x!=0]
            if x.shape[1] > th:
                if [k_out, k_in] not in edge_index and [k_in, k_out] not in edge_index:
                    # print(f"form edges between {k_out} and {k_in}")
                    edge_index.append([k_out, k_in])

    for d_out, k_out in zip(train_input, train_key):
        for d_in, k_in in zip(train_input, train_key):
            x = d_out - d_in
            x = x[x!=0]
            if x.shape[1] > th:
                if [k_out, k_in] not in edge_index and [k_in, k_out] not in edge_index:
                    # print(f"form edges between {k_out} and {k_in}")
                    edge_index.append([k_out, k_in])

    import math
    sparsity =  len(edge_index)/ math.factorial(len_nodes)

    print(f"num_edges = {len(edge_index)}")
    print(f"edges sparsity = {sparsity}" )

    edge_index = np.array(edge_index).T
    # display2screen(edge_index.shape, np.amax(edge_index.flatten()))

    # -- create train_input
    if emb_name != 'no_feat':
        train_input = emb[train_mask]
        test_input = emb[test_mask]
    else:
        train_input = train_input
        test_input = test_input

    # -- convert to tensor
    train_input = torch.tensor(train_input, dtype=torch.float)
    test_input = torch.tensor(test_input, dtype=torch.float)
    train_label = torch.tensor(train_label, dtype=torch.long)
    test_label = torch.tensor(test_label, dtype=torch.long)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    weighted_class = torch.tensor(args.weighted_class, dtype=torch.float)

    x = torch.cat((train_input,test_input), 0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # import torch_geometric
    from torch_geometric.nn import GCNConv, ChebConv, GATConv, SAGEConv


    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()

            modules = {
                # "conv1": GCNConv(20, args.hidden, cached=True),
                "conv1": GCNConv(train_input.shape[1], args.hidden, cached=True),
                "conv2": GCNConv(args.hidden, len(copd.labels2idx().keys()), cached=True)
            }

            for name, module in modules.items():
                self.add_module(name, module)

        def forward(self, x, edge_index):

            x = F.relu(self.conv1(x, edge_index))
            x = F.dropout(x, p=args.dropout, training=self.training)
            x = self.conv2(x, edge_index)
            return F.log_softmax(x, dim=1)

    gcn = Net().to(device)
    optimizer = torch.optim.Adam(gcn.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def unlabeled_weight(epochs):
        alpha = 0.0
        if epochs > param['T1']:
            if epochs > param['T2']:
                alpha = param['af']
            else:
                alpha = (epochs - param['T1']) / (param['T2'] - param['T1'] * param['af'])
        return alpha

    def train():
        gcn.train()
        optimizer.zero_grad()

        if args.pseudo_label_topk:

            labeled_loss = F.nll_loss(gcn(x, edge_index)[train_mask], train_label,
                                     weight=torch.tensor(list(map(int, args.weighted_class)), dtype=torch.float),
                                     reduction="mean")

            # -- labeled top k most confidence node to be pseduo_labels
            pseudo_label_pred = gcn(x, edge_index).max(1)[1]

            tmp = gcn(x, edge_index).max(1)[1].detach().flatten().tolist()
            tmp = [(l, i) for i, l in enumerate(tmp)]
            tmp = sorted(tmp, key=lambda x: x[0], reverse=True)  # rank label by predicted confidence value

            ranked_labels = [(l, i) for (l, i) in tmp]
            top_k_tuple = []

            for (l, i) in ranked_labels:
                if len(top_k_tuple) >= int(args.topk):
                    break

                top_k_tuple.append((i, l))  # get index of top_k to be masked during loss
            if len(top_k_tuple) >0:
                top_k = [t[0] for t in top_k_tuple]

                # -- add top_k to labeld_loss
                pseudo_label_loss = F.nll_loss(gcn(x, edge_index)[top_k], pseudo_label_pred[top_k], weight=weighted_class,
                                            reduction='mean')
            else:
                pseudo_label_loss = 0

            loss_output = labeled_loss + pseudo_label_loss
        else:
            loss_output = F.nll_loss(gcn(x, edge_index)[train_mask], train_label,
                                 weight=torch.tensor(list(map(int, args.weighted_class)), dtype=torch.float),
                                 reduction="mean")

        loss_output.backward()
        optimizer.step()
        return loss_output.data

    def test():
        gcn.eval()
        train_pred = gcn(x, edge_index)[train_mask].max(1)[1]
        train_acc = train_pred.eq(train_label).sum().item() / train_mask.shape[0]

        test_pred = gcn(x, edge_index)[test_mask].max(1)[1]
        test_acc = test_pred.eq(test_label).sum().item() / test_mask.shape[0]

        return [train_acc, test_acc]

    train_acc_hist = []
    test_acc_hist = []
    loss_hist = []
    log_list = []
    for epoch in range(epochs):
        loss_epoch = train()
        train_acc, test_acc = test()
        logging = 'Epoch: {:03d}, Train: {:.4f}, Test: {:.4f}'.format(epoch, train_acc, test_acc)
        if args.verbose:
            print(logging)
        log_list.append(logging)
        loss_hist.append(loss_epoch)
        train_acc_hist.append(train_acc)
        test_acc_hist.append(test_acc)

    split = args.split
    # -- create dir for hyperparameter config if not already exists
    weighted_class = ''.join(list(map(str, args.weighted_class)))

    HP = f'lr={args.lr}_d={args.dropout}_wd={args.weight_decay}'
    folder = f"log/{args.time_stamp}/gcn_on_disease_graph/split={split}/{HP}/"

    import os
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

    T_param = ','.join([str(param['T1']), str(param['T2'])])
    # -- creat directory if not yet created
    save_path = f'{folder}img/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if args.plot_all is True:
        args.plot_loss = True
        args.plot_no_train = True
        args.plot_train = True

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
        print(
            "writing to  " + save_path + f"LOSS_ACC_feat={feat_stat}_gene_thresh_hold={th}_wc=[{weighted_class}]_T=[{T_param}].png")
        plt.savefig(
            save_path + f'ACC_feat={feat_stat}_gene_thresh_hold={th}_wc=[{weighted_class}]_T=[{T_param}].png')
        plt.show()

    # --train_mask f1,precision,recall
    train_pred = gcn(x, edge_index)[train_mask].max(1)[1]
    train_f1 = f1_score(train_label, train_pred, average='micro')
    train_precision = precision_score(train_label, train_pred, average='micro')
    train_recall = recall_score(train_label, train_pred, average='micro')

    # -- test_mask f1,precision,recall
    test_pred = gcn(x, edge_index)[test_mask].max(1)[1]
    test_f1 = f1_score(test_label, test_pred, average='micro')
    test_precision = precision_score(test_label, test_pred, average='micro')
    test_recall = recall_score(test_label, test_pred, average='micro')

    if args.log:
        save_path = f'{folder}ACC_feat={feat_stat}_pseudo_label={pseudo_label_stat}_gene_thresh_hold={th}_wc={weighted_class}.txt'
        print(f"writing to {save_path}...")
        with open(save_path, 'w') as f:
            txt = '\n'.join(log_list)
            f.write(txt)

    if args.log:
        cm_train = confusion_matrix(gcn(x, edge_index)[train_mask].max(1)[1], train_label)
        cm_test = confusion_matrix(gcn(x, edge_index)[test_mask].max(1)[1], test_label)

        # formatter = {'float_kind': lambda x: "%.2f" % x})
        cm_train = np.array2string(cm_train)
        cm_test = np.array2string(cm_test)

        save_path = f'{folder}CM_feat={feat_stat}_pseudo_label={pseudo_label_stat}_gene_thresh_hold={th}_wc={weighted_class}.txt'
        print(f"writing to {save_path}...")

        # txt = 'class int_rep is [' + ','.join(list(map(str, np.unique(data.y.numpy()).tolist()))) + ']'
        txt = 'class int_rep is [' + ','.join([str(i) for i in range(len(copd.labels2idx().values()))]) + ']'
        txt = txt + '\n\n' + "training cm" + '\n' + cm_train + '\n' \
              + f"training_accuracy ={log_list[-1].split(',')[1]}" + '\n' \
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


# -- MLP
def run_mlp(config):
    '''
    run multi-layer perceptron
    input data is node with gene as its features.
    :return:
    '''
    # -- input arguments
    copd = config["data"]
    input = config["input"] # {disease_idx1: [[0,0,0,1,0,0],[0,1,0,0,0,0] ....], disease_idx2: [...],... }
    y = config['label']
    train_mask = config['train_mask']
    test_mask = config['test_mask']
    hidden_sizes = config['hidden_layers']
    epochs = config['epochs']
    args = config['args']
    param = config['param']
    # display2screen(np.asarray(input[key] for key,val in input.items() if int(key) in train_mask))

    # -- initialization
    # train_input = [input[key] for key,val in input.items() if int(key) in train_mask]
    # test_input  = [input[key] for key,val in input.items() if int(key) in test_mask]
    train_label = y[train_mask]
    test_label  = y[test_mask]
    train_input = []
    test_input = []

    # -- convert onehot input into the following format
    # from
    #   {disease_idx1: [[0,0,0,1,0,0],[0,1,0,0,0,0] ....], disease_idx2: [...],... }
    # to
    #   {disease_idx1: [0,1,0,1,0,0], disease_idx2: [...],... }
    for key, val in input.items():
        sum = 0
        if int(key) in train_mask:
            for v in val:
                sum = np.add(sum,v)
            input[key] = sum
            train_input.append(input[key])
        sum1 = 0
        if int(key) in test_mask:
            for v in val:
                sum1 = np.add(sum1,v)
            input[key] = sum1
            test_input.append(input[key])

    # -- normalize features vector
    train_input = normalize_features(csr_matrix(np.array(train_input)))
    test_input  = normalize_features(csr_matrix(np.array(test_input)))

    # -- convert to tensor
    train_input = torch.tensor(train_input, dtype=torch.float )
    test_input  = torch.tensor(test_input, dtype=torch.float )
    train_label = torch.tensor(train_label, dtype=torch.long )
    test_label  = torch.tensor(test_label, dtype=torch.long )
    weighted_class = torch.tensor(list(map(int,args.weighted_class)), dtype=torch.float)


    # model() -> optimizer -> loss -> model.train()-> optimizer.zero_grad() -> loss.backward() -> optimizer.step() -> next epoch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # mlp = MLP(hidden_sizes).to(device)
    mlp = nn.Sequential(
                            nn.Linear(2996, 128),
                            nn.ReLU(),
                            nn.Linear(128, 16),
                            nn.ReLU(),
                            nn.Linear(16, len(copd.labels2idx().keys())),
                            nn.LogSoftmax(dim=1)
                        )
    optimizer = torch.optim.Adam(mlp.parameters(), lr=args.lr, weight_decay=args.weight_decay)



    def train():
        mlp.train()
        optimizer.zero_grad()
        if args.pseudo_label_topk:
            labeled_loss = F.nll_loss(mlp(train_input), train_label,
                                      weight=torch.tensor(list(map(int, args.weighted_class)), dtype=torch.float),
                                      reduction="mean")

            # -- labeled top k most confidence node to be pseduo_labels
            pseudo_label_pred = mlp(train_input).max(1)[1]

            tmp = mlp(train_input).max(1)[1].detach().flatten().tolist()
            tmp = [(l, i) for i, l in enumerate(tmp)]
            tmp = sorted(tmp, key=lambda x: x[0], reverse=True)  # rank label by predicted confidence value

            ranked_labels = [(l, i) for (l, i) in tmp]
            top_k_tuple = []

            for (l, i) in ranked_labels:
                if len(top_k_tuple) >= int(args.topk):
                    break

                top_k_tuple.append((i, l))  # get index of top_k to be masked during loss
            if len(top_k_tuple) > 0:
                top_k = [t[0] for t in top_k_tuple]

                # -- add top_k to labeld_loss
                pseudo_label_loss = F.nll_loss(mlp(train_input)[top_k], pseudo_label_pred[top_k],
                                               weight=weighted_class,
                                               reduction='mean')
            else:
                pseudo_label_loss = 0

            loss_output = labeled_loss + pseudo_label_loss
        else:
            loss_output = F.nll_loss(mlp(train_input), train_label, weight=torch.tensor(list(map(int,args.weighted_class)), dtype=torch.float), reduction="mean")
        loss_output.backward()
        optimizer.step()
        return loss_output.data
    def test():
        mlp.eval()
        train_pred = mlp(train_input).max(1)[1]
        train_acc = train_pred.eq(train_label).sum().item() / train_mask.shape[0]

        test_pred = mlp(test_input).max(1)[1]
        test_acc = test_pred.eq(test_label).sum().item() / test_mask.shape[0]

        return [train_acc, test_acc]

    train_acc_hist = []
    test_acc_hist = []
    loss_hist = []
    log_list = []
    for epoch in range(epochs):
        loss_epoch = train()
        train_acc, test_acc = test()
        logging = 'Epoch: {:03d}, Train: {:.4f}, Test: {:.4f}'.format(epoch, train_acc, test_acc)
        if args.verbose:
            print(logging)
        log_list.append(logging)
        loss_hist.append(loss_epoch)
        train_acc_hist.append(train_acc)
        test_acc_hist.append(test_acc)

    split = args.split
    # -- create dir for hyperparameter config if not already exists
    weighted_class = ''.join(list(map(str, args.weighted_class)))

    HP = f'lr={args.lr}_d={args.dropout}_wd={args.weight_decay}'
    folder = f"log/{args.time_stamp}/mlp/split={split}/{HP}/"

    import os
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

    T_param = ','.join([str(param['T1']), str(param['T2'])])
    # -- creat directory if not yet created
    save_path = f'{folder}img/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if args.plot_all is True:
        args.plot_loss = True
        args.plot_no_train = True
        args.plot_train = True

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
        print(
            "writing to  " + save_path + f"LOSS_ACC_feat={feat_stat}_pseudo_label={pseudo_label_stat}_wc=[{weighted_class}]_T=[{T_param}]_topk={args.topk}.png")
        plt.savefig(
            save_path + f'ACC_feat={feat_stat}_pseudo_label={pseudo_label_stat}_wc=[{weighted_class}]_T=[{T_param}]_topk={args.topk}.png')
        plt.show()


    # --train_mask f1,precision,recall
    train_pred = mlp(train_input).max(1)[1]
    train_f1 = f1_score(train_label, train_pred, average='micro')
    train_precision = precision_score(train_label, train_pred, average='micro')
    train_recall = recall_score(train_label, train_pred, average='micro')

    # -- test_mask f1,precision,recall
    test_pred = mlp(test_input).max(1)[1]
    test_f1 = f1_score(test_label, test_pred, average='micro')
    test_precision = precision_score(test_label, test_pred, average='micro')
    test_recall = recall_score(test_label, test_pred, average='micro')

    if args.log:
        # save_path = f'log/{args.arch}/{HP}/{time_stamp}/feat_stat={feat_stat}_{args.arch}_accuracy_{emb_name}{time_stamp}_split_{split}.txt'
        save_path = f'{folder}ACC_feat={feat_stat}_pseudo_label={pseudo_label_stat}_wc={weighted_class}_topk={args.topk}.txt'
        print(f"writing to {save_path}...")
        with open(save_path, 'w') as f:
            txt = '\n'.join(log_list)
            f.write(txt)

    if args.log:
        cm_train = confusion_matrix(mlp(train_input).max(1)[1], train_label)
        cm_test = confusion_matrix(mlp(test_input).max(1)[1], test_label)

        # formatter = {'float_kind': lambda x: "%.2f" % x})
        cm_train = np.array2string(cm_train)
        cm_test = np.array2string(cm_test)

        save_path = f'{folder}CM_feat={feat_stat}_pseudo_label={pseudo_label_stat}_wc={weighted_class}_topk={args.topk}.txt'
        print(f"writing to {save_path}...")

        # txt = 'class int_rep is [' + ','.join(list(map(str, np.unique(data.y.numpy()).tolist()))) + ']'
        txt = 'class int_rep is [' + ','.join([str(i) for i in range(len(copd.labels2idx().values()))]) + ']'
        txt = txt + '\n\n' + "training cm" + '\n' + cm_train + '\n' \
              + f"training_accuracy ={log_list[-1].split(',')[1]}" + '\n' \
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

#=================================
# == preprocessing
#=================================
def normalize_features(mx):
    """
        Row-normalize sparse matrix
    :param: mx: csr_matrix
    :return mx: numpy array
    """

    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx.todense()

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt) # D^0-5 * A * D^0.5

def add_features(args):
    # ===========================
    # === add embedding as features
    # ===========================
    emb_name = args.emb_name
    emb_file = None
    # -- emb_file
    if args.emb_name == 'attentionwalk':
        emb_file = f"{args.emb_name}/{args.emb_name}_emb{args.time_stamp}.txt"

    elif args.emb_name == 'node2vec':
        if args.subgraph:
            emb_file = f"{args.emb_name}/{args.emb_name}_emb_subgraph{args.time_stamp}.txt"
        else:
            emb_file = f"{args.emb_name}/{args.emb_name}_emb_fullgraph{args.time_stamp}.txt"

    elif args.emb_name == 'bine':
        emb_file = f"{args.emb_name}/bine{args.time_stamp}.txt"

    else:
        raise ValueError("provided emb_name is not supported!")
    assert emb_file is not None, f"{args.emb_name} is not available"

    # -- emb_path
    emb_path = args.emb_path + emb_file

    with open(emb_path, 'r') as f:
        tmp = f.readlines()
        if "bine" not in emb_file:
            tmp = tmp[1:]

    # -- split symbol
    if args.emb_name == "attentionwalk":
        split = ','
    if args.emb_name == "node2vec":
        split = ' '
    if args.emb_name == "bine":
        split = ' '

    if args.emb_name == "bine":
        emb_dict = {int(float(i.split(split)[0][1:])): list(map(float, i.split(split)[1:])) for i in tmp}
    else:
        emb_dict = {int(float(i.split(split)[0])): list(map(float, i.split(split)[1:])) for i in tmp}

    # -- make sure that node embs are in ordered
    emb = sorted(emb_dict.items(), key=lambda t: t[0])
    x = np.array([[j for j in i[1]] for i in emb], dtype=np.float)
    x = torch.tensor(x, dtype=torch.float)  # torch.Size([2996, 64])

    return x
# ====================
# == dataset
# ====================
class Cora():
    def __init__(self):
        pass

    # todo here>> check feature in cora why each node only have at most 1 feature???????
    def load_data(self, path="./data/cora/", dataset="cora"):
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


class Copd():
    def __init__(self, path=None, data=None, time_stamp=None):

        self.time_stamp = time_stamp
        # todo here>> check which file does GetData read from. and what is the max_int_rep of nodes in these files
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

    # -- rename "conversion related function" (can be under this section)
    def labels2class(self):
        '''
            rename for diseases2class

        :return:
            {cuis: labels}
        '''
        cuis2labels = {self.disease2idx()[c]: self.labels2idx()[l] for c, l in zip(self.disease, self.labels)}

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

        cuis2labels = {self.disease2idx()[c]: self.labels2idx()[l] for c, l in zip(self.disease, self.labels)}

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

        # display2screen(np.unique(gene).shape)

        # 61 is the amount of diseaseid
        return {l: i + int(self.disease.shape[0]) for i, l in enumerate(list(collections.OrderedDict.fromkeys(gene.tolist()).keys()))}

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

        # display2screen(max(uniq_diseases), max(labels), max(genes))
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




