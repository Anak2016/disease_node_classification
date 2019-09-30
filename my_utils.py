import sys,os
USER = os.environ['USERPROFILE']
sys.path.insert(1,f'{USER}\\PycharmProjects\\my_utility')

from utility_code.my_utility import *
from utility_code.python_lib_essential import *

import preprocessing

# -- files manipulation
def create_copd_label_content(path='./data/{args.time_stamp}/gene_disease/raw/',file_name= "copd_label", time_stamp='', **kwargs):
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

def create_copd_label_edges(path='./data/{args.time_stamp}/gene_disease/raw/',file_name= "copd_label", time_stamp='', **kwargs):
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

def create_edges_dict(edges=None, use_nodes=None):
    '''

    :param edges:
        [(disease1, gene1), (disease2,gene2.....] where disease are sorted in ascending order
        eg. [(0,106),(1,400),(1,203),... ]

    :return:_list
            _list has the follwoing format must be in the followin format:

        graph = {source_node1: [{target_node: weight}, ... ]
                source_node2: [{target_node: weight}, ... ]
                ,....,}
        nodes_with_shared_genes =
                {source_node1: [target_node, ... ]
                source_node2: [target_node, ... ]
                ,....,}
    '''
    # adj_list = {i: [] for (i, j) in set(flatten(edges)) }
    if use_nodes == 'all':
        tmp = set(flatten(edges))
        adj_list = dict.fromkeys(tmp, [])
        nodes_with_shared_genes = {i: [] for i in set(flatten(edges)) }
        for disease, gene in edges:
            adj_list[disease].append({gene:'1'})
            adj_list[gene].append({disease:'1'})
            nodes_with_shared_genes[disease].append(gene)
            nodes_with_shared_genes[gene].append(disease)
    elif use_nodes=='gene':
        adj_list = dict.fromkeys(list(i[0] for i in edges), [])
        # nodes_with_shared_genes = dict.fromkeys(list(i[0] for i in edges), [])
        nodes_with_shared_genes = {i: [] for (i, j) in edges}
        for disease, gene in edges:
            adj_list[disease].append({gene:'1'})
            #TODO here>> nodes_with_shared_genes are wrong here
            nodes_with_shared_genes[disease].append(gene)
    elif use_nodes=='disease':
        adj_list = dict.fromkeys(list(i[1] for i in edges), [])
        nodes_with_shared_genes = {j: [] for (i, j) in edges}
        # nodes_with_shared_genes = dict.fromkeys(list(i[0] for i in edges), [])
        for disease, gene in edges:
            adj_list[gene].append({disease:'1'})
            nodes_with_shared_genes[gene].append(disease)

    return adj_list, nodes_with_shared_genes



# =======================
# == gcn_on_disease_graph
# =======================


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
    train_input = preprocessing.normalize_features(csr_matrix(np.array(train_onehot)))
    test_input = preprocessing.normalize_features(csr_matrix(np.array(test_onehot)))

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
    folder = f"log/gene_disease/{args.time_stamp}/gcn_on_disease_graph/split={split}/{HP}/"

    import os
    if not os.path.exists(folder):
        os.makedirs(folder)

    # if args.add_features:
    if args.emb_name != "no_feat":
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





