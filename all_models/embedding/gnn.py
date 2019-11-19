import sys,os
USER = os.environ['USERPROFILE']
sys.path.insert(1,f'{USER}\\PycharmProjects\\my_utility')

from utility_code.my_utility import *
from utility_code.python_lib_essential import *

from arg_parser import *
from torch_geometric.nn import GCNConv, GATConv, SAGEConv  # noqa
import plotting
import hyper_params_search
import performance_metrics

#=====================
#==Geometric model
#=====================
class Net(torch.nn.Module):
    def __init__(self, data, modules):
        super(Net, self).__init__()
        self.data = data
        self._emb_output = None
        for name, module in modules.items():
            self.add_module(name, module)

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

    def get_emb_output(self):
        return self._emb_output

    def forward(self):
        data = self.data
        x, edge_index = data.x, data.edge_index

        #TODO here>> check edge_index here whether they are the same or not.

        # display2screen(x[1,:])
        # display2screen(x.shape, edge_index.shape, np.amax(edge_index.numpy()))

        x = x.type(torch.float)
        edge_index = edge_index.type(torch.long)

        if args.arch == 'gcn':
            # todo check loss fucntion for gcn
            x  = self.conv1(x, edge_index)
            self._emb_output = x

            x = F.relu(x)
            # self._emb_output = x

            x = F.dropout(x, p=args.dropout, training=self.training)
            x = self.conv2(x, edge_index)
            # self._emb_output = x

            x = F.relu(x) # chosen
            x = F.dropout(x, p=args.dropout, training=self.training)
            x = self.Linear16_out(x) # chosen
            # x = F.relu(self.Linear16_out(x)) # chosen

            # x = F.relu(self.Linearin_32(x))
            # x = F.relu(self.Linear32_16(x))
            # x = F.relu(self.Linear32_out(x))

            # x = F.dropout(x, p=dropout, training=self.training)
            # x = self.conv3(x, edge_index)
            return F.log_softmax(x, dim=1)

        if args.arch == 'gat':
            # todo what is the loss function use in gat?
            x = F.dropout(data.x, p=0.6, training=self.training)
            x = F.elu(self.conv1(x, edge_index))
            self._emb_output = x

            x = F.dropout(x, p=0.6, training=self.training)
            x = self.conv2(x, edge_index)
            return F.log_softmax(x, dim=1)

        if args.arch == 'sage':
            # todo what is the los func use in sage?
            # graphSage original architecutre has depth = 2.
            x = F.relu(self.conv1(x, edge_index))
            self._emb_output = x
            # x = F.dropout(x, p=dropout, training=self.training)
            x = self.conv2(x, edge_index)
            return F.log_softmax(x, dim=1)


class PseudoLabel:
    def __init__(self):
        pass

# def run_GNN(data):
class GNN:
    def __init__(self, data):
        config = {
            "param": {  # Pseudo-Label
                'T1': int(args.t1_t2_alpha[0]),
                'T2': int(args.t1_t2_alpha[1]),
                'af': float(args.t1_t2_alpha[2])
            }
        }
        self.curr_time = datetime.now().strftime('%Y_%m_%d_%H_%M')
        self.data = data
        self.config = config
        self.epochs = args.epochs
        weighted_class = [1, 1, 1, 1, 1, 1, 0] if args.dataset == 'cora' else args.weighted_class
        self.weighted_class = torch.tensor(list(map(int, weighted_class)) + [0], dtype=torch.float)
        self.model = None
        self.optimizer = None
        weighted_class = ''.join(list(map(str, args.weighted_class)))

        #=====================
        #==naming convension
        #=====================
        print(self.weighted_class.numpy().astype(int))
        self.HP = f'lr={args.lr}_d={args.dropout}_wd={args.weight_decay}_wc={self.weighted_class.numpy().astype(int)}'

        self.folder = f"log/gene_disease/{args.time_stamp}/classifier/{args.added_edges_option}/gcn/split={data.split}/{self.HP}/"

        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        self.feat_stat = 'YES' if args.emb_name != "no_feat" else 'NO'
        if args.pseudo_label_all:
            self.pseudo_label_stat = "ALL"
        elif args.pseudo_label_topk:
            self.pseudo_label_stat = "TOP_K"
        elif args.pseudo_label_topk_with_replacement:
            self.pseudo_label_stat = "TOP_K_WITH_REPLACEMENT"
        else:
            self.pseudo_label_stat = "NONE"

        self.T_param = ','.join([str(self.config['param']['T1']),str(self.config['param']['T2'])])

    #=====================
    #==setup
    #=====================

    #=====================
    #==preprocessing
    #=====================

    #=====================
    #==embedding models
    #=====================

    def run_model(self, modules):
        '''model() -> optimizer -> loss -> model.train()-> optimizer.zero_grad() -> loss.backward() -> optimizer.step() -> next epoch'''
        data = self.data
        #TODO here>> these modules needs to be fed in as an argument to __init__
        if args.arch == 'gcn':
            # modules = {
            #     # "conv1": GCNConv(64, args.hidden, cached=True),
            #     "conv1": GCNConv(data.num_features, args.hidden, cached=True),
            #     "conv2": GCNConv(args.hidden, data.num_classes, cached=True),
            #
            # }
            pass
        elif args.arch == 'gat':
            modules = {
                "conv1": GATConv(data.num_features, args.hidden, heads=args.heads, dropout=0.6),
                "conv2": GATConv(args.hidden * args.heads, data.num_classes, heads=1, concat=True, dropout=0.6)
            }
        elif args.arch == 'sage':
            modules = {
                "conv1": SAGEConv(data.num_features, args.hidden, aggr=args.aggr),
                "conv2": SAGEConv(args.hidden, data.num_classes, aggr=args.aggr)
            }
        else:
            raise ValueError("input of --arch is not supported")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Net(data, modules).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    #=====================
    #==pseudo_label
    #=====================
    def pseudo_label(self, epoch,labeled_index, target):
        model = self.model
        data = self.data
        weighted_class = self.weighted_class

        if args.pseudo_label_all:
            # -- pseudo_label is differnet from cost-sensitivity
            labeled_loss = F.nll_loss(model()[data.train_mask], data.y[data.train_mask], weight=weighted_class,
                                      reduction="mean")
            # for unlabled dataset
            unlabeled_nodes = [i for i in range(len(data.y)) if
                               i > np.amax(data.train_mask.numpy().tolist() + data.test_mask.numpy().tolist())]
            # labeled pseduo_label by class that is predicted with the most confident
            pseudo_label_pred = model()[unlabeled_nodes].max(1)[1]
            unlabeled_loss = F.nll_loss(model()[unlabeled_nodes], pseudo_label_pred, weight=weighted_class,
                                        reduction='mean')
            # unlabeled_loss = unlabeled_loss/ len(unlabeled_nodes)

            loss_output = labeled_loss + self.unlabeled_weight(epoch) * unlabeled_loss

        # todo maybe i can update target and label_index inside of "data" instance
        elif args.pseudo_label_topk:
            th = 0
            if epoch == th:
                # -- pseudo_label is differnet from cost-sensitivity
                if epoch == th and labeled_index is None:
                    labeled_index = data.train_mask
                    target = data.y[labeled_index]  # target = labeled_data

                # -- leak of label of test dataset
                # labeled_loss = F.nll_loss(model()[labeled_index], data.y[labeled_index], weight=weighted_class, reduction="mean")

                # -- no leak of label of test dataset
                labeled_loss = F.nll_loss(model()[labeled_index], target, weight=weighted_class, reduction="mean")

                # -- index of all nodes
                all_nodes = [i for i in range(len(data.y))]

                assert 1 + np.amax(data.train_mask.numpy().tolist() + data.test_mask.numpy().tolist()) == len(
                    data.labeled_nodes().keys()), f"{len(data.labeled_nodes().keys())} != {1 + np.amax(data.train_mask.numpy().tolist() + data.test_mask.numpy().tolist())}"

                # -- labeled top k most confidence node to be pseduo_labels
                pseudo_label_pred = model()[all_nodes].max(1)[1]

                tmp = model()[all_nodes].max(1)[1].detach().flatten().tolist()
                tmp = [(l, i) for i, l in enumerate(tmp)]
                tmp = sorted(tmp, key=lambda x: x[0], reverse=True)  # rank label by predicted confidence value

                ranked_labels = [(l, i) for (l, i) in tmp]
                top_k_tuple = []

                for (l, i) in ranked_labels:
                    if len(top_k_tuple) >= int(args.topk):
                        break
                    # if True: # obtian highest accuracy; basically always append i
                    # if i not in  list(copd.labelnodes2idx().values()): # bad accuracy because label is not added along with it
                    if i not in labeled_index:  # this also obtain highest accuracy; every label can be included at most 1 time in loss function
                        top_k_tuple.append((i, l))  # get index of top_k to be masked during loss

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

                if len(top_k) > 0:
                    target = torch.cat((target, torch.tensor(new_labels)), 0)

                # -- add top_k to labeld_loss
                if epoch > th and len(top_k) > 0:
                    len_before_topk = labeled_index.shape[0]
                    labeled_index = torch.cat((labeled_index, torch.tensor(top_k)), 0)
                    assert len_before_topk + len(top_k) == labeled_index.shape[
                        0], "recently added top_k index are already included in labled_index"

                    unlabeled_loss = F.nll_loss(model()[top_k], pseudo_label_pred[top_k], weight=weighted_class,
                                                reduction='mean')
                    # unlabeled_loss = F.nll_loss(model()[top_k], torch.tensor(new_labels), weight=weighted_class, reduction='mean')

                    loss_output = labeled_loss + self.unlabeled_weight(epoch) * unlabeled_loss

                else:
                    if len(top_k) > 0:
                        len_before_topk = labeled_index.shape[0]

                        labeled_index = torch.cat((labeled_index, torch.tensor(top_k)), 0)
                        assert len_before_topk + len(top_k) == labeled_index.shape[
                            0], "recently added top_k index are already included in labled_index"

                        unlabeled_loss = F.nll_loss(model()[top_k], pseudo_label_pred[top_k], weight=weighted_class,
                                                    reduction='mean')
                        # unlabeled_loss = F.nll_loss(model()[top_k], torch.tensor(new_labels), weight=weighted_class,reduction='mean')

                        loss_output = labeled_loss + self.unlabeled_weight(epoch) * unlabeled_loss
                    else:
                        # top_k == 0 => unlabeled_loss has no input => output of unlabeled_los s= 0
                        unlabeled_loss = 0
                        loss_output = labeled_loss + self.unlabeled_weight(epoch) * unlabeled_loss
            else:
                loss_output = F.nll_loss(model()[data.train_mask], data.y[data.train_mask], weight=weighted_class,
                                         reduction="mean")

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
            labeled_loss = F.nll_loss(model()[data.train_mask], data.y[data.train_mask], weight=weighted_class,
                                      reduction="mean")

            # -- add top_k to labeld_loss
            if epoch > 1 and len(top_k) > 0:

                # unlabeled_loss = F.nll_loss(model()[top_k], pseudo_label_pred[top_k], weight=weighted_class, reduction='mean')
                unlabeled_loss = F.nll_loss(model()[top_k], torch.tensor(new_labels), weight=weighted_class,
                                            reduction='mean')

                loss_output = labeled_loss + self.unlabeled_weight(epoch) * unlabeled_loss

            else:
                if len(top_k) > 0:

                    # unlabeled_loss = F.nll_loss(model()[top_k], pseudo_label_pred[top_k], weight=weighted_class, reduction='mean')
                    unlabeled_loss = F.nll_loss(model()[top_k], torch.tensor(new_labels), weight=weighted_class,
                                                reduction='mean')

                    loss_output = labeled_loss + self.unlabeled_weight(epoch) * unlabeled_loss
                else:
                    # top_k == 0 => unlabeled_loss has no input => output of unlabeled_los s= 0
                    unlabeled_loss = 0
                    loss_output = labeled_loss + self.unlabeled_weight(epoch) * unlabeled_loss
        else:
            # todo here>>
            loss_output = F.nll_loss(model()[data.train_mask], data.y[data.train_mask], weight=weighted_class,
                                     reduction="mean")
        return loss_output
    #=====================
    #==plotting
    #=====================
    def plot_loss(self, loss_hist, train_acc_hist, test_acc_hist, save_path):
        data = self.data
        config = {
            "loss history": {
                'x_label': 'False Positive Rate',
                'y_label': 'loss values',
                'legend': [{"kwargs": {"loc": "lower right"}}],
                'plot': [{"args": [range(len(loss_hist)), loss_hist]}]
            },
            "accuracy history": {
                'x_label': 'False Positive Rate',
                'y_label': 'accuracy values',
                'legend': [{"kwargs": {"loc": "lower right"}}],
                'plot': [{"args": [range(len(train_acc_hist)), train_acc_hist]},
                         {"args": [range(len(test_acc_hist)), test_acc_hist]},
                         ]
            },
        }
        # =====================
        # ==not using performance_metrics.report_performances
        # =====================
        file_name = f'epoch={args.epochs}_ACC_feat={self.feat_stat}_pseudo_label={self.pseudo_label_stat}_wc={self.weighted_class}_T=[{self.T_param}]_topk={args.topk}.png'
        plotting.plot_figures(config, save_path=save_path, file_name=file_name)



    def plot_emb(self, gcn_emb, save_path, file_emb, img_emb):

        # --------save embedding
        df = pd.DataFrame(gcn_emb.detach().numpy())
        df.to_csv(save_path + file_emb, header=True, index=False, sep='\t', mode='w')
        df_pred = pd.DataFrame(pd.DataFrame({"pred": gcn_emb.max(1)[1].detach().numpy()}))

        # -- gcn emb with training feedback
        print("--gcn emb with training feedback")
        if args.plot_all is True:
            # timer(plotting.plot_2d, self.data.dataset, save_path, file_gcn_emb, emb='gcn', with_gene=True,
            #       func=args.plot_2d_func, log=args.log, save_img=img_gcn_emb, pred_label=df_pred)
            # timer(plotting.plot_2d, self.data.dataset, save_path, file_gcn_emb, emb='gcn', with_gene=False,
            #       func=args.plot_2d_func, log=args.log, save_img=img_gcn_emb, pred_label=df_pred)
            plotting.plot_2d(self.data.dataset, save_path, file_emb, emb='gcn', with_gene=True,
                             func=args.plot_2d_func, log=args.log, save_img=img_emb, pred_label=df_pred)
            plotting.plot_2d(self.data.dataset, save_path, file_emb, emb='gcn', with_gene=False,
                             func=args.plot_2d_func, log=args.log, save_img=img_emb, pred_label=df_pred)
        else:
            # timer(plotting.plot_2d, self.data.dataset, save_path, file_gcn_emb, emb='gcn', with_gene=args.with_gene,
            #       func=args.plot_2d_func, log=args.log, save_img=img_gcn_emb, pred_label=df_pred)
            plotting.plot_2d(self.data.dataset, save_path, file_emb, emb='gcn', with_gene=args.with_gene,
                             func=args.plot_2d_func, log=args.log, save_img=img_emb, pred_label=df_pred)

    def plot(self, gcn_emb, gcn_emb_no_train, loss_hist, train_acc_hist, test_acc_hist, file_gcn_emb, img_gcn_emb):
        data = self.data
        model = self.model

        # -- creat directory if not yet created
        save_path = f'{self.folder}img/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if args.plot_all is True:
            args.plot_loss = True
            args.plot_no_train = True
            args.plot_train = True

        # =====================
        # ==plot loss function
        # =====================

        config = {}
        if args.plot_loss:
            '''plot loss and acc value'''
            self.plot_loss(loss_hist, train_acc_hist, test_acc_hist, save_path)

        # ==========================
        # === plot 2D output GCN embedding
        # ==========================
        save_path = f'data/gene_disease/{args.time_stamp}/processed/embedding/gcn/'
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        self.plot_emb(gcn_emb_no_train, save_path, file_gcn_emb, img_gcn_emb)

        #=====================
        #==old plotting
        #=====================
        # if args.plot_no_train:
        #     '''plot emd before it is trained'''
        #     # file_gcn_emb = f"epoch={args.epochs}_emb={args.emb_name}_TRAIN=NO_ACC_feat={self.feat_stat}_pseudo_label={self.pseudo_label_stat}_wc={self.weighted_class}_T=[{self.T_param}]_topk={args.topk}.txt"
        #     # img_gcn_emb = f"img/epoch={args.epochs}_emb={args.emb_name}_TRAIN=NO_ACC_feat={self.feat_stat}_pseudo_label={self.pseudo_label_stat}_wc={self.weighted_class}_T=[{self.T_param}]_topk={args.topk}.png"
        #     self.plot_emb(gcn_emb_no_train, save_path, file_gcn_emb, img_gcn_emb)
        #
        #     # # --------save embedding
        #     # df = pd.DataFrame(self.data.x.numpy())  # output before first epoch
        #     # tmp = save_path + file_gcn_emb
        #     # if not os.path.exists(tmp):
        #     #     with open(tmp, 'w'): pass
        #     # with open(tmp,'w') as f:
        #     #     print(f"writing to {tmp}...")
        #     #     f.write(df.__repr__())
        #     # # df.to_csv(save_path + file_gcn_emb, sep=' ')
        #     #
        #     # df_pred = pd.DataFrame(pd.DataFrame({"pred": gcn_emb_no_train.max(1)[1].detach().numpy()}))
        #     #
        #     # # -- gcn emb with no training feedback
        #     # print("--gcn emb with no training feedback")
        #     #
        #     # if args.plot_all is True:
        #     #     # timer(plotting.plot_2d, self.data.dataset, save_path, file_gcn_emb, emb='gcn', with_gene=True,
        #     #     #       func=args.plot_2d_func, log=args.log, save_img=img_gcn_emb, pred_label=df_pred)
        #     #     # timer(plotting.plot_2d, self.data.dataset, save_path, file_gcn_emb, emb='gcn', with_gene=False,
        #     #     #       func=args.plot_2d_func, log=args.log, save_img=img_gcn_emb, pred_label=df_pred)
        #     #
        #     #     plotting.plot_2d(self.data.dataset, save_path, file_gcn_emb, emb='gcn', with_gene=True, func=args.plot_2d_func, log=args.log,save_img=img_gcn_emb, pred_label=df_pred)
        #     #     plotting.plot_2d(self.data.dataset, save_path, file_gcn_emb, emb='gcn', with_gene=False, func=args.plot_2d_func, log=args.log,save_img=img_gcn_emb, pred_label=df_pred)
        #     # else:
        #     #     # timer(plotting.plot_2d, data.dataset, save_path, file_gcn_emb, emb='gcn', with_gene=args.with_gene,
        #     #     #       func=args.plot_2d_func, log=args.log, save_img=img_gcn_emb, pred_label=df_pred)
        #     #     plotting.plot_2d(self.data.dataset, save_path, file_gcn_emb, emb='gcn', with_gene=args.with_gene,
        #     #             func=args.plot_2d_func, log=args.log, save_img=img_gcn_emb, pred_label=df_pred)
        #
        # if args.plot_train:
        #     file_gcn_emb = f"epoch={args.epochs}_emb={args.emb_name}_TRAIN=YES_ACC_feat={self.feat_stat}_pseudo_label={self.pseudo_label_stat}_wc={self.weighted_class}_T=[{self.T_param}]_topk={args.topk}.txt"
        #     img_gcn_emb = f"img/epoch={args.epochs}_emb={args.emb_name}_TRAIN=YES_ACC_feat={self.feat_stat}_pseudo_label={self.pseudo_label_stat}_wc={self.weighted_class}_T=[{self.T_param}]_topk={args.topk}.png"
        #     self.plot_emb(gcn_emb, save_path, file_gcn_emb, img_gcn_emb)


    #=====================
    #==logging
    #=====================
    def logging(self, log_list):

        data = self.data
        model = self.model
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

        save_path = f'log/gene_disease/{args.arch}/{self.HP}/{args.time_stamp}/feat_stat={self.feat_stat}_{args.arch}_accuracy_{args.emb_name}{args.time_stamp}_split_{data.split}.txt'
        save_path = f'{self.folder}emb_name={args.emb_name}_ACC_feat={self.feat_stat}_pseudo_label={self.pseudo_label_stat}_wc={self.weighted_class}.txt'
        print(f"writing to {save_path}...")
        with open(save_path, 'w') as f:
            txt = '\n'.join(log_list)
            f.write(txt)

        model = self.model
        data = self.data

        cm_train = confusion_matrix(model()[data.train_mask].max(1)[1], data.y[data.train_mask])
        cm_test = confusion_matrix(model()[data.test_mask].max(1)[1], data.y[data.test_mask])

        # formatter = {'float_kind': lambda x: "%.2f" % x})
        cm_train = np.array2string(cm_train)
        cm_test = np.array2string(cm_test)

        save_path = f'{self.folder}emb_name={args.emb_name}_CM_feat={self.feat_stat}_pseudo_label={self.pseudo_label_stat}_wc={self.weighted_class}.txt'
        print(f"writing to {save_path}...")

        # txt = 'class int_rep is [' + ','.join(list(map(str, np.unique(data.y.numpy()).tolist()))) + ']'
        txt = 'class int_rep is [' + ','.join([str(i) for i in range(data.num_classes)]) + ']'
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
    # =====================
    # ==utility
    # =====================
    def unlabeled_weight(self, epoch):
        alpha = 0.0
        if epoch > self.config["param"]['T1']:
            if epoch > self.config["param"]['T2']:
                alpha = self.config["param"]['af']
            else:
                alpha = (epoch - self.config["param"]['T1']) / (self.config["param"]['T2'] - self.config["param"]['T1'] * self.config["param"]['af'])
        return alpha
    #=====================
    #==test
    #=====================
    def test(self):
        # add f1 here
        model = self.model
        data = self.data

        model.eval()
        logits, accs = model(), []

        # for _,mask in data('train_mask', 'test_mask'):
        for mask in [data.train_mask, data.test_mask]:
            pred = logits[mask].max(1)[1]

            acc = pred.eq(data.y[mask]).sum().item() / mask.shape[0]

            accs.append(acc)
        return accs
    #=====================
    #==train
    #=====================

    def train(self, epoch, labeled_index, target):
        model = self.model
        data = self.data

        model.train()
        #TODO here>>  why does this has 6 dim?
        untrain_model = model()

        self.optimizer.zero_grad()
        loss_output = self.pseudo_label(epoch, labeled_index, target)

        try:
            loss_output.backward()
        except UnboundLocalError as e:
            display2screen(f"epoch = {epoch}", e)
        self.optimizer.step()

        return model(), loss_output.data, untrain_model, labeled_index, target

    #=====================
    #==tuning params and hyper params
    #=====================
    def hyper_param_tuning(self):
        '''move this func to hyper_params_search module'''
        folder = f"log/gene_disease/{args.time_stamp}/{args.arch}/hp_tuning"
        if not os.path.exists(folder):
            os.makedirs(folder)

        # -- write to file
        save_path = f'{folder}/{args.emb_name}{self.curr_time}.txt'
        f = open(save_path, 'w')

        best_hp_config = {0: 0}
        write_stat = False
        # count = 0
        count = Counter().counting
        while True:

            dropout = 0.1 * random.randint(3, 8)
            lr = round(random.uniform(0.01, 0.1), 2)

            decay_coeff = round(random.randint(1, 9), 2)
            decay_power = random.randint(2, 4)

            weight_decay = decay_coeff / 10 ** decay_power

            model = Net(dropout).to(self.device) # todo make self.train(epoch) compatible with self.train(self, epoch, weighted_class, labeled_index, target)

            # optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4) # original before modify
            # best lr = 0.05 weight_decay=5e-4 ====> around 60-70 percent
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

            best_val_acc = test_acc = 0
            best_test_acc = [0]
            log_list = []

            # for epoch in range(1,201):
            for epoch in range(1, args.epochs):
                gcn_emb, loss_epoch, _ = self.train(epoch)  # todo make self.train(epoch) compatible with self.train(self, epoch, weighted_class, labeled_index, target)

                train_acc, test_acc = self.test()

                if args.verbose:
                    logging = 'Epoch: {:03d}, Train: {:.4f}, Test: {:.4f}'.format(epoch, train_acc, test_acc)
                    print(logging)

                if test_acc > best_test_acc[0]:
                    best_test_acc.insert(0, test_acc)
                    if len(best_test_acc) > 10:
                        best_test_acc.pop(-1)

            # if sum(best_test_acc)/len(best_test_acc) > list(best_hp_config.values())[0]:
            if best_test_acc[0] > list(best_hp_config.values())[0]:
                best_hp_config[f"dropout = {dropout}; lr = {lr} ; weight_decay = {weight_decay}"] = best_hp_config.pop(
                    list(best_hp_config.keys())[0])
                best_hp_config[f"dropout = {dropout}; lr = {lr} ; weight_decay = {weight_decay}"] = best_test_acc[0]
                write_stat = True
            txt = '\n'.join(["========================",
                             f"loop = {count}",
                             f"dropout = {dropout}; lr = {lr} ; weight_decay = {weight_decay}",
                             f"top 10 best acc = {best_test_acc}",
                             f"average = {sum(best_test_acc) / len(best_test_acc)}",
                             f"!!! current best config is  **{list(best_hp_config.keys())[0]}** with best_acc = {best_hp_config[list(best_hp_config.keys())[0]]} and avg_acc = {sum(best_test_acc) / len(best_test_acc)} !!!"])

            txt = txt + '\n'
            print(txt)

            # -- write to file
            if write_stat:
                print("writing to file ...")
                f.write(txt)
                write_stat = False

            count() # increase count by 1

    #=====================
    #==run GNN model
    #=====================
    def run_epochs(self):
        best_val_acc = test_acc = 0
        log_list = []

        best_epoch = {0: [0]}
        best_test_acc = [0]
        loss_hist = []
        train_acc_hist = []
        test_acc_hist = []

        labeled_index = None
        target = None
        for epoch in range(1, args.epochs):
            if epoch == 1:

                _, loss_epoch, self.gcn_emb_no_train, labeled_index, target = self.train(epoch, labeled_index, target)
            else:
                self.gcn_emb_after_classifier, loss_epoch, self.gcn_emb_output, labeled_index, target = self.train(epoch,
                                                                                                         labeled_index,
                                                                                                         target)

            loss_hist.append(loss_epoch.tolist())
            train_acc, test_acc = self.test()

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

            if args.verbose:
                print(logging)
        self.log_list = log_list
        self.best_epoch = best_epoch
        self.best_test_acc = best_test_acc
        self.loss_hist = loss_hist
        self.train_acc_hist = train_acc_hist
        self.test_acc_hist = test_acc_hist

    def run(self, modules):
        '''classify eithre train or test data'''
        data = self.data

        # model = self.model

        if args.tuning:
            self.hyper_param_tuning()
        else:
            #==========================
            #==== NOT TUNING HYPER-PARAMETERS
            #==========================
            self.run_model(modules)
            # model = Net(args.dropout).to(self.device) # old style

            # original before modify
            optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            #=====================
            #==cross validation
            #=====================
            if args.cv is not None:
                print(f"running {args.arch} with cross validation ")
                s = time.time()

                all_x = torch.cat((data.x[data.train_mask], data.x[data.test_mask]), dim=0).type(torch.float)
                all_labels = torch.cat((data.y[data.train_mask], data.y[data.test_mask]), dim=0).type(torch.long)

                from sklearn.model_selection import StratifiedKFold
                # n_splits = int(all_x.shape[0] / int(args.cv)) + 1 if all_x.shape[0] % int(args.cv) != 0 else int(
                #     all_x.shape[0] / int(args.cv))
                # assert n_splits * int(args.cv) >= all_x.shape[0], "n_splits * args.cv >= all_x.shape[0]"
                n_splits = int(args.cv)

                avg_train_metrics = None
                avg_test_metrics = None
                cv = StratifiedKFold(n_splits=n_splits)

                for i, (train_index, test_index) in enumerate(cv.split(all_x, all_labels)):
                    print(f'cv_iteration = {i}')
                    data.train_mask = train_index
                    data.test_mask = test_index
                    self.run_epochs()

                    # =====================
                    # ==get performance metrics (no plot, no print to screen)
                    # =====================
                    # logits[mask].max(1)[1]
                    # y_true, y_pred, y_score = None, average = 'micro', plot_roc_auc = True, save_path = None, file_name = None, get_avg_total = False):
                    y_train_score = self.model()[train_index]
                    y_train_pred = y_train_score.max(1)[1]

                    y_test_score = self.model()[test_index]
                    y_test_pred = y_test_score.max(1)[1]

                    if i == 7 or i ==1:
                        print("")

                    #TODO here>> change below to fit
                    save_path = f"log/gene_disease/{args.time_stamp}/classifier/{args.arch}/split={args.split}/lr={args.lr}_d={args.dropout}_wd={args.weight_decay}/report_performance/"
                    file_name = f'emb={args.emb_name}_epoch={args.epochs}_wc={args.weighted_class}.txt'

                    # --------training
                    train_measurement_metrics = performance_metrics.report_performances(data.y[train_index].numpy(),
                                                                                        y_train_pred.numpy(),
                                                                                        y_train_score.detach().numpy(),
                                                                                        save_path=save_path,
                                                                                        file_name=file_name)
                    # --------testing
                    test_measurement_metrics = performance_metrics.report_performances(data.y[test_index].numpy(),
                                                                                       y_test_pred.numpy(),
                                                                                       y_test_score .detach().numpy(),
                                                                                       save_path=save_path,
                                                                                       file_name=file_name)

                    avg_train_metrics = avg_train_metrics.add(
                        train_measurement_metrics) if avg_train_metrics is not None else train_measurement_metrics
                    avg_test_metrics = avg_test_metrics.add(
                        test_measurement_metrics) if avg_test_metrics is not None else test_measurement_metrics

                avg_train_metrics = avg_train_metrics.divide(n_splits)
                avg_test_metrics = avg_test_metrics.divide(n_splits)
                if args.report_performance:
                    print(avg_train_metrics.__repr__())
                    print('\n')
                    print(avg_test_metrics.__repr__())

                # =====================
                # ==save cross validation to file
                # =====================
                save_path = f"log/gene_disease/{args.time_stamp}/classifier/{args.arch}/cross_valid={args.cv}/lr={args.lr}_d={args.dropout}_wd={args.weight_decay}/report_performance/"
                file_name = f'emb={args.emb_name}_epoch={args.epochs}_wc={args.weighted_class}.txt'
                import os
                os.makedirs(save_path + 'train/', exist_ok=True)
                os.makedirs(save_path + 'test/', exist_ok=True)
                df = pd.DataFrame(avg_train_metrics)
                df.to_csv(save_path + 'train/' + file_name, header=True, index=False, sep='\t', mode='w')
                df = pd.DataFrame(avg_test_metrics)
                df.to_csv(save_path + 'test/' + file_name, header=True, index=False, sep='\t', mode='w')

                return avg_test_metrics.iloc[-1]

            else:
                self.run_epochs()

                # =====================
                # ==performance report
                # =====================
                # # TODO here>> make report_performance compatible with mlp
                # save_path = f"log/gene_disease/{args.time_stamp}/classifier/{args.arch}/split={args.split}/lr={args.lr}_d={args.dropout}_wd={args.weight_decay}/report_performance/"
                # file_name = f'emb={args.emb_name}_epoch={args.epochs}_wc={args.weighted_class}.txt'
                # import os
                # if not os.path.exists(save_path):
                #     os.makedirs(save_path)
                #
                # report_train = performance_metrics.report_performances(
                #     y_true=data.y[data.train_mask].numpy(),
                #     y_pred=self.model()[data.train_mask].max(1)[1].numpy(),
                #     y_score=self.model()[data.train_mask].detach().numpy(),
                #     save_path=f'{save_path}train/',
                #     file_name=file_name
                # )
                # report_test = performance_metrics.report_performances(
                #     y_true=data.y[data.test_mask].numpy(),
                #     y_pred=self.model()[data.test_mask].max(1)[1].numpy(),
                #     y_score=self.model()[data.test_mask].detach().numpy(),
                #     save_path=f'{save_path}test/',
                #     file_name=file_name
                # )
                # if args.report_performance:
                #     print(report_train)
                #     print(report_test)
                # return report_test.iloc[-1]

            # display2screen(gcn_emb_output)
            # -- print set of best accuracy and its epoch.
            # if args.verbose:
            #     print(f"!!!!! {list(self.best_epoch.keys())[0]} = {self.best_epoch[list(self.best_epoch.keys())[0]]} !!!!!!! ")



            # display2screen('here')

            # =====================
            # ==plotting
            # =====================
            save_path = f'{self.folder}'
            if args.plot_no_train:
                # =====================
                # ==using performance_metrics.report_performances
                # =====================
                file_gcn_emb = f"epoch={args.epochs}_emb={args.emb_name}_TRAIN=NO_ACC_feat={self.feat_stat}_pseudo_label={self.pseudo_label_stat}_wc={self.weighted_class}_T=[{self.T_param}]_topk={args.topk}.txt"
                img_gcn_emb = f"img/{file_gcn_emb}.png"

                # --------train report
                report_train = performance_metrics.report_performances(data.y[data.train_mask].numpy(),
                                                        self.model()[data.train_mask].max(1)[1].numpy(),
                                                        self.model()[data.train_mask].detach().numpy(),
                                                        save_path=save_path + f'report_performance/train/',
                                                        file_name=f'{file_gcn_emb}')
                # --------test_report
                report_test = performance_metrics.report_performances(data.y[data.test_mask].numpy(),
                                                        self.model()[data.test_mask].max(1)[1].numpy(),
                                                        self.model()[data.test_mask].detach().numpy(),
                                                        save_path=save_path + f'report_performance/test/',
                                                        file_name=f'{file_gcn_emb}')
                if args.report_performance:
                    print(report_train)
                    print(report_test)

                self.plot(self.gcn_emb_after_classifier, self.gcn_emb_no_train, self.loss_hist, self.train_acc_hist, self.test_acc_hist, file_gcn_emb, img_gcn_emb)

            if args.plot_train:
                file_gcn_emb = f"epoch={args.epochs}_emb={args.emb_name}_TRAIN=YES_ACC_feat={self.feat_stat}_pseudo_label={self.pseudo_label_stat}_wc={self.weighted_class}_T=[{self.T_param}]_topk={args.topk}.txt"
                img_gcn_emb = f"img/{file_gcn_emb}.png"

                # # --------train report
                # report_train = performance_metrics.report_performances(data.y[data.train_mask].numpy(),
                #                                         self.model()[data.train_mask].max(1)[1].numpy(),
                #                                         self.model()[data.train_mask].detach().numpy(),
                #                                         save_path=save_path + f'report_performance/train/',
                #                                         file_name=f'{file_gcn_emb}')
                # # --------test_report
                # report_test = performance_metrics.report_performances(data.y[data.test_mask].numpy(),
                #                                         self.model()[data.test_mask].max(1)[1].numpy(),
                #                                         self.model()[data.test_mask].detach().numpy(),
                #                                         save_path=save_path + f'report_performance/test/',
                #                                         file_name=f'{file_gcn_emb}')
                # if args.report_performance:
                #     print(report_train)
                #     print(report_test)

                self.plot(self.gcn_emb_after_classifier, self.gcn_emb_no_train, self.loss_hist, self.train_acc_hist, self.test_acc_hist, file_gcn_emb, img_gcn_emb)

            #==========================
            #== logging
            #==========================

            self.save_model_emb()
            # if args.log:
            #     self.logging(self.log_list)
            # --------train report
            # file_name = f"epoch={args.epochs}_wc={self.weighted_class}_}.txt"
            self.save_model_performance()
            print('=================')
            print('=================')


        # return gcn_emb_output

    def save_model_emb(self):
        time_stamp = args.time_stamp
        save_path = None

        save_path = r'C:\\Users\\awannaphasch2016\\PycharmProjects\\disease_node_classification\\data\\gene_disease'
        save_path = save_path + f'\\{args.time_stamp}\\processed\\embedding\\{args.added_edges_option}\\gcn\\split={args.split}\\softmax\\{self.HP}\\'
        tmp = save_path
        assert args.index is not None, "please specified index of embedding file"
        if args.top_bottom_percent_edges is not None:
            if args.stochastic_edges:
                # EMBEDDING_FOLDER=f"node2vec_emb_fullgraph_common_nodes_feat={args.common_nodes_feat}{time_stamp}_added_edges=disease_{args.edges_weight_option}_top_bottom_k={args.top_bottom_percent_edges}_mask={args.mask_edges}_stochh{args.index}.txt"
                save_path = save_path + f'top_bottom_k_stoch\\{args.top_bottom_percent_edges}\\'
            else:
                # EMBEDDING_FOLDER=f"node2vec_emb_fullgraph_common_nodes_feat={args.common_nodes_feat}{time_stamp}_added_edges=disease_{args.edges_weight_option}_top_bottom_k={args.top_bottom_percent_edges}_mask={args.mask_edges}{args.index}.txt"
                save_path = save_path + f'top_bottom_k\\{args.top_bottom_percent_edges}\\'
        if args.all_nodes_random_edges_percent is not None and args.shared_nodes_random_edges_percent is not None:
            raise ValueError(
                " Either args.all_nodes_random_edges_percent or args.shared_nodes_random_edges_percent MUST NOT be None")
        if args.all_nodes_random_edges_percent is not None:
                # EMBEDDING_FOLDER=f"node2vec_emb_fullgraph_common_nodes_feat={args.common_nodes_feat}{time_stamp}_added_edges=disease_{args.edges_weight_option}_all_nodes_random={args.all_nodes_random_edges_percent}_mask={args.mask_edges}_stochh{args.index}.txt"
                save_path = save_path + f'all_nodes_random\\{args.all_nodes_random_edges_percent}\\'
        if args.shared_nodes_random_edges_percent is not None:
                # EMBEDDING_FOLDER=f"node2vec_emb_fullgraph_common_nodes_feat={args.common_nodes_feat}{time_stamp}_added_edges=disease_{args.edges_weight_option}_shared_nodes_random={args.shared_nodes_random_edges_percent}_mask={args.mask_edges}_stochh{args.index}.txt"
                save_path = save_path + f'shared_nodes_random\\{args.shared_nodes_random_edges_percent}\\'
        if args.bottom_percent_edges is not None:
            if args.stochastic_edges:
                # EMBEDDING_FOLDER=f"node2vec_emb_fullgraph_common_nodes_feat={args.common_nodes_feat}{time_stamp}_added_edges=disease_{args.edges_weight_option}_bottom_k={args.bottom_percent_edges}_mask={args.mask_edges}_stochh{args.index}.txt"
                save_path = save_path + f'bottom_k_stoch\\{args.bottom_percent_edges}\\'

            else:
                save_path = save_path + f'bottom_k\\{args.bottom_percent_edges}\\'
                # EMBEDDING_FOLDER=f"node2vec_emb_fullgraph_common_nodes_feat={args.common_nodes_feat}{time_stamp}_added_edges=disease_{args.edges_weight_option}_bottom_k={args.bottom_percent_edges}_mask={args.mask_edges}{args.index}.txt"
        if args.top_percent_edges is not None:
            if args.stochastic_edges:
                # EMBEDDING_FOLDER=f"node2vec_emb_fullgraph_common_nodes_feat={args.common_nodes_feat}{time_stamp}_added_edges=disease_{args.edges_weight_option}_top_k={args.top_percent_edges}_mask={args.mask_edges}_stochh{args.index}.txt"
                save_path = save_path + f'top_k_stoch\\{args.top_percent_edges}\\'
            else:
                # EMBEDDING_FOLDER=f"node2vec_emb_fullgraph_common_nodes_feat={args.common_nodes_feat}{time_stamp}_added_edges=disease_{args.edges_weight_option}_top_k={args.top_percent_edges}_mask={args.mask_edges}{args.index}.txt"
                save_path = save_path + f'top_k\\{args.top_percent_edges}\\'

        # save_path = f'data/gene_disease/{args.time_stamp}/processed/embedding/node2vec/'

        assert tmp != save_path , 'please check argument in GNN.save_model_emb()'
        # =====================
        # ==save gcn embedding
        # =====================
        # TODO here>> save emb train/test
        self.weighted_class = self.weighted_class.type(torch.int).tolist()  # convert from torch to list to  be used as part of file name

        # file_gcn_emb = f"epoch={args.epochs}_emb={args.emb_name}_TRAIN=YES_ACC_feat={self.feat_stat}_pseudo_label={self.pseudo_label_stat}_wc={self.weighted_class}_T=[{self.T_param}]_topk={args.topk}.txt"
        file_gcn_emb = f'{args.index}.txt'

        # --------otuput of gcn layer1 of last epoch (hence, name 'TRAIN=YES')
        gcn_emb_output = self.model.get_emb_output().detach().numpy()
        df = pd.DataFrame(gcn_emb_output)

        import os
        # if not os.path.exists(save_path):
        #     os.mkdir(save_path)
        os.makedirs(save_path, exist_ok=True)

        print(f'save gcn_emb_output to {save_path + file_gcn_emb}')
        df.to_csv(save_path + file_gcn_emb, header=True, index=True, sep=' ', mode='w')
        print('')

    def save_model_performance(self):
        data = self.data
        # save_path = r'C:\Users\awannaphasch2016\PycharmProjects\disease_node_classification\log\gene_disease\07_14_19_46\classifier\gcn'

        time_stamp = args.time_stamp
        save_path = None
        save_path = r'C:\Users\awannaphasch2016\PycharmProjects\disease_node_classification\log\gene_disease'
        save_path = save_path + f'\\{args.time_stamp}\\classifier\\{args.added_edges_option}\\gcn\\split={args.split}\\softmax\\{self.HP}\\'
        tmp = save_path

        assert args.index is not None, "please specified index of embedding file"
        if args.top_bottom_percent_edges is not None:
            if args.stochastic_edges:
                # EMBEDDING_FOLDER=f"node2vec_emb_fullgraph_common_nodes_feat={args.common_nodes_feat}{time_stamp}_added_edges=disease_{args.edges_weight_option}_top_bottom_k={args.top_bottom_percent_edges}_mask={args.mask_edges}_stochh{args.index}.txt"
                save_path = save_path + f'top_bottom_k_stoch\\{args.top_bottom_percent_edges}\\'
            else:
                # EMBEDDING_FOLDER=f"node2vec_emb_fullgraph_common_nodes_feat={args.common_nodes_feat}{time_stamp}_added_edges=disease_{args.edges_weight_option}_top_bottom_k={args.top_bottom_percent_edges}_mask={args.mask_edges}{args.index}.txt"
                save_path = save_path + f'top_bottom_k\\{args.top_bottom_percent_edges}\\'
        if args.all_nodes_random_edges_percent is not None and args.shared_nodes_random_edges_percent is not None:
            raise ValueError(
                " Either args.all_nodes_random_edges_percent or args.shared_nodes_random_edges_percent MUST NOT be None")
        if args.all_nodes_random_edges_percent is not None:
            # EMBEDDING_FOLDER=f"node2vec_emb_fullgraph_common_nodes_feat={args.common_nodes_feat}{time_stamp}_added_edges=disease_{args.edges_weight_option}_all_nodes_random={args.all_nodes_random_edges_percent}_mask={args.mask_edges}_stochh{args.index}.txt"
            save_path = save_path + f'all_nodes_random\\{args.all_nodes_random_edges_percent}\\'
        if args.shared_nodes_random_edges_percent is not None:
            # EMBEDDING_FOLDER=f"node2vec_emb_fullgraph_common_nodes_feat={args.common_nodes_feat}{time_stamp}_added_edges=disease_{args.edges_weight_option}_shared_nodes_random={args.shared_nodes_random_edges_percent}_mask={args.mask_edges}_stochh{args.index}.txt"
            save_path = save_path + f'shared_nodes_random\\{args.shared_nodes_random_edges_percent}\\'
        if args.bottom_percent_edges is not None:
            if args.stochastic_edges:
                # EMBEDDING_FOLDER=f"node2vec_emb_fullgraph_common_nodes_feat={args.common_nodes_feat}{time_stamp}_added_edges=disease_{args.edges_weight_option}_bottom_k={args.bottom_percent_edges}_mask={args.mask_edges}_stochh{args.index}.txt"
                save_path = save_path + f'bottom_k_stoch\\{args.bottom_percent_edges}\\'

            else:
                save_path = save_path + f'bottom_k\\{args.bottom_percent_edges}\\'
                # EMBEDDING_FOLDER=f"node2vec_emb_fullgraph_common_nodes_feat={args.common_nodes_feat}{time_stamp}_added_edges=disease_{args.edges_weight_option}_bottom_k={args.bottom_percent_edges}_mask={args.mask_edges}{args.index}.txt"
        if args.top_percent_edges is not None:
            if args.stochastic_edges:
                # EMBEDDING_FOLDER=f"node2vec_emb_fullgraph_common_nodes_feat={args.common_nodes_feat}{time_stamp}_added_edges=disease_{args.edges_weight_option}_top_k={args.top_percent_edges}_mask={args.mask_edges}_stochh{args.index}.txt"
                save_path = save_path + f'top_k_stoch\\{args.top_percent_edges}\\'
            else:
                # EMBEDDING_FOLDER=f"node2vec_emb_fullgraph_common_nodes_feat={args.common_nodes_feat}{time_stamp}_added_edges=disease_{args.edges_weight_option}_top_k={args.top_percent_edges}_mask={args.mask_edges}{args.index}.txt"
                save_path = save_path + f'top_k\\{args.top_percent_edges}\\'

        assert tmp != save_path, 'please check argument in GNN.save_model_emb()'

        file_name = f"{args.index}.txt"
        #TODO here>>  add name of parameter to save_path
        # > where do i get parameter name from?
        # eg. save_path + f"all_nodes_random/{args.index}.txt"
        print(f"save train to {save_path+ 'train/' + file_name}")
        print(f"save test to {save_path+ 'test/' + file_name}")
        report_train = performance_metrics.report_performances(y_true=data.y[data.train_mask].numpy(),
                                                               y_pred=self.model()[data.train_mask].max(1)[1].numpy(),
                                                               y_score=self.model()[data.train_mask].detach().numpy(),
                                                               save_path=save_path + f'train/',
                                                               file_name=f'{file_name}')
        # --------test_report
        report_test = performance_metrics.report_performances(y_true=data.y[data.test_mask].numpy(),
                                                              y_pred=self.model()[data.test_mask].max(1)[1].numpy(),
                                                              y_score=self.model()[data.test_mask].detach().numpy(),
                                                              save_path=save_path + f'test/',
                                                              file_name=f'{file_name}')
        # report_performances(y_true, y_pred, y_score=None, average='micro', save_path=None, file_name=None,
        #                     get_avg_total=False):
        if args.report_performance:
            print(report_train)
            print(report_test)
        print()
