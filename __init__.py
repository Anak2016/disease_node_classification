"to make disease_node_classification a modulle"
from anak import *
from arg_parser import *
from my_utils import *
from all_models import baseline, embedding
import all_datasets

if __name__ == "__main__":

    # ==============================
    # == data manipulation + labeling
    # ==============================
    # create_copd_label_content(time_stamp=time_stamp, sep=',')
    # create_copd_label_edges(time_stamp=time_stamp, sep=',')
    # bine_copd_label(time_stamp=time_stamp)

    #=====================
    #==datasets
    #=====================

    #--------cora
    # create_pytorch_dataset()

    #--------copd

    copd = all_datasets.Copd(path=args.copd_path, data=args.copd_data, time_stamp=args.time_stamp, undirected=not args.directed)
    # copd.create_rep_dataset()

    #=====================
    #==report dataste characteristics
    #=====================

    #--------copd report
    # copd.report_copd_characters(plot=args.plot_reports, verbose=args.verbose)

    #=====================
    #==run all data preprocessing
    #=====================
    #--------create data to be used as args to create Dataset object
    x, copd, edge_index, y = preprocessing.data_preprocessing(dataset=copd) # type of embedding and type of datasets is chosen here

    #=====================
    #==Copd_geometric_dataset
    #=====================
    #--------create Dataset object to be used with torch model
    copd_geometric_dataset = all_datasets.GeometricDataset(copd, x=x,edges_index=edge_index,y=y, split=args.split, undirected=not args.directed )
    # display2screen(copd_geometric_dataset.is_undirected())
    param = {
            #Pseudo-Label
            'T1':int(args.t1_t2_alpha[0]),
            'T2':int(args.t1_t2_alpha[1]),
            'af':float(args.t1_t2_alpha[2])}

    # ====================
    # == run models
    # ====================
    def run_model():
        if args.run_node2vec:
            run_node2vec(copd, copd_geometric_dataset, args.time_stamp)

        if args.run_svm:
            if args.emb_name == "no_feat" and args.common_nodes_feat != "no":
                # train_input, test_input = preprocessing.create_common_nodes_as_features(copd, copd_geometric_dataset)
                all_x_input = preprocessing.create_common_nodes_as_features(copd, copd_geometric_dataset, use=args.common_nodes_feat)

                # -- normalize features vector
                all_x_input = preprocessing.normalize_features(csr_matrix(all_x_input))

                config = {
                    "train_input": torch.tensor(all_x_input[copd_geometric_dataset.train_mask]),
                    'test_input': torch.tensor(all_x_input[copd_geometric_dataset.test_mask]),
                    "train_label": copd_geometric_dataset.y[copd_geometric_dataset.train_mask],
                    "test_label": copd_geometric_dataset.y[copd_geometric_dataset.test_mask]
                }
            elif args.emb_path is not None or args.emb_name in ['no_feat', "node2vec", 'attentionwalk', 'bine' ]:
                # --------identity matrix or valid emb_name
                config = {
                    "train_input": copd_geometric_dataset.x[copd_geometric_dataset.train_mask],
                    'test_input': copd_geometric_dataset.x[copd_geometric_dataset.test_mask],
                    "train_label": copd_geometric_dataset.y[copd_geometric_dataset.train_mask],
                    "test_label": copd_geometric_dataset.y[copd_geometric_dataset.test_mask]
                }
            else:
                raise ValueError('provided emb_names mayb incorrect or args.common_nodes_feat is typed incorrectly')

            svm_runtime = timer(baseline.svm, data=copd_geometric_dataset, config=config, verbose=args.verbose)
            print(f'total running time of baseline.svm == {svm_runtime}')

        if args.run_rf:
            if args.emb_name == "no_feat" and args.common_nodes_feat != 'no':

                all_x_input = preprocessing.create_common_nodes_as_features(copd, copd_geometric_dataset, use=args.common_nodes_feat)

                # -- normalize features vector
                all_x_input = preprocessing.normalize_features(csr_matrix(all_x_input))

                config = {
                    "train_input": torch.tensor(all_x_input[copd_geometric_dataset.train_mask]),
                    'test_input': torch.tensor(all_x_input[copd_geometric_dataset.test_mask]),
                    "train_label": copd_geometric_dataset.y[copd_geometric_dataset.train_mask],
                    "test_label": copd_geometric_dataset.y[copd_geometric_dataset.test_mask]
                }
            elif args.emb_path is not None or args.emb_name in ['no_feat', "node2vec", 'attentionwalk', 'bine']:
                # --------identity matrix or valid emb_name
                config = {
                    "train_input": copd_geometric_dataset.x[copd_geometric_dataset.train_mask],
                    'test_input': copd_geometric_dataset.x[copd_geometric_dataset.test_mask],
                    "train_label": copd_geometric_dataset.y[copd_geometric_dataset.train_mask],
                    "test_label": copd_geometric_dataset.y[copd_geometric_dataset.test_mask]
                }
            else:
                raise ValueError('provided emb_names mayb incorrect or args.common_nodes_feat is typed incorrectly')

            rf_runtime = timer(baseline.random_forest, data=copd_geometric_dataset, config=config, evaluate=True)
            print(f'total running time of baseline.rf == {rf_runtime}')


        if args.run_gnn:

            config = {
                "param" :{ # Pseudo-Label
                    'T1': int(args.t1_t2_alpha[0]),
                    'T2': int(args.t1_t2_alpha[1]),
                    'af': float(args.t1_t2_alpha[2])
                }
            }
            # gcn_runtime = timer(embedding.run_GCN,data=copd_geometric_dataset, lr=args.lr,weight_decay=args.weight_decay )
            gcn_runtime = timer(embedding.GNN(data=copd_geometric_dataset, config=config).run)

            print(f'total running time of baseline.gnn == {gcn_runtime}')

        if args.run_mlp:
            if args.emb_name == "no_feat" and args.common_nodes_feat != 'no':
                all_x_input = preprocessing.create_common_nodes_as_features(copd, copd_geometric_dataset,use=args.common_nodes_feat)

                # -- normalize features vector
                all_x_input = preprocessing.normalize_features(csr_matrix(all_x_input))
                # display2screen(all_x_input.shape)
                config = {
                    "data": copd,
                    "label": y.numpy(),  # tensor
                    "train_input": torch.tensor(all_x_input[copd_geometric_dataset.train_mask]),
                    'test_input': torch.tensor(all_x_input[copd_geometric_dataset.test_mask]),
                    "train_label": y.numpy()[copd_geometric_dataset.train_mask],
                    "test_label": y.numpy()[copd_geometric_dataset.test_mask],
                    # change value of hidden_layers to be used in nn.sequential
                    'sequential_layers': [
                        nn.Linear(2996, 512),
                        nn.ReLU(),
                        nn.Linear(512, 64),
                        nn.ReLU(),
                        nn.Linear(64, len(copd.labels2idx().keys())),
                        nn.LogSoftmax(dim=1)
                    ],
                    "epochs": 200,
                    "args": args,
                    "param": param
                }
            elif args.emb_path is not None:
                '''emb_path must directed to any of the emb options '''
                config = {
                    "data": copd,
                    "label": y.numpy(),  # tensor
                    "train_input": copd_geometric_dataset.x[copd_geometric_dataset.train_mask],
                    'test_input': copd_geometric_dataset.x[copd_geometric_dataset.test_mask],
                    "train_label": copd_geometric_dataset.y[copd_geometric_dataset.train_mask],
                    "test_label": copd_geometric_dataset.y[copd_geometric_dataset.test_mask],
                    # "hidden_layers": [64, 64, 128, 16, len(copd.labels2idx().keys())],
                    'sequential_layers': [ # layers and its dimension have to changes depends on how emb that emb_path is directed to
                        nn.Linear(16, len(copd.labels2idx().keys())),
                        nn.LogSoftmax(dim=1)
                    ],
                    "epochs": 200,
                    "args": args,
                    "param": param
                }

            elif args.emb_name in ["node2vec"]:
                #--------valid emb_name
                config = {
                    "data": copd,
                    "label": y.numpy(),  # tensor
                    "train_input": copd_geometric_dataset.x[copd_geometric_dataset.train_mask],
                    'test_input': copd_geometric_dataset.x[copd_geometric_dataset.test_mask],
                    "train_label": copd_geometric_dataset.y[copd_geometric_dataset.train_mask],
                    "test_label": copd_geometric_dataset.y[copd_geometric_dataset.test_mask],
                    # "hidden_layers": [64, 64, 128, 16, len(copd.labels2idx().keys())],
                    'sequential_layers':[
                        nn.Linear(64, 32),
                        nn.ReLU(),
                        nn.Linear(32, 16),
                        nn.ReLU(),
                        nn.Linear(16, len(copd.labels2idx().keys())),
                        nn.LogSoftmax(dim=1)
                    ],
                    "epochs": 200,
                    "args": args,
                    "param": param
                }
            elif args.emb_name in ['bine']:
                # --------valid emb_name
                config = {
                    "data": copd,
                    "label": y.numpy(),  # tensor
                    "train_input": copd_geometric_dataset.x[copd_geometric_dataset.train_mask],
                    'test_input': copd_geometric_dataset.x[copd_geometric_dataset.test_mask],
                    "train_label": copd_geometric_dataset.y[copd_geometric_dataset.train_mask],
                    "test_label": copd_geometric_dataset.y[copd_geometric_dataset.test_mask],
                    # "hidden_layers": [64, 64, 128, 16, len(copd.labels2idx().keys())],
                    'sequential_layers': [
                        nn.Linear(128, 64),
                        nn.ReLU(),
                        nn.Linear(64, 16),
                        nn.ReLU(),
                        nn.Linear(16, len(copd.labels2idx().keys())),
                        nn.LogSoftmax(dim=1)
                    ],
                    "epochs": 200,
                    "args": args,
                    "param": param
                }
            elif args.emb_name == 'no_feat':
                # --------identity matrix
                config = {
                    "data": copd,
                    "label": y.numpy(),  # tensor
                    "train_input": copd_geometric_dataset.x[copd_geometric_dataset.train_mask],
                    'test_input': copd_geometric_dataset.x[copd_geometric_dataset.test_mask],
                    "train_label": copd_geometric_dataset.y[copd_geometric_dataset.train_mask],
                    "test_label": copd_geometric_dataset.y[copd_geometric_dataset.test_mask],
                    # "hidden_layers": [2996, 2996, 128, 16, len(copd.labels2idx().keys())],
                    'sequential_layers': [
                        nn.Linear(2996, 128),
                        nn.ReLU(),
                        nn.Linear(128, 16),
                        nn.ReLU(),
                        nn.Linear(16, len(copd.labels2idx().keys())),
                        nn.LogSoftmax(dim=1)
                    ],
                    "epochs": 200,
                    "args": args,
                    "param": param
                }
            else:
                raise ValueError('provided emb_names mayb incorrect or args.common_nodes_feat is typed incorrectly')

            # baseline.mlp(data=copd_geometric_dataset, config=config)
            mlp_runtime = timer(baseline.mlp, data=copd_geometric_dataset, config=config)
            print(f'total running time of baseline.mlp == {mlp_runtime}')

        if args.run_lr:
            # train_input, test_input = preprocessing.create_common_nodes_as_features(copd, copd_geometric_dataset)

            if args.emb_name == "no_feat" and args.common_nodes_feat != 'no':
                all_x_input = preprocessing.create_common_nodes_as_features(copd, copd_geometric_dataset, use=args.common_nodes_feat)

                # -- normalize features vector
                all_x_input = preprocessing.normalize_features(csr_matrix(all_x_input))

                config = {
                    "train_input": torch.tensor(all_x_input[copd_geometric_dataset.train_mask]),
                    'test_input': torch.tensor(all_x_input[copd_geometric_dataset.test_mask]),
                    "train_label": copd_geometric_dataset.y[copd_geometric_dataset.train_mask],
                    "test_label": copd_geometric_dataset.y[copd_geometric_dataset.test_mask]
                }
            elif args.emb_path is not None or args.emb_name in ['no_feat', "node2vec", 'attentionwalk', 'bine']:
                # --------identity matrix or valid emb_name
                config = {
                    "train_input": copd_geometric_dataset.x[copd_geometric_dataset.train_mask],
                    'test_input': copd_geometric_dataset.x[copd_geometric_dataset.test_mask],
                    "train_label": copd_geometric_dataset.y[copd_geometric_dataset.train_mask],
                    "test_label": copd_geometric_dataset.y[copd_geometric_dataset.test_mask]
                }
            else:
                raise ValueError('provided emb_names mayb incorrect or args.common_nodes_feat is typed incorrectly')

            lr_runtime = timer(baseline.logistic_regression,config, emb_name=args.emb_name)
            print(f'total running time of baseline.lr == {lr_runtime}')

    if args.check_condition is not None:
        #--------check same condision for all base model
        for model in args.check_condition:
            if model == "all":
                args.run_svm = True
                args.run_lr = True
                args.run_rf = True
                args.run_mlp = True
                args.run_gnn = True
            if model == "svm":
                args.run_svm = True
            if model == "lr":
                args.run_lr = True
            if model == "rf":
                args.run_rf = True
            if model == "gnn":
                args.run_gnn = True
            if model == 'mlp':
                args.run_mlp = True
            assert model in ['all', 'svm','lr','rf', 'gnn','mlp'], "selected model to be check is not supported"

        rt = timer(run_model)
        print(f'\ntotal running time of args.check_condition == {rt}')
    else:
        #=====================
        #==run a model
        #=====================
        run_model()

    # if args.run_gcn_on_disease_graph:
    #     G = nx.Graph()
    #
    #     # tmp = [(i,j) for (i,j) in zip(edge_index[0].numpy(), edge_index[1].numpy()) if int(i) == 2995 or int(j) == 2995 ]
    #     edges = [[i, j] if int(i) < len(copd.disease2idx().values()) else (j, i) for (i, j) in
    #              zip(edge_index[0].numpy(), edge_index[1].numpy())]
    #     edges = list(map(lambda t: (int(t[0]), int(t[1])), edges))
    #
    #     edges = sorted(edges, reverse=False, key=lambda t: t[0])
    #
    #     adj_list = create_adj_list(edges)
    #     # -- create genes as onehot
    #     onehot_genes = preprocessing.create_onehot(adj_list, edges)
    #
    #     G.add_edges_from(edges)
    #     input = nx.adjacency_matrix(G).todense()[:len(copd.disease2idx().keys()), :]
    #     tmp = nx.adjacency_matrix(G).todense()[:len(copd.disease2idx().keys()), :]
    #
    #
    #
    #     config = {
    #         "data": copd,
    #         "input": onehot_genes,  # dictionary
    #         "label": y.numpy(),
    #         "train_mask": copd_geometric_dataset.train_mask,
    #         "test_mask": copd_geometric_dataset.test_mask,
    #         "emb": x.numpy(),
    #         "hidden_layers": [2996, 2996, 128, 16, len(copd.labels2idx().keys())],
    #         "epochs": 200,
    #         "args": args,
    #         "param": param
    #     }
    #     run_gcn_on_disease_graph(config, emb_name=args.emb_name)
