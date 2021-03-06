"to make disease_node_classification a modulle"
from anak import *
from arg_parser import *
from my_utils import *
from all_models import baseline, embedding
import all_datasets

def set_config(copd, copd_geometric_dataset, *arguments, **kwargs):

    # if args.emb_name == 'gnn':
    #     all_x_input, edges_weight, edges = preprocessing.create_common_nodes_as_features(copd, copd_geometric_dataset,
    #                                                                                      *arguments, **kwargs)
    #     all_x_input = preprocessing.normalize_features(csr_matrix(all_x_input))
    #     copd_geometric_dataset.x = all_x_input
    #
    #     #=====================
    #     #==running gnn to get embedding
    #     #=====================
    #     args.hidden = 16
    #     from all_models import embedding
    #     # func_kwargs, copd_geometric_dataset, _ = get_config(model['name'], copd, emb_name=model['emb_name'])
    #     config = {
    #         "param": {  # Pseudo-Label
    #             'T1': int(args.t1_t2_alpha[0]),
    #             'T2': int(args.t1_t2_alpha[1]),
    #             'af': float(args.t1_t2_alpha[2])
    #         }
    #     }
    #     #TODO here>> make sure that nodes are in ordered in gcn
    #     #TODO here>> does gcn use edges weight? if it does, comment out the normalized_features
    #     copd_geometric_dataset.x = embedding.GNN(data=copd_geometric_dataset, config=config).run()  # gnn as embedding
    #
    #     #=====================
    #     #==config for classifier
    #     #=====================
    #
    #     config = {
    #         "train_input": torch.tensor(copd_geometric_dataset.x[copd_geometric_dataset.train_mask]),
    #         'test_input': torch.tensor(copd_geometric_dataset.x[copd_geometric_dataset.test_mask]),
    #         "train_label": copd_geometric_dataset.y[copd_geometric_dataset.train_mask],
    #         "test_label": copd_geometric_dataset.y[copd_geometric_dataset.test_mask]
    #     }
    #
    #     return config, copd_geometric_dataset, all_x_input

    if args.common_nodes_feat != "no":
        #TODO here>> x is self loop
        all_x_input, edges_weight, edges = preprocessing.create_common_nodes_as_features(copd, copd_geometric_dataset,
                                                                                         *arguments, **kwargs)
        all_x_input = preprocessing.normalize_features(csr_matrix(all_x_input))
        copd_geometric_dataset.x = torch.tensor(all_x_input, dtype=torch.float)
        config = {
            "train_input": torch.tensor(all_x_input[copd_geometric_dataset.train_mask]),
            'test_input': torch.tensor(all_x_input[copd_geometric_dataset.test_mask]),
            "train_label": copd_geometric_dataset.y[copd_geometric_dataset.train_mask],
            "test_label": copd_geometric_dataset.y[copd_geometric_dataset.test_mask]
        }
        return config, copd_geometric_dataset, all_x_input

    if args.emb_path is not None or args.emb_name in ['no_feat', "node2vec", 'attentionwalk', 'bine']:
        # --------identity matrix or valid emb_name
        config = {
            "train_input": copd_geometric_dataset.x[copd_geometric_dataset.train_mask],
            'test_input': copd_geometric_dataset.x[copd_geometric_dataset.test_mask],
            "train_label": copd_geometric_dataset.y[copd_geometric_dataset.train_mask],
            "test_label": copd_geometric_dataset.y[copd_geometric_dataset.test_mask]
        }
        return config, copd_geometric_dataset, None

def get_config( model_name, copd, emb_name =None, emb_path=None, *arguments, **kwargs):

    #--------create data to be used as args to create Dataset object
    #TODO here>> figure out how to add edges between genes in data_preprocessing.

    x, copd,  edge_index, y = preprocessing.data_preprocessing(dataset=copd)
    copd_geometric_dataset = all_datasets.GeometricDataset(copd, x=x, edges_index=edge_index, y=y,
                                                           split=args.split, undirected=not args.directed)


    # TODO here>> create_common_nodes_as_features and normalized_features should be before copd_geometric_dataset because it preprocess data
    if model_name in ['gnn','node2vec']:
        config, copd_geometric_dataset, all_x_input = set_config(copd, copd_geometric_dataset, *arguments, **kwargs)
        return config, copd_geometric_dataset, all_x_input

    elif model_name == 'svm':
        config, copd_geometric_dataset, all_x_input =  set_config(copd, copd_geometric_dataset, *arguments, **kwargs)
        return config, copd_geometric_dataset, all_x_input
        # if args.common_nodes_feat != "no":
        #     # train_input, test_input = preprocessing.create_common_nodes_as_features(copd, copd_geometric_dataset)
        #     # all_x_input, edges_weight, edges = preprocessing.create_common_nodes_as_features(copd, copd_geometric_dataset, used_nodes='gene', edges_weight_option= args.edges_weight_option)
        #
        #     copd_geometric_dataset.x = all_x_input
        #     config = {
        #         "train_input": torch.tensor(all_x_input[copd_geometric_dataset.train_mask]),
        #         'test_input': torch.tensor(all_x_input[copd_geometric_dataset.test_mask]),
        #         "train_label": copd_geometric_dataset.y[copd_geometric_dataset.train_mask],
        #         "test_label": copd_geometric_dataset.y[copd_geometric_dataset.test_mask]
        #     }
        #     return config, copd_geometric_dataset, all_x_input
        #
        # if args.emb_path is not None or args.emb_name in ['no_feat', "node2vec", 'attentionwalk', 'bine' ]:
        #     # --------create data to be used as args to create Dataset object
        #     # x, copd, edge_index, y = preprocessing.data_preprocessing(dataset=copd)
        #     #
        #     # copd_geometric_dataset = all_datasets.GeometricDataset(copd, x=x, edges_index=edge_index, y=y,
        #     #                                                        split=args.split, undirected=not args.directed)
        #     # --------identity matrix or valid emb_name
        #     config = {
        #         "train_input": copd_geometric_dataset.x[copd_geometric_dataset.train_mask],
        #         'test_input': copd_geometric_dataset.x[copd_geometric_dataset.test_mask],
        #         "train_label": copd_geometric_dataset.y[copd_geometric_dataset.train_mask],
        #         "test_label": copd_geometric_dataset.y[copd_geometric_dataset.test_mask]
        #     }
        #     # print(copd_geometric_dataset.x)
        #     return config, copd_geometric_dataset, None

    if model_name == 'rf':
        config, copd_geometric_dataset, all_x_input =  set_config(copd, copd_geometric_dataset, *arguments, **kwargs)
        return config, copd_geometric_dataset, all_x_input

    if model_name == 'lr':
        config, copd_geometric_dataset, all_x_input =  set_config(copd, copd_geometric_dataset, *arguments, **kwargs)
        return config, copd_geometric_dataset, all_x_input

    if model_name == 'mlp':
        param = {
            # Pseudo-Label
            'T1': int(args.t1_t2_alpha[0]),
            'T2': int(args.t1_t2_alpha[1]),
            'af': float(args.t1_t2_alpha[2])}
        if args.emb_name == "no_feat" and args.common_nodes_feat != 'no':

            all_x_input, edges_weight, edges = preprocessing.create_common_nodes_as_features(copd,
                                                                                             copd_geometric_dataset,
                                                                                             *arguments, **kwargs)
            all_x_input = preprocessing.normalize_features(csr_matrix(all_x_input))
            copd_geometric_dataset.x = all_x_input

            # display2screen(all_x_input.shape)
            config = {
                "data": copd,
                "label": y.numpy(),  # tensor
                "train_input": torch.tensor(all_x_input[copd_geometric_dataset.train_mask]),
                'test_input': torch.tensor(all_x_input[copd_geometric_dataset.test_mask]),
                "train_label": y.numpy()[copd_geometric_dataset.train_mask],
                "test_label": y.numpy()[copd_geometric_dataset.test_mask],
                # change value of hidden_layers to be used in nn.sequential
                'sequential_layers': [nn.Linear(2996, 512),
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
            return config, copd_geometric_dataset, all_x_input

        elif args.emb_path is not None or args.emb_name in ['no_feat', "node2vec", 'attentionwalk', 'bine']:
            '''emb_path must directed to any of the emb options '''
            # copd_geometric_dataset.x = torch.from_numpy(copd_geometric_dataset.x)
            # copd_geometric_dataset.y = torch.from_numpy(copd_geometric_dataset.y)
            # copd_geometric_dataset.train_mask = torch.from_numpy(copd_geometric_dataset.train_mask)
            # copd_geometric_dataset.test_mask = torch.from_numpy(copd_geometric_dataset.test_mask)

            config = {
                "data": copd,
                "label": y.numpy(),  # tensor
                "train_input": copd_geometric_dataset.x[copd_geometric_dataset.train_mask],
                'test_input': copd_geometric_dataset.x[copd_geometric_dataset.test_mask],
                "train_label": copd_geometric_dataset.y[copd_geometric_dataset.train_mask],
                "test_label": copd_geometric_dataset.y[copd_geometric_dataset.test_mask],
                # "hidden_layers": [64, 64, 128, 16, len(copd.labels2idx().keys())],
                'sequential_layers': [
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
            return config, copd_geometric_dataset, None

        else:
            raise ValueError('provided emb_names mayb incorrect or args.common_nodes_feat is typed incorrectly')

    raise ValueError('provided emb_names mayb incorrect or args.common_nodes_feat is typed incorrectly')



def run_ensemble(copd, config=None):
    assert config is not None , 'config must not be None'

    model_predict = []
    model_predict_prob = []

    # TODO here>> change it so that ensemble run through each model
    # > ensemble contains n number of train models
    # > ensemble then feed data for model to be predicted
    # > collect prediction from each data and selected each predicion that have the most vote.

    # args.added_edges_option = 'longest_path'
    # args.added_edges_option = 'shared_gene'

    for name,model  in config.items():

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        #=====================
        #==code below is bad, I can fix it easily by create "preprocessing" class and group all preprocess method into it
        #=====================
        # # reassign args of jaccard_coeff function
        if model.get('edges_selection',None) is not None and model.get('emb_path', None) is not None:
            raise ValueError('only edges_selection or emb_path should be selected')
        # copd_geometric_dataset = None
        if model.get('edges_selection',None):
            args.common_nodes_feat = model['edges_selection']['common_nodes_feat']

            if args.common_nodes_feat != "no":

                args.stochastic_edges     = model['edges_selection']['stochastic_edges']
                args.mask_edges           = model['edges_selection']['mask_edges']
                args.edges_weight_limit   = model['edges_selection']['edges_weight_limit']
                args.self_loop            = model['edges_selection']['self_loop']
                args.edges_weight_percent = model['edges_selection']['edges_weight_percent']
                args.top_percent_edges    = model['edges_selection']['top_percent_edges']
                args.bottom_percent_edges    = model['edges_selection']['bottom_percent_edges']
                args.shared_nodes_random_edges_percent    = model['edges_selection']['shared_nodes_random_edges_percent']
                args.all_nodes_random_edges_percent    = model['edges_selection']['all_nodes_random_edges_percent']
                if args.all_nodes_random_edges_percent is not None:
                    args.percent = args.all_nodes_random_edges_percent
                elif args.shared_nodes_random_edges_percent is not None:
                    args.percent = args.all_nodes_random_edges_percent
                elif args.bottom_percent_edges is not None:
                    args.percent = args.bottom_percent_edges
                elif args.top_percent_edges is not None:
                    args.percent = args.top_percent_edges
                else:
                    raise ValueError('no percent specified')
                func_kwargs, copd_geometric_dataset, _ = get_config(model['name'], copd,
                                                                    used_nodes=args.common_nodes_feat,
                                                                    edges_weight_option=model['edges_selection']['edges_weight_option'],
                                                                    added_edges_option=args.added_edge_option)
            else:
                raise ValueError("error in run_ensemble: common_nodes_feat is None")
        elif model.get('emb_path', None): # done
            args.emb_path = model.get('emb_path', None)
            func_kwargs, copd_geometric_dataset, _ = get_config(model['name'], copd,  emb_path=model['emb_path'],added_edges_option=args.added_edge_option)
        elif model.get('emb_name', None): # todo need to check further
            args.emb_name = model.get('emb_name', 'no_feat')
            func_kwargs, copd_geometric_dataset, _ = get_config(model['name'], copd, emb_name=model['emb_name'],added_edges_option=args.added_edge_option)

        else:
            raise ValueError('func_kwargs is None')

        args.cv = None
        #=====================
        #==get trained model
        #=====================
        func = model['func']['model']

        performance, trained_model = func(copd_geometric_dataset, func_kwargs) # pred must be real prediction

        print(performance.to_dict())
        #TODO here>> why predict_proba does not give the same resutl as predict
        test_input = copd_geometric_dataset.x[copd_geometric_dataset.test_mask]
        test_labels = copd_geometric_dataset.y[copd_geometric_dataset.test_mask]
        if model['name'] == 'svm':
            # pred_proba = trained_model.predict_proba(test_input)
            pred_proba = trained_model.decision_function(test_input)
            pred = pred_proba.argmax(1)

            # pred = pred_proba.argmax(1)
            model_predict_prob.append(pred_proba)
            model_predict.append(pred)
        elif model['name'] =='lr':
            pred_proba = trained_model.decision_function(test_input)
            # pred_proba = trained_model.predict_proba(test_input)
            # pred = trained_model.predict(test_input)
            pred = pred_proba.argmax(1)

            model_predict_prob.append(pred_proba)
            model_predict.append(pred)
        elif model['name'] == 'rf':
            # pred_proba = trained_model.decision_function(test_input)
            pred_proba = trained_model.predict_proba(test_input)
            pred = pred_proba.argmax(1)
            # pred = trained_model.predict(test_input)

            model_predict_prob.append(pred_proba)
            model_predict.append(pred)
        elif model['name'] == 'mlp':
            # TODO here>> how to get probability of mlp
            # i need pred_prob and pred
            pred_proba = trained_model(test_input)
            pred = pred_proba.max(1)[1].numpy()
            pred_proba = pred_proba.detach().numpy()

            model_predict_prob.append(pred_proba)
            model_predict.append(pred)

        else:
            raise ValueError('model name is not supported for incorrectly typed')



    model_predict = np.array(model_predict)
    model_predict_prob = np.array(model_predict_prob)

    #=====================
    #==select most vote
    #=====================
    # collect voting output along axis 0 # figure out how??
    from scipy.stats import mode
    if model_predict.shape[0] > 1:
        ensemble_pred_prob = model_predict_prob.mean(axis = 0) # check dimension
    else:
        ensemble_pred_prob = model_predict_prob[0]

    if model_predict.shape[0] > 1:
        ensemble_pred = np.apply_along_axis(mode, 0, model_predict)[0][0]
    else:
        ensemble_pred = model_predict[0]
    # ensemble_pred_proba = np.apply_along_axis(mode, 0, model_predict)[0][0]
    # ensemble_pred = ensemble_pred.reshape(2,-1)
    # ensemble_pred = [i[0][0] for i in ensemble.tolist()]

    #TODO here>> check test label = 0, no prediction => AUC should be zero ( Do i understnad it correct or value of AUC where there is no pred could be more than 0 )
    #=====================
    #==create y_score for
    #=====================
    # emsemble_pred

    #=====================
    #==print models performance
    #=====================
    if args.embedding_name == 'node2vec':
        tmp = args.emb_path.split("\\")[10:]
        args.percent = tmp[-2]
        # file_name = '_'.join(tmp) # eg node2vec_all_nodes_random_0.05_0.txt
        file_name = tmp[-1]  # eg node2vec_all_nodes_random_0.05_0.txt
        folder = args.emb_path.split('\\')[-4]
        folder = folder + '\\' + args.emb_path.split('\\')[-3]
        folder = folder + '\\' + f"{args.percent}/ensemble/"

        # tmp = args.emb_path.split("\\")[10:]
        # args.percent = tmp[-2]
        # file_name = tmp[-1]  # eg node2vec_all_nodes_random_0.05_0.txt
        # folder = args.emb_path.split('\\')[10]
        # folder = folder + '\\' + args.emb_path.split('\\')[11]
        # folder = folder + '\\' + args.emb_path.split('\\')[12]
        # folder = folder + '\\' + f"{args.percent}/ensemble/"

    elif args.embedding_name == 'gcn':
        folder = '\\'.join(args.emb_path.split('\\')[-6:-1]) + '\\' + f"{args.percent}/ensemble/"
        file_name = args.emb_path.split("\\")[-1]
    else:
        raise ValueError('in all_models > baseline > models > svm() > else > else ')

    import performance_metrics
    # save_path = r'C:\Users\awannaphasch2016\PycharmProjects\disease_node_classification\output\gene_disease\ensemble/'
    if args.added_edge_option == 'longest_path':
        save_path = f"log/gene_disease/{args.time_stamp}/classifier/{args.added_edge_option}/{model['name']}/split={args.split}/"
    elif args.added_edge_option in ['shared_gene', 'no_shared_gene']:
        save_path = f"log/gene_disease/{args.time_stamp}/classifier/{args.added_edge_option}/{model['name']}/split={args.split}/report_performance/"
    else:
        raise ValueError('args.added_edges_option is wrong')

    # tmp = args.emb_path.split("\\")
    # folder = tmp[-4] + '\\' + tmp[-3] + '\\' + tmp[-2] + '\\' +  "ensemble/"
    # save_path = save_path + folder
    # tmp = '_'.join(tmp[-3:-1])
    # file_name = f'{tmp}.txt'

    save_path = save_path + folder
    print(f'folder = {folder}')
    print(save_path + f'{file_name} ')

    print('-----ensemble report')
    report = performance_metrics.report_performances(
        y_true= test_labels.numpy(),
        y_pred=ensemble_pred,
        y_score=ensemble_pred_prob, # if this is None, Roc is not shown
        save_path=f'{save_path}',
        file_name=file_name
    )
    print(report)


def run_main():
    # ==============================
    # == data manipulation + labeling
    # ==============================
    # create_copd_label_content(time_stamp=time_stamp, sep=',')
    # create_copd_label_edges(time_stamp=time_stamp, sep=',')
    # bine_copd_label(time_stamp=time_stamp)
    #=====================
    #==args setting
    #=====================
    # if (not args.ensemble) and args.emb_name == 'no_feat' and args.emb_path is None:
    #     if args.edges_weight_limit is not None and args.edges_weight_percent is not None and args.top_percent_edges is not None:
    #         raise ValueError('only edges_weight_limit or edges_weight_percent or top_percent_edges can be used at a time')
    #     if args.edges_weight_limit is not None and args.edges_weight_percent is not None :
    #         raise ValueError('only edges_weight_limit or edges_weight_percent can be used at a time')
    #     if args.edges_weight_limit is not None and args.top_percent_edges is not None :
    #         raise ValueError('only edges_weight_limit or top_percent_edges can be used at a time')
    #     if args.edges_weight_percent is not None and args.top_percent_edges is not None :
    #         raise ValueError('only edges_weight_percent or top_percent_edges can be used at a time')
    #     if args.edges_weight_limit is  None and args.edges_weight_percent is None and args.top_percent_edges is None:
    #         raise ValueError('you must set edges_weight_liit or edges_weight_percent or top_percent_edges ')


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
    if not args.ensemble:
        #--------create data to be used as args to create Dataset object
        x, copd, edge_index, y = preprocessing.data_preprocessing(dataset=copd) # type of embedding and type of datasets is chosen here

        #--------add_edges_weight


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
    # MODEL_PERFORMANCE = {i:{} for i in ['svm','rf','lr','gnn','mlp']}
    MODEL_PERFORMANCE = {}

    def run_model(num_run = 1):
        '''

        :param num_run: number of time that experiment will be repeat
        :return:
        '''

        if args.run_node2vec:
            run_node2vec(copd, copd_geometric_dataset, args.time_stamp)

        if args.run_svm:

            if args.emb_name == "no_feat" and args.common_nodes_feat != "no":
                # train_input, test_input = preprocessing.create_common_nodes_as_features(copd, copd_geometric_dataset)
                all_x_input, edges_weight, edges = preprocessing.create_common_nodes_as_features(copd, copd_geometric_dataset, used_nodes='gene', edges_weight_option= args.edges_weight_option)

                all_x_input = preprocessing.normalize_features(csr_matrix(all_x_input))
                copd_geometric_dataset.x = all_x_input
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

            # config = get_config('svm',copd_geometric_dataset, used_nodes='gene', edges_weight_option= args.edges_weight_option)

            # repeat_model_run(copd_geometric_dataset, name=f'svm_{args.emb_name}', num_run=num_run, model=baseline.svm, data=copd_geometric_dataset, config=config, verbose=args.verbose)
            # repeat_model_run(copd_geometric_dataset, name=f'svm_{args.emb_name}', num_run=num_run, model=baseline.svm, data=copd_geometric_dataset, config=config, verbose=args.verbose)
            report_avg, _ = baseline.svm(data=copd_geometric_dataset, config=config, verbose=args.verbose)
            print(report_avg.to_dict())

            # svm_runtime = timer(baseline.svm, data=copd_geometric_dataset, config=config, verbose=args.verbose)
            # print(f'total running time of baseline.svm == {svm_runtime}')

        if args.run_rf:
            if args.emb_name == "no_feat" and args.common_nodes_feat != 'no':

                all_x_input,edges_weight, edges = preprocessing.create_common_nodes_as_features(copd, copd_geometric_dataset, used_nodes='gene', edges_weight_option= args.edges_weight_option)

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

            # repeat_model_run(name=f'rf_{args.emb_name}', num_run=num_run, model=baseline.random_forest, data=copd_geometric_dataset,config=config, evaluate=True)

            rf_runtime = timer(baseline.random_forest, data=copd_geometric_dataset, config=config, evaluate=True)
            print(f'total running time of baseline.rf == {rf_runtime}')


        if args.run_gnn:
            if args.emb_name == "no_feat" and args.common_nodes_feat != 'no':

                all_x_input, edges_weight,edges = preprocessing.create_common_nodes_as_features(copd, copd_geometric_dataset, used_nodes='gene', edges_weight_option= args.edges_weight_option)
                # -- normalize features vector
                all_x_input = preprocessing.normalize_features(csr_matrix(all_x_input))


            elif args.emb_path is not None or args.emb_name in ['no_feat', "node2vec", 'attentionwalk', 'bine']:
                '''this pass is already supported by gnn model in class Net()'''
                pass
            else:
                raise ValueError('provided emb_names mayb incorrect or args.common_nodes_feat is typed incorrectly')

            config = {
                "param" :{ # Pseudo-Label
                    'T1': int(args.t1_t2_alpha[0]),
                    'T2': int(args.t1_t2_alpha[1]),
                    'af': float(args.t1_t2_alpha[2])
                }
            }
            #TODO here>> cross validation + common_nodes_feat

            # gcn_runtime = timer(embedding.run_GCN,data=copd_geometric_dataset, lr=args.lr,weight_decay=args.weight_decay )
            # repeat_model_run(name=f'gnn_{args.emb_name}', num_run=num_run, model=embedding.GNN(data=copd_geometric_dataset,config=config).run)

            gcn_runtime = timer(embedding.GNN(data=copd_geometric_dataset).run)
            print(f'total running time of baseline.gnn == {gcn_runtime}')

        if args.run_mlp:
            if args.emb_name == "no_feat" and args.common_nodes_feat != 'no':
                all_x_input, edges_weight, edges = preprocessing.create_common_nodes_as_features(copd, copd_geometric_dataset,used_nodes='gene', edges_weight_option= args.edges_weight_option)

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
                    'sequential_layers': [                        nn.Linear(2996, 512),
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
                    # 'sequential_layers': [ # layers and its dimension have to changes depends on how emb that emb_path is directed to
                    #     nn.Linear(16, len(copd.labels2idx().keys())),
                    #     nn.LogSoftmax(dim=1)
                    # ]
                    'sequential_layers': [ # layers and its dimension have to changes depends on how emb that emb_path is directed to
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
            # repeat_model_run(name=f'mlp_{args.emb_name}', num_run=num_run, model=baseline.mlp, data=copd_geometric_dataset,config=config)

            mlp_runtime = timer(baseline.mlp, data=copd_geometric_dataset, config=config)
            print(f'total running time of baseline.mlp == {mlp_runtime}')

        if args.run_lr:
            # train_input, test_input = preprocessing.create_common_nodes_as_features(copd, copd_geometric_dataset)

            if args.emb_name == "no_feat" and args.common_nodes_feat != 'no':
                all_x_input, edges_weight, edges = preprocessing.create_common_nodes_as_features(copd, copd_geometric_dataset, used_nodes='gene', edges_weight_option= args.edges_weight_option)

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
            # repeat_model_run(name=f'lr_{args.emb_name}', num_run=num_run, model=baseline.logistic_regression, config=config, emb_name=args.emb_name)

            lr_runtime = timer(baseline.logistic_regression,config, emb_name=args.emb_name)
            print(f'total running time of baseline.lr == {lr_runtime}')

    if args.ensemble:

        ensemble_config = {
            # 'model_0':{
            #     'name': 'svm',
            #     'func':{
            #         "model": baseline.svm,
            #         # "args":[copd_geometric_dataset],
            #         # "kwargs": get_config("svm", copd, copd_geometric_dataset, used_nodes="gene", edges_weight_option='jaccard')
            #         # "kwargs": get_config("svm", copd, copd_geometric_dataset, emb_name='node2vec')
            #         # # #-- stoch 0.05
            #         # "kwargs": get_config("svm", copd, copd_geometric_dataset,
            #         #                      emb_name=r'C:\Users\awannaphasch2016\PycharmProjects\disease_node_classification\data\gene_disease\07_14_19_46\processed\embedding\node2vec\node2vec_emb_fullgraph_common_nodes_feat=gene07_14_19_46_added_edges=disease_jaccard_top_k=0.05_mask=True_stoch.txt')
            #     },
            #     # "emb_path": r'C:\Users\awannaphasch2016\PycharmProjects\disease_node_classification\data\gene_disease\07_14_19_46\processed\embedding\node2vec\node2vec_emb_fullgraph_common_nodes_feat=gene07_14_19_46_added_edges=disease_no_top_k=0.05_mask=True_stochhhhhhhhhhh1.txt',
            #     # "emb_path": r'C:\Users\awannaphasch2016\PycharmProjects\disease_node_classification\data\gene_disease\07_14_19_46\processed\embedding\node2vec\node2vec_emb_fullgraph_common_nodes_feat=gene07_14_19_46_added_edges=disease_no_top_k=0.05_mask=True_stoch0.txt',
            #     # "emb_path": r'C:\Users\awannaphasch2016\PycharmProjects\disease_node_classification\data\gene_disease\07_14_19_46\processed\embedding\node2vec\normalized_node2vec_emb_fullgraph_common_nodes_feat=gene07_14_19_46_added_edges=disease_no_top_k=0.05_mask=True_stochh1.txt',
            #     # "emb_path": r'C:\Users\awannaphasch2016\PycharmProjects\disease_node_classification\data\gene_disease\07_14_19_46\processed\embedding\node2vec\normalized_node2vec_emb_fullgraph_common_nodes_feat=gene07_14_19_46_added_edges=disease_no_top_k=0.05_mask=True_stochh1.txt',
            #     # "emb_path": r'C:\Users\awannaphasch2016\PycharmProjects\disease_node_classification\data\gene_disease\07_14_19_46\processed\embedding\node2vec\normalized_new\normalized_node2vec_emb_fullgraph_common_nodes_feat=gene07_14_19_46_added_edges=disease_no_top_k=0.05_mask=True_stochh1.txt',
            #     # "emb_path": r'C:\Users\awannaphasch2016\PycharmProjects\disease_node_classification\data\gene_disease\07_14_19_46\processed\embedding\node2vec\normalized_node2vec_emb_fullgraph_common_nodes_feat=gene07_14_19_46_added_edges=disease_no_top_k=0.05_mask=True_stochh1.txt',
            #     # "emb_path": r'C:\Users\awannaphasch2016\PycharmProjects\disease_node_classification\data\gene_disease\07_14_19_46\processed\embedding\node2vec\unnormalized_node2vec_emb_fullgraph_common_nodes_feat=gene07_14_19_46_added_edges=disease_no_top_k=0.05_mask=True_stochh1.txt',
            #     # "emb_path": r'C:\Users\awannaphasch2016\PycharmProjects\disease_node_classification\data\gene_disease\07_14_19_46\processed\embedding\node2vec\unnormalized_node2vec_emb_fullgraph_common_nodes_feat=gene07_14_19_46_added_edges=disease_no_top_k=0.05_mask=True_stochh1.txt',
            #     # "emb_path": r'C:\Users\awannaphasch2016\PycharmProjects\disease_node_classification\data\gene_disease\07_14_19_46\processed\embedding\node2vec\normalized_node2vec_emb_fullgraph_common_nodes_feat=gene07_14_19_46_added_edges=disease_no_top_k=0.1_mask=True_stochh1.txt',
            #     # "emb_name": r'node2vec',
            #     'edges_selection':{
            #                         'common_nodes_feat': 'gene',
            #                         'edges_weight_option': 'jaccard',
            #                         'mask_edges': True,
            #                         'self_loop': False,
            #                         'edges_weight_limit': None,
            #                         'edges_weight_percent': None,
            #                         'top_percent_edges': None,
            #                         'bottom_percent_edges': 0.05,
            #                         'stochastic_edges':True,
            #                         'shared_nodes_random_edges_percent': None,
            #                         'all_nodes_random_edges_percent': None,
            #                         }
            # },
            # 'model_1':{
            #     'name': 'svm',
            #     'func':{
            #         "model": baseline.svm,
            #     },
            #     "emb_path": r'C:\Users\awannaphasch2016\PycharmProjects\disease_node_classification\data\gene_disease\07_14_19_46\processed\embedding\node2vec\node2vec_emb_fullgraph_common_nodes_feat=gene07_14_19_46_added_edges=disease_no_bottom_k=0.05_mask=True_stochh0.txt',
            # },
            'model_2':{
                'name': 'svm',
                'func':{
                    "model": baseline.svm,
                },
                "emb_path": r'C:\Users\awannaphasch2016\PycharmProjects\disease_node_classification\data\gene_disease\07_14_19_46\processed\embedding\full_name_embedding_file\node2vec_emb_fullgraph_common_nodes_feat=gene07_14_19_46_added_edges=disease_no_top_bottom_k=0.5_mask=True9.txt',
            },
            'model_3':{
                'name': 'svm',
                'func':{
                    "model": baseline.svm,
                },
                # "emb_path": r'C:\Users\awannaphasch2016\PycharmProjects\disease_node_classification\data\gene_disease\07_14_19_46\processed\embedding\node2vec\node2vec_emb_fullgraph_common_nodes_feat=gene07_14_19_46_added_edges=disease_no_top_k=0.05_mask=True_stoch4.txt',
                "emb_path": r'C:\Users\awannaphasch2016\PycharmProjects\disease_node_classification\data\gene_disease\07_14_19_46\processed\embedding\full_name_embedding_file\node2vec_emb_fullgraph_common_nodes_feat=gene07_14_19_46_added_edges=disease_no_top_bottom_k=0.5_mask=True0.txt',
            },
            'model_4': {
                'name': 'svm',
                'func': {
                    "model": baseline.svm,
                },
                "emb_path": r'C:\Users\awannaphasch2016\PycharmProjects\disease_node_classification\data\gene_disease\07_14_19_46\processed\embedding\full_name_embedding_file\node2vec_emb_fullgraph_common_nodes_feat=gene07_14_19_46_added_edges=disease_no_top_bottom_k=0.5_mask=True1.txt',

            },
            'model_5': {
                'name': 'svm',
                'func': {
                    "model": baseline.svm,
                },
                "emb_path": r'C:\Users\awannaphasch2016\PycharmProjects\disease_node_classification\data\gene_disease\07_14_19_46\processed\embedding\full_name_embedding_file\node2vec_emb_fullgraph_common_nodes_feat=gene07_14_19_46_added_edges=disease_no_top_bottom_k=0.5_mask=True2.txt',
            },
            'model_6': {
                'name': 'svm',
                'func': {
                    "model": baseline.svm,
                },
                "emb_path": r'C:\Users\awannaphasch2016\PycharmProjects\disease_node_classification\data\gene_disease\07_14_19_46\processed\embedding\full_name_embedding_file\node2vec_emb_fullgraph_common_nodes_feat=gene07_14_19_46_added_edges=disease_no_top_bottom_k=0.5_mask=True3.txt',
            },
            'model_7': {
                'name': 'svm',
                'func': {
                    "model": baseline.svm,
                },
                "emb_path": r'C:\Users\awannaphasch2016\PycharmProjects\disease_node_classification\data\gene_disease\07_14_19_46\processed\embedding\full_name_embedding_file\node2vec_emb_fullgraph_common_nodes_feat=gene07_14_19_46_added_edges=disease_no_top_bottom_k=0.5_mask=True4.txt',
            },
            'model_8': {
                'name': 'svm',
                'func': {
                    "model": baseline.svm,
                },
                "emb_path": r'C:\Users\awannaphasch2016\PycharmProjects\disease_node_classification\data\gene_disease\07_14_19_46\processed\embedding\full_name_embedding_file\node2vec_emb_fullgraph_common_nodes_feat=gene07_14_19_46_added_edges=disease_no_top_bottom_k=0.5_mask=True5.txt',

            },
            'model_9': {
                'name': 'svm',
                'func': {
                    "model": baseline.svm,
                },
                "emb_path": r'C:\Users\awannaphasch2016\PycharmProjects\disease_node_classification\data\gene_disease\07_14_19_46\processed\embedding\full_name_embedding_file\node2vec_emb_fullgraph_common_nodes_feat=gene07_14_19_46_added_edges=disease_no_top_bottom_k=0.5_mask=True6.txt',

            },
            'model_10': {
                'name': 'svm',
                'func': {
                    "model": baseline.svm,
                },
                "emb_path": r'C:\Users\awannaphasch2016\PycharmProjects\disease_node_classification\data\gene_disease\07_14_19_46\processed\embedding\full_name_embedding_file\node2vec_emb_fullgraph_common_nodes_feat=gene07_14_19_46_added_edges=disease_no_top_bottom_k=0.5_mask=True7.txt',

            },
            'model_11': {
                'name': 'svm',
                'func': {
                    "model": baseline.svm,
                },
                "emb_path": r'C:\Users\awannaphasch2016\PycharmProjects\disease_node_classification\data\gene_disease\07_14_19_46\processed\embedding\full_name_embedding_file\node2vec_emb_fullgraph_common_nodes_feat=gene07_14_19_46_added_edges=disease_no_top_bottom_k=0.5_mask=True8.txt',

            },
            # 'model_12': {
            #     'name': 'svm',
            #     'func': {
            #         "model": baseline.svm,
            #     },
            #     "emb_path": r'C:\Users\awannaphasch2016\PycharmProjects\disease_node_classification\data\gene_disease\07_14_19_46\processed\embedding\node2vec\normalized_node2vec_emb_fullgraph_common_nodes_feat=gene07_14_19_46_added_edges=disease_no_top_k=0.25_mask=True_stochh5.txt',
            #
            # },
            # 'model_13': {
            #     'name': 'svm',
            #     'func': {
            #         "model": baseline.svm,
            #     },
            #     "emb_path": r'C:\Users\awannaphasch2016\PycharmProjects\disease_node_classification\data\gene_disease\07_14_19_46\processed\embedding\node2vec\normalized_node2vec_emb_fullgraph_common_nodes_feat=gene07_14_19_46_added_edges=disease_no_top_k=0.25_mask=True_stochh6.txt',
            #
            # },
            # 'model_14': {
            #     'name': 'svm',
            #     'func': {
            #         "model": baseline.svm,
            #     },
            #     "emb_path": r'C:\Users\awannaphasch2016\PycharmProjects\disease_node_classification\data\gene_disease\07_14_19_46\processed\embedding\node2vec\normalized_node2vec_emb_fullgraph_common_nodes_feat=gene07_14_19_46_added_edges=disease_no_top_k=0.25_mask=True_stochh7.txt',
            #
            # },
            # 'model_15': {
            #     'name': 'svm',
            #     'func': {
            #         "model": baseline.svm,
            #     },
            #     "emb_path": r'C:\Users\awannaphasch2016\PycharmProjects\disease_node_classification\data\gene_disease\07_14_19_46\processed\embedding\node2vec\normalized_node2vec_emb_fullgraph_common_nodes_feat=gene07_14_19_46_added_edges=disease_no_top_k=0.25_mask=True_stochh8.txt',
            #
            # },
            # 'model_16': {
            #     'name': 'svm',
            #     'func': {
            #         "model": baseline.svm,
            #     },
            #     "emb_path": r'C:\Users\awannaphasch2016\PycharmProjects\disease_node_classification\data\gene_disease\07_14_19_46\processed\embedding\node2vec\normalized_node2vec_emb_fullgraph_common_nodes_feat=gene07_14_19_46_added_edges=disease_no_top_k=0.25_mask=True_stochh9.txt',
            #
            # },
            # 'model_17': {
            #     'name': 'svm',
            #     'func': {
            #         "model": baseline.svm,
            #     },
            #     "emb_path": r'C:\Users\awannaphasch2016\PycharmProjects\disease_node_classification\data\gene_disease\07_14_19_46\processed\embedding\node2vec\normalized_node2vec_emb_fullgraph_common_nodes_feat=gene07_14_19_46_added_edges=disease_no_top_k=0.25_mask=True_stochh3.txt',
            #
            # },

        }
        # run_ensemble(copd, config=ensemble_config)

        #=====================
        #==automated
        #=====================
        # model = {"svm" : baseline.svm,
        #          'lr'  :baseline.logistic_regression,
        #          'rf'  : baseline.random_forest,
        #         # 'gnn':,
        #          'mlp' :baseline.mlp}
        #
        model = {
            "svm": baseline.svm,
            'lr'  :baseline.logistic_regression ,
            'mlp': baseline.mlp,
            'rf'  : baseline.random_forest,
        }
        # args.added_edge_option = 'longest_path'
        args.added_edge_option = 'no_shared_gene'
        # args.added_edge_option = 'shared_gene'
        # args.added_edge_option = 'same_class'

        args.embedding_name = 'node2vec'
        # args.embedding_name = 'gcn'
        tmp = f'C:\\Users\\awannaphasch2016\\PycharmProjects\\disease_node_classification\\data\\gene_disease\\07_14_19_46\\processed\\embedding\\{args.added_edge_option}\\node2vec'
        # tmp = r'C:\Users\awannaphasch2016\PycharmProjects\disease_node_classification\data\gene_disease\07_14_19_46\processed\embedding\shared_gene\node2vec\top_bottom_k_stoch\0'
        # tmp = f'C:\\Users\\awannaphasch2016\\PycharmProjects\\disease_node_classification\\data\\gene_disease\\07_14_19_46\\processed\\embedding\\gcn\\split=0.8\\lr=0.09_d=0.5_wd=0.006_wc=[1 1 1 1 1 0]'
        # tmp = r'C:\Users\awannaphasch2016\PycharmProjects\disease_node_classification\data\gene_disease\07_14_19_46\processed\embedding\same_class\node2vec'

        # if tmp.split('\\')[-1] == 'node2vec':
        #     args.embedding_name = 'node2vec' # please use embedding_name not emb_name; they have different purpose
        # elif tmp.split('\\')[-3] == 'gcn':
        #     args.embedding_name = 'gcn' # please use embedding_name not emb_name; they have different purpose
        # else:
        #     raise ValueError('In __init__.py >if__name__ == "main" > if args.ensemble > else')
        print(args.split)

        #=====================
        #==preprocess copd_geometric_dataset
        #=====================

        # --------create data to be used as args to create Dataset object
        # x, copd, edge_index, y = preprocessing.data_preprocessing(dataset=copd)
        # copd_geometric_dataset = all_datasets.GeometricDataset(copd, x=x, edges_index=edge_index, y=y,
        #                                                        split=args.split, undirected=not args.directed)
        for model_key, model_func in model.items():
            for root, dirs, files in os.walk(tmp, topdown=True):
                ensemble_config = {}
                # if root.split('\\')[-2] not in ["all_nodes_random", 'bottom_k', 'bottom_k_stoch',]:
                # if root.split('\\')[-2]  in ['top_k_stoch']:
                #     if root.split('\\')[-1] in ["0.5", '0.05']:
                for i,name in enumerate(files):
                    ensemble_config[f'model_{args.split}_{i}'] = {}
                    # ensemble_config[f'model_{args.split}_{i}']['name'] = 'svm'
                    # ensemble_config[f'model_{args.split}_{i}']['func'] = {"model": baseline.svm}
                    ensemble_config[f'model_{args.split}_{i}']['name'] = model_key
                    ensemble_config[f'model_{args.split}_{i}']['func'] = {"model": model_func}
                    ensemble_config[f'model_{args.split}_{i}']['emb_path'] = os.path.join(root, name)
                    # ensemble_config[f'model_{args.split}_{i}']['emb_name'] = 'gnn'
                    print(f'load data from {os.path.join(root, name)}')
                    # print(ensemble_config)
                    # exit()
                if len(ensemble_config) > 0:
                    run_ensemble(copd, config=ensemble_config)
                    # exit()
            print('=================')
            print(f'====finish running {model_key}')
            print('=================')

    elif args.check_condition is not None:
        #--------check same condision for all base model
        for model in args.check_condition:
            if model == "all":
                args.run_svm = True
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


        run_model(args.num_run)
        # rt = timer(run_model)
        # print(f'\ntotal running time of args.check_condition == {rt}')
    else:
        #=====================
        #==run a model
        #=====================
        run_model()


if __name__ == "__main__":
    run_main()