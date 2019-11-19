import sys,os
USER = os.environ['USERPROFILE']
sys.path.insert(1,f'{USER}\\PycharmProjects\\my_utility')

from performance_metrics.metrics import *

from arg_parser import  *
import performance_metrics
# =====================
# ==base line models
# =====================

# -- logistic regression with node embedding
# def logistic_regression(config=None, emb_name=None):
def logistic_regression(data, config=None, decision_func='ovr', verbose=True):
    '''
    run logistic regression

    :param config:
    :param use_emb: use node embedding in logistic regression
    :return:
    '''

    x_train, label_train = config['train_input'], config['train_label']
    x_test, label_test = config['test_input'], config['test_label']

    assert isinstance(x_train, type(torch.rand(1))), "x_train must have tensor type "
    assert isinstance(label_train, type(torch.rand(1))), "label_train must have tensor type "
    assert isinstance(x_test, type(torch.rand(1))), "x_test must have tensor type "
    assert isinstance(label_test, type(torch.rand(1))), "label_test must have tensor type "

    # train_input = x_train.type(torch.float).numpy()
    # test_input = x_test.type(torch.float).numpy()
    # train_label = label_train.type(torch.long).numpy()
    # test_label = label_test.type(torch.long).numpy()

    train_input = x_train.type(torch.double).numpy()
    test_input = x_test.type(torch.double).numpy()
    train_label = label_train.type(torch.double).numpy()
    test_label = label_test.type(torch.double).numpy()

    from sklearn import metrics
    from sklearn.linear_model import LogisticRegression

    # model = LogisticRegression(solver='lbfgs', multi_class='ovr')
    model = LogisticRegression(solver='lbfgs', multi_class='multinomial')


    # --------prediction with cross validation
    if args.cv is not None:
        print(f"running {logistic_regression.__name__} with cross validation ")

        s = time.time()
        # pred_train = cross_val_predict(clf, x_train, label_train, cv=int(args.cv)) # cv deafault = 3

        # TODO here>> predict test with train model
        proba = cross_val_predict(model, np.concatenate((x_train, x_test), axis=0),
                                  np.concatenate((label_train, label_test), axis=0), cv=int(args.cv),
                                  method='predict_proba')
        pred = proba.argmax(1)

        f = time.time()
        total = f - s
        print(f'training {total}')

        # =====================
        # ==report performance
        # =====================
        save_path = f"log/gene_disease/{args.time_stamp}/classifier/lr/cross_valid={args.cv}/lr={args.lr}_d={args.dropout}_wd={args.weight_decay}/report_performance/"
        file_name = f'cross_validation={args.cv}_emb={args.emb_name}_epoch={args.epochs}_wc={args.weighted_class}.txt'
        import os
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        report_test = performance_metrics.report_performances(
            y_true=np.concatenate((label_train, label_test), axis=0),
            y_pred=pred,
            y_score=proba,
            save_path=f'{save_path}train/',
            file_name=file_name
        )
        if args.report_performance:
            print(report_test)

    else:
        model.fit(train_input, train_label)

        log_list = []

        # args.cv = "NO"
        print(f"running {logistic_regression.__name__} without cross validation ")
        y_pred_train = model.predict(train_input)
        y_pred_train_proba = model.predict_proba(train_input)
        # y_pred_train_proba = model.decision_function(x_train)
        # y_pred_train = y_pred_train_proba.argmax(1)


        y_pred_test = model.predict(test_input)
        y_pred_test_proba = model.predict_proba(test_input)
        # y_pred_test_proba = model.decision_function(x_test)
        # y_pred_test = y_pred_test_proba.argmax(1)
        # =====================
        # ==performance report
        # =====================
        # save_path = f"log/gene_disease/{args.time_stamp}/classifier/lr/split={args.split}/lr={args.lr}_d={args.dropout}_wd={args.weight_decay}/report_performance/"
        # file_name = f'emb={args.emb_name}_epoch={args.epochs}_wc={args.weighted_class}.txt'
        if args.embedding_name == 'node2vec':
            tmp = args.emb_path.split("\\")[10:]
            args.percent = tmp[-2]
            file_name = tmp[-1]   # eg node2vec_all_nodes_random_0.05_0.txt
            folder = args.emb_path.split('\\')[10]
            folder = folder + '\\' + args.emb_path.split('\\')[11]
            folder = folder + '\\' + args.emb_path.split('\\')[12] + '\\' + f"{args.percent}/"
        elif args.embedding_name == 'gcn':
            folder = '\\'.join(args.emb_path.split('\\')[-6:-1]) + '\\' + f"{args.percent}/"
            file_name = args.emb_path.split("\\")[-1]
        else:
            raise ValueError('in all_models > baseline > models > logistic_regression() > else > else ')

        save_path = f"log/gene_disease/{args.time_stamp}/classifier/{args.added_edge_option}/lr/split={args.split}/report_performance/{folder}/"

        import os
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        print(f'save to {save_path+"train/"+file_name}')
        print(f'save to {save_path + "test/" + file_name}')
        report_test = None
        report_train = None
        report_train = performance_metrics.report_performances(
            y_true=train_label,
            y_pred=y_pred_train,
            y_score=y_pred_train_proba,
            save_path=f'{save_path}train/',
            file_name=file_name
        )
        report_test = performance_metrics.report_performances(
            y_true=test_label,
            y_pred=y_pred_test,
            y_score=y_pred_test_proba,
            save_path=f'{save_path}test/',
            file_name=file_name
            )
        if args.report_performance:
            print(report_train)
            print(report_test)

    return report_test, model
    # return report_test.iloc[-1], model


# run_node2vec(x_train,label_train, x_test, label_test, decision_func='ovo'):
def svm(data, config=None, decision_func='ovr', verbose=True):
    '''

    :param data:
    :param config:
    :param decision_func:
    :param verbose:
    :return:
        proba: confident over
    '''
    from sklearn import svm

    x_train, label_train = config['train_input'], config['train_label']
    x_test, label_test = config['test_input'], config['test_label']

    assert isinstance(x_train, type(torch.rand(1))), "x_train must have tensor type "
    assert isinstance(label_train, type(torch.rand(1))), "label_train must have tensor type "
    assert isinstance(x_test, type(torch.rand(1))), "x_test must have tensor type "
    assert isinstance(label_test, type(torch.rand(1))), "label_test must have tensor type "

    # =====================
    # ==cast train and test data to the correct type
    # =====================

    x_train = x_train.type(torch.double).numpy()
    x_test = x_test.type(torch.double).numpy()
    label_train = label_train.type(torch.double).numpy()
    label_test = label_test.type(torch.double).numpy()

    # =====================
    # ==fitting
    # =====================
    clf = svm.SVC(gamma='scale', decision_function_shape=decision_func, probability=True)


    # =====================
    # ==prediction
    # =====================

    #--------prediction with cross validation
    if args.cv is not None:
        print(f"running {svm.__name__} with cross validation ")

        s = time.time()
        # pred_train = cross_val_predict(clf, x_train, label_train, cv=int(args.cv)) # cv deafault = 3

        # proba = cross_val_predict(clf, np.concatenate((x_train, x_test), axis=0), np.concatenate((label_train, label_test), axis=0), cv=int(args.cv), method='predict_proba')
        proba = cross_val_predict(clf, np.concatenate((x_train, x_test), axis=0), np.concatenate((label_train, label_test), axis=0), cv=int(args.cv), method='decision_function')
        pred = proba.argmax(1) # expecting to get the predected class of each instances

        f = time.time()
        total = f-s
        print(f'training {total}' )

        # =====================
        # ==report performance
        # =====================
        # if args.report_performance:
        save_path = f"log/gene_disease/{args.time_stamp}/classifier/svm/cross_valid={args.cv}/lr={args.lr}_d={args.dropout}_wd={args.weight_decay}/report_performance/"
        file_name = f'cross_validation={args.cv}_emb={args.emb_name}_epoch={args.epochs}_wc={args.weighted_class}.txt'
        import os
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        report_test = performance_metrics.report_performances(
            y_true=np.concatenate((label_train, label_test), axis=0), # is this correct? why train+test?
            y_pred=pred,
            y_score=proba,
            save_path=f'{save_path}train/',
            file_name=file_name
        )
        if args.report_performance:
            print(report_test)

        raise ValueError('please check svm=> if args.cv before moving on')
    else:
        clf.fit(x_train, label_train)

        # args.cv = "NO"
        print(f"running {svm.__name__} without cross validation ")
        accs = []
        # --------train
        # proba_train = clf.predict_proba(x_train)
        proba_train = clf.decision_function(x_train)
        pred_train = proba_train.argmax(1)
        # print(f'predition signle = {pred_train}`')
        # accs.append(accuracy(pred_train, label_train))

        # --------test
        # proba_test = clf.predict_proba(x_test)
        proba_test = clf.decision_function(x_test) # todo Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.
        pred_test = proba_test.argmax(1)

        print(f'predition signle = {pred_test}`')
        print(f'label_test = {label_test}')
        # accs.append(accuracy(pred_test, label_test))
        # proba_test = clf.predict_proba(x_test)

        if verbose:
            # print(f"test prediction: {result}")
            print("[train, test] accs")
            print(accs)

        #=====================
        #==report performance
        #=====================
        #TODO here>> split cases for emb_path and emb_name
        if args.embedding_name == 'node2vec':
            tmp = args.emb_path.split("\\")[10:]
            args.percent = tmp[-2]
            # file_name = '_'.join(tmp) # eg node2vec_all_nodes_random_0.05_0.txt
            file_name = tmp[-1]   # eg node2vec_all_nodes_random_0.05_0.txt
            folder = args.emb_path.split('\\')[-4]
            folder = folder + '\\'+ args.emb_path.split('\\')[-3]
            folder = folder + '\\' + args.emb_path.split('\\')[-2]
        elif args.embedding_name == 'gcn':
            folder = '\\'.join(args.emb_path.split('\\')[-6:-1]) + '\\' + f"{args.percent}/"
            file_name = args.emb_path.split("\\")[-1]
        else:
            raise ValueError('in all_models > baseline > models > svm() > else > else ')

        save_path = f"log/gene_disease/{args.time_stamp}/classifier/{args.added_edge_option}/svm/split={args.split}/report_performance/{folder}/"

        # file_name = f'cross_validation={args.cv}_emb={args.emb_name}_epoch={args.epochs}_wc={args.weighted_class}.txt'


        import os
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        print(f'folder = {folder}')
        print(f'save to {save_path+"train/"+file_name}')
        print(f'save to {save_path + "test/" + file_name}')
        report_train = performance_metrics.report_performances(
            y_true=label_train,
            y_pred=pred_train,
            y_score=proba_train,
            save_path=f'{save_path}train/',
            file_name=file_name
        )
        report_test = performance_metrics.report_performances(
            y_true=label_test,
            y_pred=pred_test,
            y_score=proba_test,
            save_path=f'{save_path}test/',
            file_name=file_name
        )
        if args.report_performance:
            print(report_train)
            print(report_test)



        # return report_test.iloc[-1]
    return report_test.iloc[-1], clf

    # return pred_train, pred_test, accs


def mlp(data, config):
# def mlp(data, config=None):
    '''
    run multi-layer perceptron
    input data is node with gene as its features.
    :return:
    '''
    # # -- input arguments
    # copd = config["data"]
    # # input = config["input"]  # {disease_idx1: [[0,0,0,1,0,0],[0,1,0,0,0,0] ....], disease_idx2: [...],... }
    # # y = config['label']
    # # train_mask = config['train_mask']
    # # test_mask = config['test_mask']
    # train_input = config['train_input']
    # test_input = config['test_input']
    # train_label = config['train_label']
    # test_label = config['test_label']
    # seqential_layers = config['sequential_layers']
    # # hidden_sizes = config['hidden_layers']
    # epochs = config['epochs']
    # args = config['args']
    # param = config['param']
    #
    # # -- convert to tensor
    # train_input = torch.tensor(train_input, dtype=torch.float)
    # test_input = torch.tensor(test_input, dtype=torch.float)
    # train_label = torch.tensor(train_label, dtype=torch.long)
    # test_label = torch.tensor(test_label, dtype=torch.long)

    #=====================
    #==newly adding part
    #=====================

    x_train, label_train = config['train_input'], config['train_label']
    x_test, label_test = config['test_input'], config['test_label']

    assert isinstance(x_train, type(torch.rand(1))), "x_train must have tensor type "
    assert isinstance(label_train, type(torch.rand(1))), "label_train must have tensor type "
    assert isinstance(x_test, type(torch.rand(1))), "x_test must have tensor type "
    assert isinstance(label_test, type(torch.rand(1))), "label_test must have tensor type "

    seqential_layers = config['sequential_layers']
    epochs = config['epochs']
    param = config['param']

    # =====================
    # ==cast train and test data to the correct type
    # =====================

    # -- convert to tensor
    train_input = torch.tensor(x_train, dtype=torch.float)
    test_input = torch.tensor(x_test, dtype=torch.float)
    train_label = torch.tensor(label_train, dtype=torch.long)
    test_label = torch.tensor(label_test, dtype=torch.long)


    weighted_class = torch.tensor(list(map(int, args.weighted_class)), dtype=torch.float)

    #=====================
    #==build models
    #=====================
    # model() -> optimizer -> loss -> model.train()-> optimizer.zero_grad() -> loss.backward() -> optimizer.step() -> next epoch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = nn.Sequential(*seqential_layers)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def train(train_data=None, all_train_labels=None):
        if train_data is not None: #??
            train_input = train_data

        if all_train_labels is not None:
            train_label = all_train_labels

        model.train()
        optimizer.zero_grad()
        if args.pseudo_label_topk:
            labeled_loss = F.nll_loss(model(train_input), train_label,
                                      weight=torch.tensor(list(map(int, args.weighted_class)), dtype=torch.float),
                                      reduction="mean")

            # -- labeled top k most confidence node to be pseduo_labels
            pseudo_label_pred = model(train_input).max(1)[1]

            tmp = model(train_input).max(1)[1].detach().flatten().tolist()
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
                pseudo_label_loss = F.nll_loss(model(train_input)[top_k], pseudo_label_pred[top_k],
                                               weight=weighted_class,
                                               reduction='mean')
            else:
                pseudo_label_loss = 0

            loss_output = labeled_loss + pseudo_label_loss
        else:
            loss_output = F.nll_loss(model(train_input), train_label,
                                     weight=torch.tensor(list(map(int, args.weighted_class)), dtype=torch.float),
                                     reduction="mean")
        loss_output.backward()
        optimizer.step()
        return loss_output.data

    def test(train_data=None, all_train_labels=None, test_data=None, all_test_labels=None):
        if train_data is not None:
            train_input = train_data

        if test_data is not None:
            test_input = test_data

        if all_train_labels is not None:
            train_label = all_train_labels

        if all_test_labels is not None:
            test_label = all_test_labels

        model.eval()
        train_prob = model(train_input)
        train_pred = train_prob.max(1)[1]
        train_acc = train_pred.eq(train_label).sum().item() / train_label.shape[0]

        test_prob = model(test_input)
        test_pred = test_prob.max(1)[1]
        test_acc = test_pred.eq(test_label).sum().item() / test_label.shape[0]

        return [train_acc, test_acc, train_pred, test_pred, train_prob, test_prob, model]

    train_acc_hist = []
    test_acc_hist = []
    loss_hist = []
    log_list = []

    # all_x[train_index], all_labels[test_index]
    def run_epochs(train_data=None, all_train_label=None, test_data=None, all_test_label=None):
        for epoch in range(epochs):
            loss_epoch = train(train_data,all_train_label)
            train_acc, test_acc, train_pred, test_pred, train_prob, test_prob, trained_model  = test(train_data,all_train_label, test_data, all_test_label)
            logging = 'Epoch: {:03d}, Train: {:.4f}, Test: {:.4f}'.format(epoch, train_acc, test_acc)
            if args.verbose:
                print(logging)
            log_list.append(logging)
            loss_hist.append(loss_epoch)
            train_acc_hist.append(train_acc)
            test_acc_hist.append(test_acc)

        return [train_pred, test_pred, train_prob, test_prob, all_train_label, all_test_label, trained_model]


    #=====================
    #==cross validation
    #=====================
    if args.cv is not None:
        print(f"running {mlp.__name__} with cross validation ")
        s = time.time()
        # pred_train = cross_val_predict(clf, x_train, label_train, cv=int(args.cv)) # cv deafault = 3
        from sklearn.model_selection import StratifiedKFold

        all_x = np.concatenate((train_input.numpy(), test_input.numpy()), axis=0)
        all_labels = np.concatenate((train_label.numpy(), test_label.numpy()), axis=0)
        n_splits = int(args.cv)
        # n_splits = int(all_x.shape[0]/int(args.cv)) + 1 if all_x.shape[0]%int(args.cv) != 0 else int(all_x.shape[0]/int(args.cv))
        # assert n_splits * int(all_x.shape[0]/n_splits) >= all_x.shape[0], "n_splits * args.cv <= all_x.shape[0]"

        #--------convert to torch
        all_x = torch.tensor(all_x).type(torch.float)
        all_labels = torch.tensor(all_labels).type(torch.long)

        # avg_metrics = None
        # avg_train_metrics = None
        avg_test_metrics = None
        np.random.seed(args.seed)
        cv = StratifiedKFold(n_splits=n_splits)

        for train_index , test_index in cv.split(all_x, all_labels):
            train_pred, test_pred, train_prob, test_prob,all_train_label, all_test_label  = run_epochs(all_x[train_index], all_labels[train_index], all_x[test_index], all_labels[test_index] )

            #=====================
            #==get performance metrics (no plot, no print to screen)
            #=====================
            #--------training
            # train_measurement_metrics = performance_metrics.report_performances(all_train_label.numpy(),
            #                                                                     train_pred.numpy(),
            #                                                                     train_prob.detach().numpy(),
            #                                                                     get_avg_total=True)
            # # --------testing
            test_measurement_metrics = performance_metrics.report_performances(all_test_label.numpy(),
                                                                               test_pred.numpy(),
                                                                               test_prob.detach().numpy(),
                                                                               get_avg_total=True)


            # avg_metrics = avg_metrics.add(train_measurement_metrics) if avg_metrics is not None else train_measurement_metrics
            # avg_train_metrics = avg_train_metrics.add(train_measurement_metrics) if avg_train_metrics is not None else train_measurement_metrics
            avg_test_metrics = avg_test_metrics.add(test_measurement_metrics) if avg_test_metrics is not None else test_measurement_metrics
            # print('here')

        if args.report_performance:
            # avg_metrics = avg_metrics.divide(n_splits)
            # print(avg_metrics.__repr__())
            # avg_train_metrics = avg_train_metrics.divide(n_splits)
            # print(avg_train_metrics.__repr__())
            # print('\n')
            avg_test_metrics = avg_test_metrics.divide(n_splits)
            print(avg_test_metrics.__repr__())

        # =====================
        # ==save cross validation to file
        # =====================
        save_path = f"log/gene_disease/{args.time_stamp}/classifier/mlp/cross_valid={args.cv}/lr={args.lr}_d={args.dropout}_wd={args.weight_decay}/report_performance/"
        file_name = f'emb={args.emb_name}_epoch={args.epochs}_wc={args.weighted_class}.txt'
        import os
        os.makedirs(save_path, exist_ok=True)

        df = pd.DataFrame(avg_test_metrics)
        df.to_csv(save_path + file_name, header=True, index=False, sep='\t', mode='w')

        raise ValueError('please check mlp=> if args.cv before moving on')

        return avg_test_metrics.iloc[-1]

    else:
        # train_pred, test_pred, train_prob, test_prob, all_train_label, all_test_label = run_epochs(all_x[train_index], all_labels[train_index], all_x[test_index], all_labels[test_index])
        # def run_epochs(train_data=None, all_train_label=None, test_data=None, all_test_label=None):
        train_pred, test_pred, train_prob , test_prob, all_train_label, all_test_label, trained_model = run_epochs(train_input, train_label, test_input, test_label)
        #=====================
        #==report performance
        #=====================
        # save_path = f"log/gene_disease/{args.time_stamp}/classifier/mlp/split={args.split}/lr={args.lr}_d={args.dropout}_wd={args.weight_decay}/report_performance/"
        # file_name = f'emb={args.emb_name}_epoch={args.epochs}_wc={args.weighted_class}.txt'
        # import os
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)
        # #--------train
        # report_train = performance_metrics.report_performances(all_train_label.numpy(),
        #                                                 train_pred.numpy(),
        #                                                 train_prob.detach().numpy(),
        #                                                 get_avg_total=True)
        # #--------test
        # report_test = performance_metrics.report_performances(all_test_label.numpy(),
        #                                                test_pred.numpy(),
        #                                                test_prob.detach().numpy(),
        #                                                get_avg_total=True)
        # if args.report_performance:
        #     print(report_train)
        #     print(report_test)
        if args.embedding_name == 'node2vec':
            tmp = args.emb_path.split("\\")[10:]
            args.percent = tmp[-2]
            file_name = tmp[-1]   # eg node2vec_all_nodes_random_0.05_0.txt
            # file_name = '_'.join(tmp) # eg node2vec_all_nodes_random_0.05_0.txt
            folder = args.emb_path.split('\\')[10]
            folder = folder + '\\' + args.emb_path.split('\\')[11]
            folder = folder + '\\' + args.emb_path.split('\\')[12] + '\\' + f"{args.percent}/"
        elif args.embedding_name == 'gcn':
            folder = '\\'.join(args.emb_path.split('\\')[-6:-1]) + '\\' + f"{args.percent}/"
            file_name = args.emb_path.split("\\")[-1]
        else:
            raise ValueError('in all_models > baseline > models > mlp() > else > else ')

        save_path = f"log/gene_disease/{args.time_stamp}/classifier/{args.added_edge_option}/mlp/split={args.split}/report_performance/{folder}/"

        import os
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        print(f'save to {save_path + "train/" + file_name}')
        print(f'save to {save_path + "test/" + file_name}')
        # --------train
        report_train = performance_metrics.report_performances(y_true=all_train_label.numpy(),
                                                               y_pred=train_pred.numpy(),
                                                               y_score=train_prob.detach().numpy(),
                                                               save_path=f'{save_path}train/',
                                                               file_name=file_name)
        # --------test
        report_test = performance_metrics.report_performances(y_true=all_test_label.numpy(),
                                                              y_pred=test_pred.numpy(),
                                                              y_score=test_prob.detach().numpy(),
                                                              save_path=f'{save_path}test/',
                                                              file_name=file_name)
        if args.report_performance:
            print(report_train)
            print(report_test)

    return report_test.iloc[-1], trained_model



# def random_forest(x_train,label_train,x_test,label_test, bs,epoch, lr):
def random_forest(data, config, evaluate=False):
# def random_forest(data, config=None, decision_func='ovr', verbose=True):
    '''
    url: https://towardsdatascience.com/an-implementation-and-explanation-of-the-random-forest-in-python-77bf308a9b76
    :param x_train:
    :param label_train:
    :param x_test:
    :param label_test:
    :return:
    '''
    # # x_train, label_train = data.x[data.train_mask], data.y[data.train_mask]
    # # x_test, label_test = data.x[data.test_mask], data.y[data.test_mask]
    #
    # x_train, label_train = config['train_input'], config['train_label']
    # x_test, label_test = config['test_input'], config['test_label']
    #
    # x_train = x_train.type(torch.float).numpy()
    # x_test = x_test.type(torch.float).numpy()
    # label_train = label_train.type(torch.long).numpy()
    # label_test = label_test.type(torch.long).numpy()
    #
    # from sklearn.ensemble import RandomForestClassifier
    # RSEED = 50
    #
    # # Create the model with 100 trees
    # model = RandomForestClassifier(n_estimators=100,
    #                                random_state=RSEED,
    #                                max_features='sqrt',
    #                                n_jobs=-1, verbose=1)
    #
    #
    #=====================
    #==newly added
    #=====================
    from sklearn.ensemble import RandomForestClassifier

    RSEED = 50
    x_train, label_train = config['train_input'], config['train_label']
    x_test, label_test = config['test_input'], config['test_label']

    assert isinstance(x_train, type(torch.rand(1))), "x_train must have tensor type "
    assert isinstance(label_train, type(torch.rand(1))), "label_train must have tensor type "
    assert isinstance(x_test, type(torch.rand(1))), "x_test must have tensor type "
    assert isinstance(label_test, type(torch.rand(1))), "label_test must have tensor type "

    # =====================
    # ==cast train and test data to the correct type
    # =====================

    x_train = x_train.type(torch.double).numpy()
    x_test = x_test.type(torch.double).numpy()
    label_train = label_train.type(torch.double).numpy()
    label_test = label_test.type(torch.double).numpy()

    model = RandomForestClassifier(n_estimators=100,
                                   random_state=RSEED,
                                   max_features='sqrt',
                                   n_jobs=-1, verbose=1)

    #--------prediction with cross validation
    if args.cv is not None:
        print(f"running {random_forest.__name__} with cross validation ")

        s = time.time()
        # pred_train = cross_val_predict(clf, x_train, label_train, cv=int(args.cv)) # cv deafault = 3

        proba = cross_val_predict(model, np.concatenate((x_train, x_test), axis=0), np.concatenate((label_train, label_test), axis=0), cv=int(args.cv), method='predict_proba')
        pred = proba.argmax(1)

        f = time.time()
        total = f-s
        print(f'training {total}' )

        # =====================
        # ==report performance
        # =====================
        save_path = f"log/gene_disease/{args.time_stamp}/classifier/rf/cross_valid={args.cv}/lr={args.lr}_d={args.dropout}_wd={args.weight_decay}/report_performance/"
        file_name = f'cross_validation={args.cv}_emb={args.emb_name}_epoch={args.epochs}_wc={args.weighted_class}.txt'
        import os
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        report_train = performance_metrics.report_performances(
            y_true=np.concatenate((label_train, label_test), axis=0),
            y_pred=pred,
            y_score=proba,
            save_path=f'{save_path}train/',
            file_name=file_name
        )
        if args.report_performance:
            print(report_train)

    else:

        # Fit on training data
        model.fit(x_train, label_train)

        n_nodes = []
        max_depths = []

        # Stats about the trees in random forest
        for ind_tree in model.estimators_:
            n_nodes.append(ind_tree.tree_.node_count)
            max_depths.append(ind_tree.tree_.max_depth)

        print(f'Average number of nodes {int(np.mean(n_nodes))}')
        print(f'Average maximum depth {int(np.mean(max_depths))}')

        # args.cv = "NO"
        print(f"running {random_forest.__name__} without cross validation ")
        # Training predictions (to demonstrate overfitting)
        train_rf_predictions = model.predict(x_train)
        train_rf_probs = model.predict_proba(x_train)
        # train_rf_probs = model.predict_proba(x_train)[:, 1]

        # Testing predictions (to determine performance)
        rf_predictions = model.predict(x_test)
        rf_probs = model.predict_proba(x_test)
        # rf_probs = model.predict_proba(x_test)[:, 1]

        # =====================
        # ==performance report
        # =====================
        # save_path = f"log/gene_disease/{args.time_stamp}/classifier/rf/split={args.split}/lr={args.lr}_d={args.dropout}_wd={args.weight_decay}/report_performance/"
        # file_name = f'emb={args.emb_name}_epoch={args.epochs}_wc={args.weighted_class}.txt'

        if args.embedding_name == 'node2vec':
            tmp = args.emb_path.split("\\")[10:]
            args.percent = tmp[-2]
            file_name = tmp[-1]   # eg node2vec_all_nodes_random_0.05_0.txt
            # file_name = '_'.join(tmp) # eg node2vec_all_nodes_random_0.05_0.txt
            folder = args.emb_path.split('\\')[10]
            folder = folder + '\\' + args.emb_path.split('\\')[11]
            folder = folder + '\\' + args.emb_path.split('\\')[12] + '\\' + f"{args.percent}/"
        elif args.embedding_name == 'gcn':
            folder = '\\'.join(args.emb_path.split('\\')[-6:-1]) + '\\' + f"{args.percent}/"
            file_name = args.emb_path.split("\\")[-1]
        else:
            raise ValueError('in all_models > baseline > models > random_forest() > else > else ')

        save_path = f"log/gene_disease/{args.time_stamp}/classifier/{args.added_edge_option}/rf/split={args.split}/report_performance/{folder}/"
        import os
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        print(f'save to {save_path+"train/"+file_name}')
        print(f'save to {save_path + "test/" + file_name}')

        report_train = performance_metrics.report_performances(
            y_true=label_train,
            y_pred=train_rf_predictions,
            y_score=train_rf_probs,
            save_path=f'{save_path}train/',
            file_name=file_name
        )
        report_test = performance_metrics.report_performances(
            y_true=label_test,
            y_pred=rf_predictions,
            y_score=rf_probs,
            save_path=f'{save_path}test/',
            file_name=file_name
        )
        if args.report_performance:
            print(report_train)
            print(report_test)

    return report_train.iloc[-1], model




