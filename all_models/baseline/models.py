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
def logistic_regression(config, emb_name):
    '''
    run logistic regression

    :param config:
    :param use_emb: use node embedding in logistic regression
    :return:
    '''

    x_train, label_train = config['train_input'], config['train_label']
    x_test, label_test = config['test_input'], config['test_label']

    train_input = x_train.type(torch.float).numpy()
    test_input = x_test.type(torch.float).numpy()
    train_label = label_train.type(torch.long).numpy()
    test_label = label_test.type(torch.long).numpy()

    from sklearn import metrics
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(solver='lbfgs', multi_class='ovr')


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
        if args.report_performance:
            save_path = f"log/gene_disease/{args.time_stamp}/classifier/lr/cross_valid={args.cv}/lr={args.lr}_d={args.dropout}_wd={args.weight_decay}/report_performance/"
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
            print(report_train)

    else:
        model.fit(train_input, train_label)

        log_list = []

        # args.cv = "NO"
        print(f"running {logistic_regression.__name__} without cross validation ")
        y_pred_train = model.predict(train_input)
        y_pred_test = model.predict(test_input)

        # =====================
        # ==performance report
        # =====================
        if args.report_performance:
            save_path = f"log/gene_disease/{args.time_stamp}/classifier/lr/split={args.split}/lr={args.lr}_d={args.dropout}_wd={args.weight_decay}/report_performance/"
            file_name = f'emb={args.emb_name}_epoch={args.epochs}_wc={args.weighted_class}.txt'
            import os
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            report = performance_metrics.report_performances(
                y_true=train_label,
                y_pred=y_pred_train,
                y_score=model.predict_proba(train_input),
                save_path=f'{save_path}train/',
                file_name=file_name
            )
            print(report)
            report = performance_metrics.report_performances(
                y_true=test_label,
                y_pred=y_pred_test,
                y_score=model.predict_proba(test_input),
                save_path=f'{save_path}test/',
                file_name=file_name
            )
            print(report)

    #=====================
    #==metrix results
    #=====================
    # # -- training datset
    # cm_train = confusion_matrix(y_pred_train, train_label)
    # cm_train = np.array2string(cm_train)
    # count_misclassified = (train_label != y_pred_train).sum()
    # accuracy = metrics.accuracy_score(train_label, y_pred_train)
    #
    # txt = ["For training data", 'Misclassified samples: {}'.format(count_misclassified),
    #        'Accuracy: {:.2f}'.format(accuracy)]
    # log_list.append('\n'.join(txt))
    # print(log_list[-1])

    # #-- test dataset
    # cm_test = confusion_matrix(y_pred_test, test_label)
    # cm_test = np.array2string(cm_test)
    # count_misclassified = (test_label != y_pred_test).sum()
    # accuracy = metrics.accuracy_score(test_label, y_pred_test)
    #
    # txt = ["For test data ", 'Misclassified samples: {}'.format(count_misclassified),
    #        'Accuracy: {:.2f}'.format(accuracy)]
    # log_list.append('\n'.join(txt))
    # print(log_list[-1])

    # ===================================
    # == logging signature initialization
    # ===================================
    # split = args.split
    # # -- create dir for hyperparameter config if not already exists
    # weighted_class = ''.join(list(map(str, args.weighted_class)))
    #
    # folder = f"log/gene_disease/{args.time_stamp}/LogistircRegression/split={split}/"
    #
    # import os
    # if not os.path.exists(folder):
    #     os.makedirs(folder)
    #
    # # -- creat directory if not yet created
    # save_path = f'{folder}img/'
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    #
    # if args.log:
    #     save_path = f'emb_name={emb_name}_LogisticRegression_results.txt'
    #     print(f"writing to {save_path}...")
    #     with open(save_path, 'w') as f:
    #         txt = '\n\n'.join(log_list)
    #         f.write(txt)


# def svm(x_train,label_train, x_test, label_test, decision_func='ovo'):
def svm(data, config, decision_func='ovr', verbose=True):
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

        proba = cross_val_predict(clf, np.concatenate((x_train, x_test), axis=0), np.concatenate((label_train, label_test), axis=0), cv=int(args.cv), method='predict_proba')
        pred = proba.argmax(1)

        f = time.time()
        total = f-s
        print(f'training {total}' )

        # =====================
        # ==report performance
        # =====================
        if args.report_performance:
            save_path = f"log/gene_disease/{args.time_stamp}/classifier/svm/cross_valid={args.cv}/lr={args.lr}_d={args.dropout}_wd={args.weight_decay}/report_performance/"
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
            print(report_train)

    else:
        clf.fit(x_train, label_train)

        # args.cv = "NO"
        print(f"running {svm.__name__} without cross validation ")
        accs = []
        # --------train
        pred_train = clf.decision_function(x_train).argmax(1)
        proba_train = clf.predict_proba(x_train)
        # accs.append(accuracy(pred_train, label_train))

        # --------test
        pred_test = clf.decision_function(x_test).argmax(1) # todo Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.
        # accs.append(accuracy(pred_test, label_test))
        proba_test = clf.predict_proba(x_test)
        if verbose:
            # print(f"test prediction: {result}")
            print("[train, test] accs")
            print(accs)

        #=====================
        #==report performance
        #=====================
        if args.report_performance:
            save_path = f"log/gene_disease/{args.time_stamp}/classifier/svm/split={args.split}/lr={args.lr}_d={args.dropout}_wd={args.weight_decay}/report_performance/"
            file_name = f'cross_validation={args.cv}_emb={args.emb_name}_epoch={args.epochs}_wc={args.weighted_class}.txt'
            import os
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            report_train = performance_metrics.report_performances(
                y_true=label_train,
                y_pred=pred_train,
                y_score=proba_train,
                save_path=f'{save_path}train/',
                file_name=file_name
            )
            print(report_train)

            report_test = performance_metrics.report_performances(
                y_true=label_test,
                y_pred=pred_test,
                y_score=proba_test,
                save_path=f'{save_path}test/',
                file_name=file_name
            )
            print(report_test)

    # return pred_train, pred_test, accs


def mlp(data, config):
    '''
    run multi-layer perceptron
    input data is node with gene as its features.
    :return:
    '''
    # -- input arguments
    copd = config["data"]
    # input = config["input"]  # {disease_idx1: [[0,0,0,1,0,0],[0,1,0,0,0,0] ....], disease_idx2: [...],... }
    # y = config['label']
    # train_mask = config['train_mask']
    # test_mask = config['test_mask']
    train_input = config['train_input']
    test_input = config['test_input']
    train_label = config['train_label']
    test_label = config['test_label']
    seqential_layers = config['sequential_layers']
    # hidden_sizes = config['hidden_layers']
    epochs = config['epochs']
    args = config['args']
    param = config['param']

    # -- convert to tensor
    train_input = torch.tensor(train_input, dtype=torch.float)
    test_input = torch.tensor(test_input, dtype=torch.float)
    train_label = torch.tensor(train_label, dtype=torch.long)
    test_label = torch.tensor(test_label, dtype=torch.long)
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

        return [train_acc, test_acc, train_pred, test_pred, train_prob, test_prob]

    train_acc_hist = []
    test_acc_hist = []
    loss_hist = []
    log_list = []

    # all_x[train_index], all_labels[test_index]
    def run_epochs(train_data=None, all_train_label=None, test_data=None, all_test_label=None):
        for epoch in range(epochs):
            loss_epoch = train(train_data,all_train_label)
            train_acc, test_acc, train_pred, test_pred, train_prob, test_prob  = test(train_data,all_train_label, test_data, all_test_label)
            logging = 'Epoch: {:03d}, Train: {:.4f}, Test: {:.4f}'.format(epoch, train_acc, test_acc)
            if args.verbose:
                print(logging)
            log_list.append(logging)
            loss_hist.append(loss_epoch)
            train_acc_hist.append(train_acc)
            test_acc_hist.append(test_acc)

        return [train_pred, test_pred, train_prob, test_prob, all_train_label, all_test_label]


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
        # df = pd.DataFrame(avg_train_metrics)
        # df.to_csv(save_path+ file_name, header=True, index=False, sep='\t', mode='w')
        # os.makedirs(save_path + 'train/', exist_ok=True)
        # os.makedirs(save_path + 'test/', exist_ok=True)
        # df = pd.DataFrame(avg_train_metrics)
        # df.to_csv(save_path + 'train/' + file_name, header=True, index=False, sep='\t', mode='w')
        df = pd.DataFrame(avg_test_metrics)
        df.to_csv(save_path + 'test/' + file_name, header=True, index=False, sep='\t', mode='w')

    else:
        # train_pred, test_pred, train_prob, test_prob, all_train_label, all_test_label = run_epochs(all_x[train_index], all_labels[train_index], all_x[test_index], all_labels[test_index])
        # def run_epochs(train_data=None, all_train_label=None, test_data=None, all_test_label=None):
        train_pred, test_pred, train_prob , test_prob, all_train_label, all_test_label = run_epochs(train_input, train_label, test_input, test_label)
        #=====================
        #==report performance
        #=====================
        if args.report_performance:
            save_path = f"log/gene_disease/{args.time_stamp}/classifier/mlp/split={args.split}/lr={args.lr}_d={args.dropout}_wd={args.weight_decay}/report_performance/"
            file_name = f'emb={args.emb_name}_epoch={args.epochs}_wc={args.weighted_class}.txt'
            import os
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            #--------train
            report = performance_metrics.report_performances(all_train_label.numpy(),
                                                            train_pred.numpy(),
                                                            train_prob.detach().numpy(),
                                                            get_avg_total=True)
            print(report)
            #--------test
            report = performance_metrics.report_performances(all_test_label.numpy(),
                                                           test_pred.numpy(),
                                                           test_prob.detach().numpy(),
                                                           get_avg_total=True)
            print(report)
            # report = performance_metrics.report_performances(
            #     y_true=train_label.numpy(),
            #     y_pred=model(train_input).max(1)[1].numpy(),
            #     y_score=model(train_input).detach().numpy(),
            #     save_path=f'{save_path}train/',
            #     file_name=file_name
            # )
            # print(report)
            # report = performance_metrics.report_performances(
            #     y_true=test_label.numpy(),
            #     y_pred=model(test_input).max(1)[1].numpy(),
            #     y_score=model(test_input).detach().numpy(),
            #     save_path=f'{save_path}test/',
            #     file_name=file_name
            # )
            # print(report)


    # # =====================
    # # ==logging and write2files
    # # =====================
    # split = args.split
    #
    # # -- create dir for hyperparameter config if not already exists
    # weighted_class = ''.join(list(map(str, args.weighted_class)))
    #
    # HP = f'lr={args.lr}_d={args.dropout}_wd={args.weight_decay}'
    # folder = f"log/gene_disease/{args.time_stamp}/mlp/split={split}/{HP}/"
    #
    # import os
    # if not os.path.exists(folder):
    #     os.makedirs(folder)
    #
    # # if args.add_features:
    # if args.emb_name != "no_feat":
    #     feat_stat = "YES"
    # else:
    #     feat_stat = "NO"
    #
    # if args.pseudo_label_all:
    #     pseudo_label_stat = "ALL"
    # elif args.pseudo_label_topk:
    #     pseudo_label_stat = "TOP_K"
    # elif args.pseudo_label_topk_with_replacement:
    #     pseudo_label_stat = "TOP_K_WITH_REPLACEMENT"
    # else:
    #     pseudo_label_stat = "NONE"
    #
    # T_param = ','.join([str(param['T1']), str(param['T2'])])
    # # -- creat directory if not yet created
    # save_path = f'{folder}img/'
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # if args.plot_all is True:
    #     args.plot_loss = True
    #     args.plot_no_train = True
    #     args.plot_train = True
    #
    # if args.plot_loss:
    #     # ======================
    #     # == plot loss and acc vlaue
    #     # ======================
    #     plt.figure(1)
    #     # -- plot loss hist
    #     plt.subplot(211)
    #     plt.plot(range(len(loss_hist)), loss_hist)
    #     plt.ylabel("loss values")
    #     plt.title("loss history")
    #
    #     # -- plot acc hist
    #     plt.subplot(212)
    #     plt.plot(range(len(train_acc_hist)), train_acc_hist)
    #     plt.plot(range(len(test_acc_hist)), test_acc_hist)
    #     plt.ylabel("accuracy values")
    #     plt.title("accuracy history")
    #     print(
    #         "writing to  " + save_path + f"LOSS_ACC_feat={feat_stat}_pseudo_label={pseudo_label_stat}_wc=[{weighted_class}]_T=[{T_param}]_topk={args.topk}.png")
    #     plt.savefig(
    #         save_path + f'ACC_feat={feat_stat}_pseudo_label={pseudo_label_stat}_wc=[{weighted_class}]_T=[{T_param}]_topk={args.topk}.png')
    #     plt.show()
    #
    #
    # if args.log:
    #     # --train_mask f1,precision,recall
    #     train_pred = mlp(train_input).max(1)[1]
    #     train_f1 = f1_score(train_label, train_pred, average='micro')
    #     train_precision = precision_score(train_label, train_pred, average='micro')
    #     train_recall = recall_score(train_label, train_pred, average='micro')
    #
    #     # -- test_mask f1,precision,recall
    #     test_pred = mlp(test_input).max(1)[1]
    #     test_f1 = f1_score(test_label, test_pred, average='micro')
    #     test_precision = precision_score(test_label, test_pred, average='micro')
    #     test_recall = recall_score(test_label, test_pred, average='micro')
    #
    #     save_path = f'{folder}ACC_feat={feat_stat}_pseudo_label={pseudo_label_stat}_wc={weighted_class}_topk={args.topk}.txt'
    #     print(f"writing to {save_path}...")
    #     with open(save_path, 'w') as f:
    #         txt = '\n'.join(log_list)
    #         f.write(txt)
    #
    #     cm_train = confusion_matrix(mlp(train_input).max(1)[1], train_label)
    #     cm_test = confusion_matrix(mlp(test_input).max(1)[1], test_label)
    #
    #     # formatter = {'float_kind': lambda x: "%.2f" % x})
    #     cm_train = np.array2string(cm_train)
    #     cm_test = np.array2string(cm_test)
    #
    #     save_path = f'{folder}CM_feat={feat_stat}_pseudo_label={pseudo_label_stat}_wc={weighted_class}_topk={args.topk}.txt'
    #     print(f"writing to {save_path}...")
    #
    #     # txt = 'class int_rep is [' + ','.join(list(map(str, np.unique(data.y.numpy()).tolist()))) + ']'
    #     txt = 'class int_rep is [' + ','.join([str(i) for i in range(len(copd.labels2idx().values()))]) + ']'
    #     txt = txt + '\n\n' + "training cm" + '\n' + cm_train + '\n' \
    #           + f"training_accuracy ={log_list[-1].split(',')[1]}" + '\n' \
    #           + f"training_f1       ={train_f1}" + '\n' \
    #           + f"training_precision={train_precision}" + '\n' \
    #           + f"training_recall   ={train_recall}" + '\n'
    #
    #     txt = txt + '\n\n' + "test cm" + '\n' + cm_test + '\n' \
    #           + f"test_accuracy ={log_list[-1].split(',')[2]}" + '\n' \
    #           + f"test_f1       ={test_f1}" + '\n' \
    #           + f"test_precision={test_precision}" + '\n' \
    #           + f"test_recall   ={test_recall}" + '\n'
    #
    #     with open(save_path, 'w') as f:
    #         f.write(txt)


# def mlp( data, config, **kwargs):
#     '''
#
#     :param x: type torch
#     :param label:
#     :param num_feat:
#     :param test_data:
#     :param layers:
#     :return:
#     '''
#     # x_train, label_train, x_test, label_test
#     x_train, label_train = data.x[data.train_mask], data.y[data.train_mask]  # return tensor
#     x_test, label_test = data.x[data.test_mask], data.y[data.test_mask]  # return tensor
#     bs = config['bs']
#     start_i = 0
#     end_i = bs
#     epoch = config['epoch']
#     lr = config['lr']
#     n = epoch
#
#     num_instance = x_train.shape[0]
#     num_feat = x_train.shape[1]
#     num_class = np.unique(x_train.numpy())
#
#     if kwargs['layers'] is None:
#         # randomly choose sequential layer
#         kwargs["layers"] = [  # i am not sure if my sequential is correct
#             nn.Linear(num_instance, num_feat),
#             nn.ReLU(),
#             nn.Linear(64, 32),
#             nn.ReLU(),
#             nn.Linear(32, num_class),
#             nn.LogSoftmax(dim=1)
#         ]
#
#     model = nn.Sequential(*kwargs) # this may cause error
#     # loss_fn = nn.LogSoftmax(dim=1)
#
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#
#     if kwargs.get('weightclass', None) is None:
#         weightclass = [1 for i in num_class]
#
#     def train(xb_train, yb_train):
#
#         model.train()
#         # loss_output = F.nll_loss(xb_train, yb_train, weight=weightclass,
#         #                          reduction="mean")
#
#         optimizer.zero_grad()
#         loss_output.backward()
#         optimizer.step()
#         return loss_output
#
#     def test(data):
#         '''
#
#         :param data: has the following structure
#             [[xb_train,yb_train],[[xb_test,yb_test]]]
#         :return: measurement matric
#             accs
#         '''
#         #TODO here>> lets check how to do test in mlp
#         # >I may have to used trian loader here too
#         # >use trainLoader
#         model.eval()
#         accs = []
#         preds = []
#
#         for d in data:
#             x, y = d[0], d[1]
#             pred = model(x).max(1)[1]
#             # acc = pred.eq(y).sum().item() / x.shape[0]
#             acc = accuracy(pred, y)
#             accs.append(acc) # contain [train_acc_list, test_acc_list]
#             preds.append(pred)  # contain [train_acc_list, test_acc_list]
#         return accs, preds
#
#     #=====================
#     #== training model
#     #=====================
#
#     # train_dataset = Dataset(x_train, label_train) # todo gene_disease dataset have to be of type Dataset
#     # train_dataset = zip(x_train,label_train ) # todo this is most likely wrong but I will fix it when i finally run the function
#     # test_dataset = zip(x_test,label_test )
#
#     # train_loader = DataLoader(dataset=train_dataset, batch_size=bs, shuffle=False)
#     # test_loader = DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False)
#     accs = None
#     preds = None
#     for i in range((n-1)//bs+1):
#         xb_train = x_train[i * bs: i * bs + bs]
#         yb_train = label_train[i * bs: i * bs + bs]
#
#         xb_test = x_train[i * bs: i * bs + bs]
#         yb_test = label_train[i * bs: i * bs + bs]
#
#         loss_output = train(xb_train, yb_train)
#         accs, preds = test([[xb_train,yb_train],[xb_test,yb_test]])
#
#     return accs, preds # return prediction and accuaracy of train and test of the last epoc.


# def random_forest(x_train,label_train,x_test,label_test, bs,epoch, lr):
def random_forest(data, config, evaluate=False):
    '''
    url: https://towardsdatascience.com/an-implementation-and-explanation-of-the-random-forest-in-python-77bf308a9b76
    :param x_train:
    :param label_train:
    :param x_test:
    :param label_test:
    :return:
    '''
    # x_train, label_train = data.x[data.train_mask], data.y[data.train_mask]
    # x_test, label_test = data.x[data.test_mask], data.y[data.test_mask]

    x_train, label_train = config['train_input'], config['train_label']
    x_test, label_test = config['test_input'], config['test_label']

    x_train = x_train.type(torch.float).numpy()
    x_test = x_test.type(torch.float).numpy()
    label_train = label_train.type(torch.long).numpy()
    label_test = label_test.type(torch.long).numpy()

    from sklearn.ensemble import RandomForestClassifier
    RSEED = 50

    # Create the model with 100 trees
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
        if args.report_performance:
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
        if args.report_performance:
            save_path = f"log/gene_disease/{args.time_stamp}/classifier/rf/split={args.split}/lr={args.lr}_d={args.dropout}_wd={args.weight_decay}/report_performance/"
            file_name = f'emb={args.emb_name}_epoch={args.epochs}_wc={args.weighted_class}.txt'
            import os
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            report = performance_metrics.report_performances(
                y_true=label_train,
                y_pred=train_rf_predictions,
                y_score=train_rf_probs,
                save_path=f'{save_path}train/',
                file_name=file_name
            )
            print(report)
            report = performance_metrics.report_performances(
                y_true=label_test,
                y_pred=rf_predictions,
                y_score=rf_probs,
                save_path=f'{save_path}test/',
                file_name=file_name
            )
            print(report)


    #=====================
    #==below code is old plotting and logging style
    #=====================

    # from sklearn.metrics import precision_score, recall_score
    # import matplotlib.pyplot as plt


    # # Plot formatting
    # plt.style.use('fivethirtyeight')
    # plt.rcParams['font.size'] = 18
    # if evaluate == True:
    #     def evaluate_model(predictions, probs, train_predictions, train_probs):
    #         """Compare machine learning model to baseline performance.
    #         Computes statistics and shows ROC curve."""
    #         baseline = {}
    #
    #         baseline['recall'] = recall_score(label_test,
    #                                           [1 for _ in range(len(label_test))], average='micro')
    #
    #         baseline['precision'] = precision_score(label_test,
    #                                                 [1 for _ in range(len(label_test))], average='micro')
    #         baseline['roc'] = 0.5
    #
    #         results = {}
    #
    #         results['recall'] = recall_score(label_test, predictions, average='micro')
    #         results['precision'] = precision_score(label_test, predictions, average='micro')
    #         # results['roc'] = roc_auc_score(label_test, probs)
    #
    #         train_results = {}
    #         train_results['recall'] = recall_score(label_train, train_predictions, average='micro')
    #         train_results['precision'] = precision_score(label_train, train_predictions, average='micro')
    #         # train_results['roc'] = roc_auc_score(label_train, train_probs)
    #
    #         for metric in ['recall', 'precision']:
    #             print(
    #                 f'{metric.capitalize()} Baseline: {round(baseline[metric], 2)} Test: {round(results[metric], 2)} Train: {round(train_results[metric], 2)}')
    #
    #         # Calculate false positive rates and true positive rates
    #         # base_fpr, base_tpr, _ = roc_curve(label_test, [1 for _ in range(len(label_test))])
    #         # model_fpr, model_tpr, _ = roc_curve(label_test, probs)
    #
    #         plt.figure(figsize=(8, 6))
    #         plt.rcParams['font.size'] = 16
    #
    #         #------Plot both curves
    #         # plt.plot(base_fpr, base_tpr, 'b', label='baseline')
    #         # plt.plot(model_fpr, model_tpr, 'r', label='model')
    #         # plt.legend()
    #         # plt.xlabel('False Positive Rate')
    #         # plt.ylabel('True Positive Rate')
    #         # plt.title('ROC Curves')
    #         # plt.show()
    #
    #     def plot_confusion_matrix(cm, classes,
    #                               normalize=False,
    #                               title='Confusion matrix',
    #                               cmap=plt.cm.Oranges):
    #         """
    #         This function prints and plots the confusion matrix.
    #         Normalization can be applied by setting `normalize=True`.
    #         Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    #         """
    #         if normalize:
    #             cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #             print("Normalized confusion matrix")
    #         else:
    #             print('Confusion matrix, without normalization')
    #
    #         print(cm)
    #
    #         # Plot the confusion matrix
    #         plt.figure(figsize=(10, 10))
    #         plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #         plt.title(title, size=24)
    #         plt.colorbar(aspect=4)
    #         tick_marks = np.arange(len(classes))
    #         plt.xticks(tick_marks, classes, rotation=45, size=14)
    #         plt.yticks(tick_marks, classes, size=14)
    #
    #         fmt = '.2f' if normalize else 'd'
    #         thresh = cm.max() / 2.
    #
    #         # Labeling the plot
    #         for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #             plt.text(j, i, format(cm[i, j], fmt), fontsize=20,
    #                      horizontalalignment="center",
    #                      color="white" if cm[i, j] > thresh else "black")
    #
    #         plt.grid(None)
    #         plt.tight_layout()
    #         plt.ylabel('True label', size=18)
    #         plt.xlabel('Predicted label', size=18)
    #
    #     evaluate_model(rf_predictions, rf_probs, train_rf_predictions, train_rf_probs)
    #     plt.savefig('roc_auc_curve.png')
    #
    #     from sklearn.metrics import confusion_matrix
    #     import itertools
    #
    #     # =====================
    #     # ==measurement matric
    #     # =====================
    #
    #     # --------Confusion matrix
    #     cm = confusion_matrix(label_test, rf_predictions)
    #
    #     # =====================
    #     # ==plotting
    #     # =====================
    #     plot_confusion_matrix(cm, classes=['Poor Health', 'Good Health'],
    #                           title='Health Confusion Matrix')
    #
    #     plt.savefig('output/gene_disease/baseline/cm.png')



