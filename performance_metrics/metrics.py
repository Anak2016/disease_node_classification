import sys,os
USER = os.environ['USERPROFILE']
sys.path.insert(1,f'{USER}\\PycharmProjects\\my_utility')

from utility_code.my_utility import *
from utility_code.python_lib_essential import *

from scipy import interp
from  sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
from arg_parser import *
import plotting

#=====================
#==full performance report
#=====================
def report_performances(y_true, y_pred, y_score=None, average='micro', save_path =None, file_name=None, get_avg_total=False):
    '''
    this function is an improvement on "classification_report" (function in sklearn.metrics)

    classification_report doesn't report roc_curve
    while
    report_performance does report roc_curve

    note that accuracy can be calculated by sum pred of all the classes and divide it by avg pred. (this fact need to be confirmede!)
    usecase:
        report_with_auc = class_report(
                            y_true=y_test,
                            y_pred=model.predict(X_test),
                            y_score=model.predict_proba(X_test))

        report_test = performance_metrics.report_performances(
            y_true=label_test,
            y_pred=pred_test,
            y_score=proba_test,
            save_path=f'{save_path}test/',
            file_name=file_name
        )

    :param y_true:
    :param y_pred:
    :param y_score:
    :param average:
    :param get_avg_total:
        if true: return roc_auc['avg/total']
        if false: return str representation of roc_auc
    :return:
    '''
    if y_true.shape != y_pred.shape:
        print("Error! y_true %s is not the same shape as y_pred %s" % (
              y_true.shape,
              y_pred.shape)
        )
        return

    lb = LabelBinarizer() # ???

    if len(y_true.shape) == 1:
        lb.fit(y_true)

    #Value counts of predictions
    #TODO here>> this may be a problem
    labels, cnt = np.unique(y_pred,return_counts=True)
    all_labels = np.unique(y_true)
    for i in all_labels:
        if i not in labels:
            labels = np.hstack((labels,i))
            cnt = np.hstack((cnt,0))

    # n_classes = len(labels)
    n_classes = len(np.unique(y_true))
    pred_cnt = pd.Series(cnt, index=labels)


    metrics_summary = precision_recall_fscore_support(
            y_true=y_true,
            y_pred=y_pred,
            labels=labels) #

    '''
    >weight depends on number of positive true per binary confusion matrix. (one vs all) 
    >(there are (n choose 2) number of confusion matrix where n is number of classes in multiple class problems)
    >formular of average='weighted' is shown below.
         (value of measurment_metric of class i) * (number of positive true label.) / (number of all classes) 
    '''
    avg = list(precision_recall_fscore_support(
            y_true=y_true,
            y_pred=y_pred,
            average='weighted'))

    metrics_sum_index = ['precision', 'recall', 'f1-score', 'support']
    class_report_df = pd.DataFrame(
        list(metrics_summary),
        index=metrics_sum_index,
        columns=labels)

    support = class_report_df.loc['support'] # The support is the number of occurrences of each class in y_true. todo why is it not a full number
    total = support.sum()
    class_report_df['avg / total'] = avg[:-1] + [total]

    class_report_df = class_report_df.T
    class_report_df['pred'] = pred_cnt # total number of pred of each class
    class_report_df['pred'].iloc[-1] = total


    config = {}
    #--------accuracy

    true_label_per_class_ind = [np.where(y_true == i)[0] for i in labels] # labels of each predicted instances.
    if y_score is not None:
        pred_score_per_class = np.array([y_score[i].argmax(1) for i in true_label_per_class_ind])
    else:
        pred_score_per_class = np.array([y_pred[i] for i in true_label_per_class_ind])
    true_label_per_class = np.array([y_true[i] for i in true_label_per_class_ind])

    # class_report_df['acc'] = pd.Series(
    #     [np.sum(np.equal(i, j)) / j.shape[0] for i, j in zip(pred_score_per_class, true_label_per_class)], index=labels)
    class_report_df['acc'] = pd.Series(
        [np.sum(np.equal(i, j)) for i, j in zip(pred_score_per_class, true_label_per_class)], index=labels)

    if average == 'micro':
        #TODO here>> do i need to have both line of code below. i feel like i only need one
        class_report_df.loc['avg / total']['acc'] =  class_report_df['acc'].sum() / (y_true.shape[0])
        class_report_df['acc'][:-1] = class_report_df['acc'][:-1].divide(class_report_df['support'][:-1])

    if average == 'macro':
        #TODO here>> add this
        class_report_df['acc'][:-1] = class_report_df['acc'][:-1].divide(class_report_df['support'][:-1])
        class_report_df['avg / total']['acc'] = class_report_df['acc'][:-1].sum() / (y_true.shape[0])
            # class_report_df.loc['avg / total']['acc'] =


    if not (y_score is None):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        labels = sorted(labels)
        for label_it, label in enumerate(labels):
            fpr[label], tpr[label], _ = roc_curve(
                (y_true == label).astype(int),
                y_score[:, label_it])

            roc_auc[label] = auc(fpr[label], tpr[label])
            fig = {
                f"class_{label_it}": {
                    'x_label': 'False Positive Rate',
                    'y_label': 'True Positive Rate',
                    'x_lim': [-0.1, 1.1],
                    "y_lim": [-0.1, 1.1],
                    'legend': [{"kwargs": {"loc": "lower right"}}],
                    'plot': [{"args": [fpr[label], tpr[label]],
                            "kwargs": {"label": 'ROC curve (area = %0.2f)' % roc_auc[label]}}]
                }
            }
            config.update(fig)
            # plotting.plot_figures(config)
        # =====================
        # == plot_roc_auc
        # =====================
        if args.plot_roc:
            plotting.plot_figures(config, f'{save_path}img/', file_name)

        if average == 'micro':
            '''none_class is assumed to be at the end of the
             eg. if there are 3 classes,
                 1 row of y_score with out none class has the followin format
                          [class1, class2,class3]
                 1 row of y_score with none class has the followin format
                          [class1, class2,class3, none_class]'''
            if np.unique(y_true).shape[0] != y_score.shape[1]: # this helps with None class
                y_score = y_score[:, :np.unique(y_true).shape[0]]

            if n_classes <= 2:
                fpr["avg / total"], tpr["avg / total"], _ = roc_curve(
                    lb.transform(y_true).ravel(),
                    y_score[:, 1].ravel())
            else:
                #TODO here>> why do I need to convert it to two classes and feed it to roc_curve
                fpr["avg / total"], tpr["avg / total"], _ = roc_curve(
                        lb.transform(y_true).ravel(),
                        y_score.ravel())

            # fpr["avg / total"], tpr["avg / total"], _ = roc_curve(
            #     lb.transform(y_true).ravel(),
            #     y_score.ravel())

            roc_auc["avg / total"] = auc( fpr["avg / total"], tpr["avg / total"])

        elif average == 'macro':
            # First aggregate all false positive rates
            all_fpr = np.unique(np.concatenate([
                fpr[i] for i in labels]
            ))

            # Then interpolate all ROC curves at this points
            mean_tpr = np.zeros_like(all_fpr)
            for i in labels:
                mean_tpr += interp(all_fpr, fpr[i], tpr[i])

            # Finally average it and compute AUC
            mean_tpr /= n_classes

            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr

            roc_auc["avg / total"] = auc(fpr["macro"], tpr["macro"])

        class_report_df['AUC'] = pd.Series(roc_auc)

    if get_avg_total:
        return class_report_df

    if save_path is not None:
        # if not os.path.exists(save_path):
        #     os.mkdir(save_path)
        os.makedirs(save_path, exist_ok=True)

        df = pd.DataFrame(class_report_df)  # output before first epoch
        df = df.round(6)


        with open(save_path+file_name, 'w') as f:
            f.write(df.__repr__())

        # df.to_csv(save_path+file_name, header=True, index=True,  sep='\t', mode='w')
        # print(df)
        # print(save_path+file_name)
        # exit()
    return class_report_df


#=====================
#==measurement matrix
#=====================
def accuracy(pred, label):
    '''

    :param pred: type = torch, numpy
        dim = [# instances, # features]
    :param label: type = torch, numpy
        dim = [# instances]
    :return:
    '''
    if not isinstance(pred, type(torch.rand(1))):
        pred = torch.tensor(pred, dtype=torch.double)
    if not isinstance(label, type(torch.rand(1))):
        label = torch.tensor(label, dtype=torch.double)
    pred = pred.type(torch.double)
    label = label.type(torch.double)
    # get index of max index of each row
    value, indices = pred.max(1)

    indices = indices.type(torch.double)

    return   indices.eq(label).sum().item()/ indices.shape[0]

