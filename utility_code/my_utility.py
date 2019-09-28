from utility_code.python_lib_essential import *
from  sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
from arg_parser import *
# import plotting
from scipy import interp
#===============================
#==Pytorch library essential
#===============================

def setup_essential(ROOT_DIR =None, ignore_root=False):
    '''
    THIS FUNCION MAY NOT BE NEEDED ANYMORE,  IT IS LEFT HERE FOR COMPATIBILITY REASON

    :param Root_DIR:path to directory of the current project
    :return:
    '''
    # #--sysmtem control such as path
    import os, sys
    if not ignore_root:
        if ROOT_DIR is None:
            raise ValueError("Root_DIR must already exist.")
        if not os.path.exists(ROOT_DIR):
            raise ValueError("Root_DIR must already exist.")
        os.chdir(ROOT_DIR)
        # sys.path.append(ROOT_DIR)
        # sys.path.insert(1,ROOT_DIR)

    # USER = os.environ['USERPROFILE']

    # os.chdir(f'{USER}\\PycharmProjects\\my_utility')
    # (f'{USER}\\PycharmProjects\\my_utility')

#=====================
#==plotting
#=====================
def plot_figures(config, save_path=None, file_name=None):
    file_name = file_name.split('.')[:-1]
    file_name = '/'.join(file_name) + ".png"
    print(f"save plot to {save_path}{file_name}...")
    num_fig = len(config.keys())

    if num_fig <= 3:
        num_col = num_fig
        num_row = 1
    elif num_fig / 3 == int(num_fig / 3):
        num_col = 3
        num_row = int(num_fig / 3)
    else:
        num_col = 3
        num_row = int(num_fig / 3) + 1

    assert num_row * num_col >= num_fig, "num_row * num_col must be more than or equal to num_fig"

    fig, axes = plt.subplots(num_row, num_col)

    # num_fig = len(config.keys())
    for i, (c, v) in enumerate(config.items()):
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
        if num_row == 1 and num_col == 1:
            plot_args = v.get('plot', [{'args': []}])
            plot_kwargs = v.get('plot', [{'kwargs': []}])
            legend_kwargs = v.get('legend', [{'kwargs': []}])

            for i, j, k in zip(plot_args, plot_kwargs, legend_kwargs):
                axes.plot(*i.get('args', []), **j.get('kwargs', {}))
                axes.legend(**k.get('kwargs', {}))

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
                axes[col].plot(*i.get('args', []), **j.get('kwargs', {}))
                axes[col].legend(**k.get('kwargs', {}))

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
                axes[row, col].plot(*i.get('args', []), **j.get('kwargs', {}))
                axes[row, col].legend(**k.get('kwargs', {}))

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


#=====================
#==report performance
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
    labels, cnt = np.unique(y_pred,return_counts=True)

    # n_classes = len(labels)
    n_classes = len(np.unique(y_true))
    pred_cnt = pd.Series(cnt, index=labels)


    metrics_summary = precision_recall_fscore_support(
            y_true=y_true,
            y_pred=y_pred,
            labels=labels)

    '''
    >weight depends on number of positive true per binary confusion matrix. 
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

    #--------accuracy
    true_label_per_class_ind = [np.where(y_true == i)[0] for i in labels]
    pred_score_per_class = np.array([y_score[i].argmax(1) for i in true_label_per_class_ind])
    true_label_per_class = np.array([y_true[i] for i in true_label_per_class_ind])

    # class_report_df['acc'] = pd.Series(
    #     [np.sum(np.equal(i, j)) / j.shape[0] for i, j in zip(pred_score_per_class, true_label_per_class)], index=labels)
    class_report_df['acc'] = pd.Series(
        [np.sum(np.equal(i, j)) for i, j in zip(pred_score_per_class, true_label_per_class)], index=labels)
    class_report_df.loc['avg / total']['acc'] =  class_report_df['acc'].sum() / (y_true.shape[0])
    class_report_df['acc'][:-1] = class_report_df['acc'][:-1].divide(class_report_df['support'][:-1])
    config = {}
    if not (y_score is None):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
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
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        df = pd.DataFrame(class_report_df)  # output before first epoch
        # tmp = save_path+file_name
        # with open(tmp,'w') as f:
        #     print(f"writing to {tmp}...")
        #     f.write(df.__repr__())
        df.to_csv(save_path+file_name, header=True, index=False,  sep='\t', mode='w')

    return class_report_df

#=====================
#==model selection
#=====================

#===============
#==utility
#===============
def write2files(data, path="./data/{args.time_stamp}/gene_disease/", file_name=None, type='df'):
    '''
    usecase:
        my_utils.write2files(gene_disease,
                            path=f'data/gene_disease/{args.time_stamp}/processed/rep/',
                            file_name=f"rep_copd_label_edges{self.time_stamp}.txt")

    :parame data: content to be written in files
    :param path:
    :param dataset:
    :param type: type of content arg;  df, np, dict.
    :return:
    '''
    print(f'write to {path + file_name}...')

    if file_name is None:
        raise ValueError('In write2files, dataset is not given as an argument')
    if isinstance(data, pd.DataFrame):
        data.to_csv(path + file_name, sep='\t', index=False, header=None)
    elif isinstance(data, np.ndarray):
        pd.DataFrame(data, dtype="U")  # convert to type string
        data.to_csv(path + file_name, sep='\t', index=False, header=None)
    elif isinstance(data, dict):
        pd.DataFrame.from_dict(data, dtype="U", columns=None, orient='columns')  # convert to type string
        data.to_csv(path + file_name, sep='\t', index=False, header=None)
    else:
        raise ValueError('type of given data are not accpeted by write2files function')



def flatten(container):
    '''
    usecase example
        nests = [1, 2, [3, 4, [5],['hi']], [6, [[[7, 'hello']]]]]
        print list(flatten(nests))
    :param container:
    :return:
    '''
    for i in container:
        if isinstance(i, (list,tuple)):
            for j in flatten(i):
                yield j
        else:
            yield i


def flatten_list(alist):
    '''

    :param alist: type == list
    :return:
    '''
    from itertools import chain
    return list(chain(*alist))

def is_file_existed(files):
    if isinstance(files, list):
        for f in files:
            # C:\Users\Corland\PycharmProjects\my_utility\weekly_productivity_log
            assert os.path.exists(f) and os.path.isfile(f), f'{f} does not exist.'
    else:
        file = files
        assert os.path.exists(file) and os.path.isfile(file), f'{file} does not exist.'
    return True

def is_dir_existed(dirs):
    if isinstance(dirs, list):
        for f in dirs:
            # C:\Users\Corland\PycharmProjects\my_utility\weekly_productivity_log
            assert os.path.exists(f), f'{f} does not exist.'

    else:
        d = dirs
        assert os.path.exists(d), f'{d} does not exist.'

    return True

def find_date(txt, use_django_datetime=True):
    '''
    # partially broken it does not work in some cases

    find date from any string

    param txt: a str txt that contains date and time to be extracted
    return daily_date: type == datetime.
    '''
    import pytz
    daily_date = []

    matches = datefinder.find_dates(txt)  # return type datetimes
    # print(list(matches))
    assert len(list(matches)) > 0 , "datefinder.findd_dates(txt) cannot find date in txt"

    try:
        for i in matches:
            # daily_date = f'{i.month}/{i.day}/{i.year}'
            # daily_date.append(i.strftime("%Y-%m-%d"))
            if use_django_datetime:
                # eg
                #   django_datetime       = datetime.datetime(2013, 11, 20, 20, 8, 7, 127325, tzinfo=pytz.UTC)
                #   naive_datetime_object = datetime.datetime(2013, 11, 20, 20, 9, 26, 423063)
                # dt.strftime('%Y-%m-%dT%H:%M:%S.%f') + 'Z'
                datetime_obj = i
                i = i.strptime(datetime_obj, "%Y-%m-%dT%H:%M:%S.%f").replace(tzinfo=pytz.UTC)

            daily_date.append(i)
    except:
        print('daily_date variable is not datetime.')

    assert len(daily_date) > 0 , 'daily_date must have len more than 0'

    return daily_date

class Counter():
    '''usecase
    count = Counter().counting

    for i in range(10):
        print(count())
    '''
    def __init__(self):
        self.count = 0
    def counting(self):
        self.count += 1
        return self.count

def pause():
    print("done")
    exit()

def display2screen(*args,**kwargs):
    stop = kwargs.get('stop', True)

    if args is not None:
        for arg in args:
            print(arg)

    for k, v in kwargs.items():
        print(f"{k}: {v}")
    if stop:
        pause()

def timer(func, *args, **kwargs):
    print(f"running {func.__name__}")
    import time
    s = time.time()
    func(*args, **kwargs)
    f = time.time()
    total = f-s
    print(f"{func.__name__} takes in total {total} s to run")
    return total

# def json_indent(myjson):
#     return json.dumps({'4': 5, '6': 7}, sort_keys=True, indent=4, separators=(',', ': '))

def is_json(myjson):
    '''
    :param myjson: str: str that have json format
    :return:
    '''
    import json
    import ast
    try:
        if isinstance(myjson, str):
            # myjson = ast.literal_eval(myjson)  # accept cases where double quotes and finl comma is used are used
            myjson = myjson.replace("'", '"')
            json_object = json.loads(myjson) # return string
        else:
            json_object = json.dumps(myjson)

    except ValueError as e:
        print("provided file doesn't have correct json format")
        print(f"in is_json func of utility file, ValueError: {e}")
        return False
    print("provided file has json format !!")
    return True

def refine_str(txt):
    '''

    convert str to desired format.
    desired format must obey following rules
        > no special character
            :only " " (aka blank space) is allowed
        > character must be all lower case
    param txt: str or list
    :return: desired format str
    '''

    # import timeit
    #
    # bad_chars = '(){}<>'
    #
    # setup = """import re
    #      import string
    #      s = 'Barack (of Washington)'
    #      bad_chars = '(){}<>'
    #      rgx = re.compile('[%s]' % bad_chars)"""

    #--------fastest way to strip multiple chracter from str, but it is not so reable
    # timer = timeit.Timer('s.translate(string.maketrans("", "", ), bad_chars)', setup=setup)
    # print("string.translate: ", timer.timeit(100000))

    #--------second fastest
    # timer = timeit.Timer("o= rgx.sub('', s)", setup=setup)
    # print("Regular expression: ", timer.timeit(100000))

    #--------thrid fastest
    # timer = timeit.Timer('for c in bad_chars: s = s.replace(c, "")', setup=setup)
    # print("Replace in loop: ", timer.timeit(100000))

    #--------fouth fastest
    # timer = timeit.Timer('o = "".join(c for c in s if c not in bad_chars)', setup=setup)
    # print("List comprehension: ", timer.timeit(100000))

    bad_chars = '(){}<>!@#$%^&*_+-=:;'"?/.,\|][~`"
    rgx = re.compile('[%s]' % bad_chars)
    if len(txt)>0:
        if isinstance(txt, list):
            tmp = [rgx.sub('', i) for i in txt]
            tmp = [i.lower() for i in tmp]
            txt = tmp
        else:
            txt = rgx.sub('', txt)

    return txt

def timer_wrapped(func, *args, **kwargs):
    import time
    s = time.time()
    func(*args, **kwargs)
    f = time.time()
    total = f-s
    return total

def provide_progress_bar(function, estimated_time, tstep=0.2, tqdm_kwargs={}, args=[], kwargs={}):
    """Tqdm wrapper for a long-running function

    args:
        function - function to run
        estimated_time - how long you expect the function to take
        tstep - time delta (seconds) for progress bar updates
        tqdm_kwargs - kwargs to construct the progress bar
        args - args to pass to the function
        kwargs - keyword args to pass to the function
    ret:
        function(*args, **kwargs)
    """
    ret = [None]  # Mutable var so the function can store its return value
    def myrunner(function, ret, *args, **kwargs):
        ret[0] = function(*args, **kwargs)
    
    thread = threading.Thread(target=myrunner, args=(function, ret) + tuple(args), kwargs=kwargs)
    pbar = tqdm.tqdm(total=estimated_time, **tqdm_kwargs)

    thread.start()
    while thread.is_alive():
        thread.join(timeout=tstep)
        pbar.update(tstep)
    pbar.close()
    return ret[0]


def progress_wrapped(estimated_time, tstep=0.2, tqdm_kwargs={}):
    """Decorate a function to add a progress bar
    usecase
    @progress_wrapped(estimated_time=5)
    def some_func():
        ...
    """
    def real_decorator(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            return provide_progress_bar(function, estimated_time=estimated_time, tstep=tstep, tqdm_kwargs=tqdm_kwargs, args=args, kwargs=kwargs)
        return wrapper
    return real_decorator



# ===================
# == Pytorch
#
#train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
#train_dl = WrappedDataLoader(train_dl, preprocess)
#valid_dl = WrappedDataLoader(valid_dl, preprocess)
#
#model, opt = get_model()
#fit(epochs, model, loss_func, opt, train_dl, valid_dl)
#
# ===================
#---use_cpu
def use_gpu(gpu=True):
    if gpu:
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")
    return dev

#----get_data
def get_data(train_ds, valid_ds, bs, valid_mul=2):
    '''

    :param train_ds: output of torch.utils.data
        such as
            train_ds = TensorDataset(x_train, y_train)
            xb,yb = train_ds[start_i : end_i] # size = batch

    :param valid_ds:
    :param bs: batch size
    :param valid_mul: batch size of valid to batch size of train
    :return:
    '''
    from torch.utils.data import DataLoader
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * valid_mul),
    )
#----loss_batch
def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)

#----fit pytorch
def fit(epochs, model, loss_func, opt, train_dl, valid_dl):

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

            model.eval()

        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        print(epoch, val_loss)
#----WrappeddataLoader Preprocessing
class WrappedDataLoader:

    def __init__(self, dl, func):
        '''
        use case
            train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
            train_dl = WrappedDataLoader(train_dl, preprocess)
            valid_dl = WrappedDataLoader(valid_dl, preprocess)

        :param dl: Dataloader
            eg. output of my_utility.get_data()
        :param func: preprocessor func
        '''
        self.dl = dl
        self.func = func

    # def __getitem__(self, ind):
    #     return self.dl[ind]

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))


