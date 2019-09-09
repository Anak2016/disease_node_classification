from utility_code.python_lib_essential import *

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

#===============
#==utility
#===============
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


