import sys,os
USER = os.environ['USERPROFILE']
sys.path.insert(1,f'{USER}\\PycharmProjects\\my_utility')

from utility_code.my_utility import *
from utility_code.python_lib_essential import *

from hyper_params_search.hp_tuning import *
# from hp_tuning import *

if __name__ == '__main__':
    from sklearn import svm
    from sklearn.model_selection import GridSearchCV
    def svc_param_selection(X, y, nfolds):
        Cs = [0.001, 0.01, 0.1, 1, 10]
        gammas = [0.001, 0.01, 0.1, 1]
        param_grid = {'C': Cs, 'gamma': gammas}
        grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nfolds)
        grid_search.fit(X, y)
        display2screen( grid_search.best_params_, stop=False)
        return grid_search.best_params_
    x = [[0, 0], [1, 1], [2,2], [-1,-1]]
    y = [0, 1, 1, 0]
    result = svc_param_selection(x, y, 3)
