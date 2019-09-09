import sys,os
USER = os.environ['USERPROFILE']
sys.path.insert(1,f'{USER}\\PycharmProjects\\my_utility')

from utility_code.my_utility import *
from utility_code.python_lib_essential import *

def svm():
    pass

def mlp():
    pass

def random_forest():
    pass

if __name__ == '__main__':
    from sklearn import svm

    X = [[0, 0], [1, 1]]
    y = [0, 1]
    clf = svm.SVC(gamma='scale')
    clf.fit(X,y)
    display2screen(clf.predict([[2.,2.]]))
