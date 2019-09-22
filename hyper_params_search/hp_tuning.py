import sys,os
USER = os.environ['USERPROFILE']
sys.path.insert(1,f'{USER}\\PycharmProjects\\my_utility')

from utility_code.my_utility import *
from utility_code.python_lib_essential import *

#=====================
#==option of search si taken as an argument eg. grid and random.
#=====================

def hp_svm(search_type='grid'):
    pass

def hp_mlp(search_type='grid'):
    pass

def hp_random_forest(search_type='grid'):
    pass



