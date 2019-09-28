import sys,os
USER = os.environ['USERPROFILE']
sys.path.insert(1,f'{USER}\\PycharmProjects\\my_utility')

from utility_code.my_utility import *
from utility_code.python_lib_essential import *

def pearson_correlation_coefficient(x):
    '''

    :param x: networkx graph
    :return: similarity of all edges in the grpah
    '''
    pass