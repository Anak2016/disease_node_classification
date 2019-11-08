import sys,os
USER = os.environ['USERPROFILE']
sys.path.insert(1,f'{USER}\\PycharmProjects\\my_utility')

from utility_code.my_utility import *
from utility_code.python_lib_essential import *

from arg_parser import *
from my_utils import *
from all_models import baseline, embedding
import all_datasets
from __init__ import get_config
from anak import run_node2vec
from torch_geometric.nn import GCNConv

#TODO here>> run all of embbedding with different configuration and save reuslt to data
# embedding_config = {
#     #TODO here>>
#     'top_percent0.05_0': {
#         'name': 'svm',
#         'func': {
#             "model": run_node2vec,
#         },
#
#         'edges_selection': {
#             'common_nodes_feat': 'gene',
#             'edges_weight_option': 'jaccard',
#             'mask_edges': True,
#             'self_loop': False,
#             'edges_weight_limit': None,
#             'edges_weight_percent': None,
#             'top_percent_edges': 0.05,
#             'bottom_percent_edges': None,
#             'stochastic_edges': False,
#             'top_bottom_percent_edges': None,
#             'shared_nodes_random_edges_percent': None,
#             'all_nodes_random_edges_percent': None,
#         }
#     },
#     'top_percent0.05_1': {
#         'name': 'svm',
#         'func': {
#             "model": run_node2vec,
#         },
#
#         'edges_selection': {
#             'common_nodes_feat': 'gene',
#             'edges_weight_option': 'jaccard',
#             'mask_edges': True,
#             'self_loop': False,
#             'edges_weight_limit': None,
#             'edges_weight_percent': None,
#             'top_percent_edges': 0.05,
#             'bottom_percent_edges': None,
#             'stochastic_edges': False,
#             'top_bottom_percent_edges': None,
#             'shared_nodes_random_edges_percent': None,
#             'all_nodes_random_edges_percent': None,
#         }
#     },
#     'top_percent0.05_2': {
#         'name': 'svm',
#         'func': {
#             "model": run_node2vec,
#         },
#
#         'edges_selection': {
#             'common_nodes_feat': 'gene',
#             'edges_weight_option': 'jaccard',
#             'mask_edges': True,
#             'self_loop': False,
#             'edges_weight_limit': None,
#             'edges_weight_percent': None,
#             'top_percent_edges': 0.05,
#             'bottom_percent_edges': None,
#             'stochastic_edges': False,
#             'top_bottom_percent_edges': None,
#             'shared_nodes_random_edges_percent': None,
#             'all_nodes_random_edges_percent': None,
#         }
#     },
#     'top_percent0.05_3': {
#         'name': 'svm',
#         'func': {
#             "model": run_node2vec,
#         },
#
#         'edges_selection': {
#             'common_nodes_feat': 'gene',
#             'edges_weight_option': 'jaccard',
#             'mask_edges': True,
#             'self_loop': False,
#             'edges_weight_limit': None,
#             'edges_weight_percent': None,
#             'top_percent_edges': 0.05,
#             'bottom_percent_edges': None,
#             'stochastic_edges': False,
#             'top_bottom_percent_edges': None,
#             'shared_nodes_random_edges_percent': None,
#             'all_nodes_random_edges_percent': None,
#         }
#     },
#     'top_percent0.05_4': {
#         'name': 'svm',
#         'func': {
#             "model": run_node2vec,
#         },
#
#         'edges_selection': {
#             'common_nodes_feat': 'gene',
#             'edges_weight_option': 'jaccard',
#             'mask_edges': True,
#             'self_loop': False,
#             'edges_weight_limit': None,
#             'edges_weight_percent': None,
#             'top_percent_edges': 0.05,
#             'bottom_percent_edges': None,
#             'stochastic_edges': False,
#             'top_bottom_percent_edges': None,
#             'shared_nodes_random_edges_percent': None,
#             'all_nodes_random_edges_percent': None,
#         }
#     },
#     'top_percent0.05_5': {
#         'name': 'svm',
#         'func': {
#             "model": run_node2vec,
#         },
#
#         'edges_selection': {
#             'common_nodes_feat': 'gene',
#             'edges_weight_option': 'jaccard',
#             'mask_edges': True,
#             'self_loop': False,
#             'edges_weight_limit': None,
#             'edges_weight_percent': None,
#             'top_percent_edges': 0.05,
#             'bottom_percent_edges': None,
#             'stochastic_edges': False,
#             'top_bottom_percent_edges': None,
#             'shared_nodes_random_edges_percent': None,
#             'all_nodes_random_edges_percent': None,
#         }
#     },
#     'top_percent0.05_6': {
#         'name': 'svm',
#         'func': {
#             "model": run_node2vec,
#         },
#
#         'edges_selection': {
#             'common_nodes_feat': 'gene',
#             'edges_weight_option': 'jaccard',
#             'mask_edges': True,
#             'self_loop': False,
#             'edges_weight_limit': None,
#             'edges_weight_percent': None,
#             'top_percent_edges': 0.05,
#             'bottom_percent_edges': None,
#             'stochastic_edges': False,
#             'top_bottom_percent_edges': None,
#             'shared_nodes_random_edges_percent': None,
#             'all_nodes_random_edges_percent': None,
#         }
#     },
#     'top_percent0.05_7': {
#         'name': 'svm',
#         'func': {
#             "model": run_node2vec,
#         },
#
#         'edges_selection': {
#             'common_nodes_feat': 'gene',
#             'edges_weight_option': 'jaccard',
#             'mask_edges': True,
#             'self_loop': False,
#             'edges_weight_limit': None,
#             'edges_weight_percent': None,
#             'top_percent_edges': 0.05,
#             'bottom_percent_edges': None,
#             'stochastic_edges': False,
#             'top_bottom_percent_edges': None,
#             'shared_nodes_random_edges_percent': None,
#             'all_nodes_random_edges_percent': None,
#         }
#     },
#     'top_percent0.05_8': {
#         'name': 'svm',
#         'func': {
#             "model": run_node2vec,
#         },
#
#         'edges_selection': {
#             'common_nodes_feat': 'gene',
#             'edges_weight_option': 'jaccard',
#             'mask_edges': True,
#             'self_loop': False,
#             'edges_weight_limit': None,
#             'edges_weight_percent': None,
#             'top_percent_edges': 0.05,
#             'bottom_percent_edges': None,
#             'stochastic_edges': False,
#             'top_bottom_percent_edges': None,
#             'shared_nodes_random_edges_percent': None,
#             'all_nodes_random_edges_percent': None,
#         }
#     },
#     'top_percent0.05_9': {
#         'name': 'svm',
#         'func': {
#             "model": run_node2vec,
#         },
#
#         'edges_selection': {
#             'common_nodes_feat': 'gene',
#             'edges_weight_option': 'jaccard',
#             'mask_edges': True,
#             'self_loop': False,
#             'edges_weight_limit': None,
#             'edges_weight_percent': None,
#             'top_percent_edges': 0.05,
#             'bottom_percent_edges': None,
#             'stochastic_edges': False,
#             'top_bottom_percent_edges': None,
#             'shared_nodes_random_edges_percent': None,
#             'all_nodes_random_edges_percent': None,
#         }
#     },
#     'top_percent0.1_0': {
#         'name': 'svm',
#         'func': {
#             "model": run_node2vec,
#         },
#
#         'edges_selection': {
#             'common_nodes_feat': 'gene',
#             'edges_weight_option': 'jaccard',
#             'mask_edges': True,
#             'self_loop': False,
#             'edges_weight_limit': None,
#             'edges_weight_percent': None,
#             'top_percent_edges': 0.1,
#             'bottom_percent_edges': None,
#             'stochastic_edges': False,
#             'top_bottom_percent_edges': None,
#             'shared_nodes_random_edges_percent': None,
#             'all_nodes_random_edges_percent': None,
#         }
#     },
#     'top_percent0.1_1': {
#         'name': 'svm',
#         'func': {
#             "model": run_node2vec,
#         },
#
#         'edges_selection': {
#             'common_nodes_feat': 'gene',
#             'edges_weight_option': 'jaccard',
#             'mask_edges': True,
#             'self_loop': False,
#             'edges_weight_limit': None,
#             'edges_weight_percent': None,
#             'top_percent_edges': 0.1,
#             'bottom_percent_edges': None,
#             'stochastic_edges': False,
#             'top_bottom_percent_edges': None,
#             'shared_nodes_random_edges_percent': None,
#             'all_nodes_random_edges_percent': None,
#         }
#     },
#     'top_percent0.1_2': {
#         'name': 'svm',
#         'func': {
#             "model": run_node2vec,
#         },
#
#         'edges_selection': {
#             'common_nodes_feat': 'gene',
#             'edges_weight_option': 'jaccard',
#             'mask_edges': True,
#             'self_loop': False,
#             'edges_weight_limit': None,
#             'edges_weight_percent': None,
#             'top_percent_edges': 0.1,
#             'bottom_percent_edges': None,
#             'stochastic_edges': False,
#             'top_bottom_percent_edges': None,
#             'shared_nodes_random_edges_percent': None,
#             'all_nodes_random_edges_percent': None,
#         }
#     },
#     'top_percent0.1_3': {
#         'name': 'svm',
#         'func': {
#             "model": run_node2vec,
#         },
#
#         'edges_selection': {
#             'common_nodes_feat': 'gene',
#             'edges_weight_option': 'jaccard',
#             'mask_edges': True,
#             'self_loop': False,
#             'edges_weight_limit': None,
#             'edges_weight_percent': None,
#             'top_percent_edges': 0.1,
#             'bottom_percent_edges': None,
#             'stochastic_edges': False,
#             'top_bottom_percent_edges': None,
#             'shared_nodes_random_edges_percent': None,
#             'all_nodes_random_edges_percent': None,
#         }
#     },
#     'top_percent0.1_4': {
#         'name': 'svm',
#         'func': {
#             "model": run_node2vec,
#         },
#
#         'edges_selection': {
#             'common_nodes_feat': 'gene',
#             'edges_weight_option': 'jaccard',
#             'mask_edges': True,
#             'self_loop': False,
#             'edges_weight_limit': None,
#             'edges_weight_percent': None,
#             'top_percent_edges': 0.1,
#             'bottom_percent_edges': None,
#             'stochastic_edges': False,
#             'top_bottom_percent_edges': None,
#             'shared_nodes_random_edges_percent': None,
#             'all_nodes_random_edges_percent': None,
#         }
#     },
#     'top_percent0.1_5': {
#         'name': 'svm',
#         'func': {
#             "model": run_node2vec,
#         },
#
#         'edges_selection': {
#             'common_nodes_feat': 'gene',
#             'edges_weight_option': 'jaccard',
#             'mask_edges': True,
#             'self_loop': False,
#             'edges_weight_limit': None,
#             'edges_weight_percent': None,
#             'top_percent_edges': 0.1,
#             'bottom_percent_edges': None,
#             'stochastic_edges': False,
#             'top_bottom_percent_edges': None,
#             'shared_nodes_random_edges_percent': None,
#             'all_nodes_random_edges_percent': None,
#         }
#     },
#     'top_percent0.1_6': {
#         'name': 'svm',
#         'func': {
#             "model": run_node2vec,
#         },
#
#         'edges_selection': {
#             'common_nodes_feat': 'gene',
#             'edges_weight_option': 'jaccard',
#             'mask_edges': True,
#             'self_loop': False,
#             'edges_weight_limit': None,
#             'edges_weight_percent': None,
#             'top_percent_edges': 0.1,
#             'bottom_percent_edges': None,
#             'stochastic_edges': False,
#             'top_bottom_percent_edges': None,
#             'shared_nodes_random_edges_percent': None,
#             'all_nodes_random_edges_percent': None,
#         }
#     },
#     'top_percent0.1_7': {
#         'name': 'svm',
#         'func': {
#             "model": run_node2vec,
#         },
#
#         'edges_selection': {
#             'common_nodes_feat': 'gene',
#             'edges_weight_option': 'jaccard',
#             'mask_edges': True,
#             'self_loop': False,
#             'edges_weight_limit': None,
#             'edges_weight_percent': None,
#             'top_percent_edges': 0.1,
#             'bottom_percent_edges': None,
#             'stochastic_edges': False,
#             'top_bottom_percent_edges': None,
#             'shared_nodes_random_edges_percent': None,
#             'all_nodes_random_edges_percent': None,
#         }
#     },
#     'top_percent0.1_8': {
#         'name': 'svm',
#         'func': {
#             "model": run_node2vec,
#         },
#
#         'edges_selection': {
#             'common_nodes_feat': 'gene',
#             'edges_weight_option': 'jaccard',
#             'mask_edges': True,
#             'self_loop': False,
#             'edges_weight_limit': None,
#             'edges_weight_percent': None,
#             'top_percent_edges': 0.1,
#             'bottom_percent_edges': None,
#             'stochastic_edges': False,
#             'top_bottom_percent_edges': None,
#             'shared_nodes_random_edges_percent': None,
#             'all_nodes_random_edges_percent': None,
#         }
#     },
#     'top_percent0.1_9': {
#         'name': 'svm',
#         'func': {
#             "model": run_node2vec,
#         },
#
#         'edges_selection': {
#             'common_nodes_feat': 'gene',
#             'edges_weight_option': 'jaccard',
#             'mask_edges': True,
#             'self_loop': False,
#             'edges_weight_limit': None,
#             'edges_weight_percent': None,
#             'top_percent_edges': 0.1,
#             'bottom_percent_edges': None,
#             'stochastic_edges': False,
#             'top_bottom_percent_edges': None,
#             'shared_nodes_random_edges_percent': None,
#             'all_nodes_random_edges_percent': None,
#         }
#     },
#     'top_percent0.2_0': {
#         'name': 'svm',
#         'func': {
#             "model": run_node2vec,
#         },
#
#         'edges_selection': {
#             'common_nodes_feat': 'gene',
#             'edges_weight_option': 'jaccard',
#             'mask_edges': True,
#             'self_loop': False,
#             'edges_weight_limit': None,
#             'edges_weight_percent': None,
#             'top_percent_edges': 0.2,
#             'bottom_percent_edges': None,
#             'stochastic_edges': False,
#             'top_bottom_percent_edges': None,
#             'shared_nodes_random_edges_percent': None,
#             'all_nodes_random_edges_percent': None,
#         }
#     },
#     'top_percent0.2_1': {
#         'name': 'svm',
#         'func': {
#             "model": run_node2vec,
#         },
#
#         'edges_selection': {
#             'common_nodes_feat': 'gene',
#             'edges_weight_option': 'jaccard',
#             'mask_edges': True,
#             'self_loop': False,
#             'edges_weight_limit': None,
#             'edges_weight_percent': None,
#             'top_percent_edges': 0.2,
#             'bottom_percent_edges': None,
#             'stochastic_edges': False,
#             'top_bottom_percent_edges': None,
#             'shared_nodes_random_edges_percent': None,
#             'all_nodes_random_edges_percent': None,
#         }
#     },
#     'top_percent0.2_2': {
#         'name': 'svm',
#         'func': {
#             "model": run_node2vec,
#         },
#
#         'edges_selection': {
#             'common_nodes_feat': 'gene',
#             'edges_weight_option': 'jaccard',
#             'mask_edges': True,
#             'self_loop': False,
#             'edges_weight_limit': None,
#             'edges_weight_percent': None,
#             'top_percent_edges': 0.2,
#             'bottom_percent_edges': None,
#             'stochastic_edges': False,
#             'top_bottom_percent_edges': None,
#             'shared_nodes_random_edges_percent': None,
#             'all_nodes_random_edges_percent': None,
#         }
#     },
#     'top_percent0.2_3': {
#         'name': 'svm',
#         'func': {
#             "model": run_node2vec,
#         },
#
#         'edges_selection': {
#             'common_nodes_feat': 'gene',
#             'edges_weight_option': 'jaccard',
#             'mask_edges': True,
#             'self_loop': False,
#             'edges_weight_limit': None,
#             'edges_weight_percent': None,
#             'top_percent_edges': 0.2,
#             'bottom_percent_edges': None,
#             'stochastic_edges': False,
#             'top_bottom_percent_edges': None,
#             'shared_nodes_random_edges_percent': None,
#             'all_nodes_random_edges_percent': None,
#         }
#     },
#     'top_percent0.2_4': {
#         'name': 'svm',
#         'func': {
#             "model": run_node2vec,
#         },
#
#         'edges_selection': {
#             'common_nodes_feat': 'gene',
#             'edges_weight_option': 'jaccard',
#             'mask_edges': True,
#             'self_loop': False,
#             'edges_weight_limit': None,
#             'edges_weight_percent': None,
#             'top_percent_edges': 0.2,
#             'bottom_percent_edges': None,
#             'stochastic_edges': False,
#             'top_bottom_percent_edges': None,
#             'shared_nodes_random_edges_percent': None,
#             'all_nodes_random_edges_percent': None,
#         }
#     },
#     'top_percent0.2_5': {
#         'name': 'svm',
#         'func': {
#             "model": run_node2vec,
#         },
#
#         'edges_selection': {
#             'common_nodes_feat': 'gene',
#             'edges_weight_option': 'jaccard',
#             'mask_edges': True,
#             'self_loop': False,
#             'edges_weight_limit': None,
#             'edges_weight_percent': None,
#             'top_percent_edges': 0.2,
#             'bottom_percent_edges': None,
#             'stochastic_edges': False,
#             'top_bottom_percent_edges': None,
#             'shared_nodes_random_edges_percent': None,
#             'all_nodes_random_edges_percent': None,
#         }
#     },
#     'top_percent0.2_6': {
#         'name': 'svm',
#         'func': {
#             "model": run_node2vec,
#         },
#
#         'edges_selection': {
#             'common_nodes_feat': 'gene',
#             'edges_weight_option': 'jaccard',
#             'mask_edges': True,
#             'self_loop': False,
#             'edges_weight_limit': None,
#             'edges_weight_percent': None,
#             'top_percent_edges': 0.2,
#             'bottom_percent_edges': None,
#             'stochastic_edges': False,
#             'top_bottom_percent_edges': None,
#             'shared_nodes_random_edges_percent': None,
#             'all_nodes_random_edges_percent': None,
#         }
#     },
#     'top_percent0.2_7': {
#         'name': 'svm',
#         'func': {
#             "model": run_node2vec,
#         },
#
#         'edges_selection': {
#             'common_nodes_feat': 'gene',
#             'edges_weight_option': 'jaccard',
#             'mask_edges': True,
#             'self_loop': False,
#             'edges_weight_limit': None,
#             'edges_weight_percent': None,
#             'top_percent_edges': 0.2,
#             'bottom_percent_edges': None,
#             'stochastic_edges': False,
#             'top_bottom_percent_edges': None,
#             'shared_nodes_random_edges_percent': None,
#             'all_nodes_random_edges_percent': None,
#         }
#     },
#     'top_percent0.2_8': {
#         'name': 'svm',
#         'func': {
#             "model": run_node2vec,
#         },
#
#         'edges_selection': {
#             'common_nodes_feat': 'gene',
#             'edges_weight_option': 'jaccard',
#             'mask_edges': True,
#             'self_loop': False,
#             'edges_weight_limit': None,
#             'edges_weight_percent': None,
#             'top_percent_edges': 0.2,
#             'bottom_percent_edges': None,
#             'stochastic_edges': False,
#             'top_bottom_percent_edges': None,
#             'shared_nodes_random_edges_percent': None,
#             'all_nodes_random_edges_percent': None,
#         }
#     },
#     'top_percent0.2_9': {
#         'name': 'svm',
#         'func': {
#             "model": run_node2vec,
#         },
#
#         'edges_selection': {
#             'common_nodes_feat': 'gene',
#             'edges_weight_option': 'jaccard',
#             'mask_edges': True,
#             'self_loop': False,
#             'edges_weight_limit': None,
#             'edges_weight_percent': None,
#             'top_percent_edges': 0.2,
#             'bottom_percent_edges': None,
#             'stochastic_edges': False,
#             'top_bottom_percent_edges': None,
#             'shared_nodes_random_edges_percent': None,
#             'all_nodes_random_edges_percent': None,
#         }
#     },
#     'top_percent0.3_0': {
#         'name': 'svm',
#         'func': {
#             "model": run_node2vec,
#         },
#
#         'edges_selection': {
#             'common_nodes_feat': 'gene',
#             'edges_weight_option': 'jaccard',
#             'mask_edges': True,
#             'self_loop': False,
#             'edges_weight_limit': None,
#             'edges_weight_percent': None,
#             'top_percent_edges': 0.3,
#             'bottom_percent_edges': None,
#             'stochastic_edges': False,
#             'top_bottom_percent_edges': None,
#             'shared_nodes_random_edges_percent': None,
#             'all_nodes_random_edges_percent': None,
#         }
#     },
#     'top_percent0.3_1': {
#         'name': 'svm',
#         'func': {
#             "model": run_node2vec,
#         },
#
#         'edges_selection': {
#             'common_nodes_feat': 'gene',
#             'edges_weight_option': 'jaccard',
#             'mask_edges': True,
#             'self_loop': False,
#             'edges_weight_limit': None,
#             'edges_weight_percent': None,
#             'top_percent_edges': 0.3,
#             'bottom_percent_edges': None,
#             'stochastic_edges': False,
#             'top_bottom_percent_edges': None,
#             'shared_nodes_random_edges_percent': None,
#             'all_nodes_random_edges_percent': None,
#         }
#     },
#     'top_percent0.3_2': {
#         'name': 'svm',
#         'func': {
#             "model": run_node2vec,
#         },
#
#         'edges_selection': {
#             'common_nodes_feat': 'gene',
#             'edges_weight_option': 'jaccard',
#             'mask_edges': True,
#             'self_loop': False,
#             'edges_weight_limit': None,
#             'edges_weight_percent': None,
#             'top_percent_edges': 0.3,
#             'bottom_percent_edges': None,
#             'stochastic_edges': False,
#             'top_bottom_percent_edges': None,
#             'shared_nodes_random_edges_percent': None,
#             'all_nodes_random_edges_percent': None,
#         }
#     },
#     'top_percent0.3_3': {
#         'name': 'svm',
#         'func': {
#             "model": run_node2vec,
#         },
#
#         'edges_selection': {
#             'common_nodes_feat': 'gene',
#             'edges_weight_option': 'jaccard',
#             'mask_edges': True,
#             'self_loop': False,
#             'edges_weight_limit': None,
#             'edges_weight_percent': None,
#             'top_percent_edges': 0.3,
#             'bottom_percent_edges': None,
#             'stochastic_edges': False,
#             'top_bottom_percent_edges': None,
#             'shared_nodes_random_edges_percent': None,
#             'all_nodes_random_edges_percent': None,
#         }
#     },
#     'top_percent0.3_4': {
#         'name': 'svm',
#         'func': {
#             "model": run_node2vec,
#         },
#
#         'edges_selection': {
#             'common_nodes_feat': 'gene',
#             'edges_weight_option': 'jaccard',
#             'mask_edges': True,
#             'self_loop': False,
#             'edges_weight_limit': None,
#             'edges_weight_percent': None,
#             'top_percent_edges': 0.3,
#             'bottom_percent_edges': None,
#             'stochastic_edges': False,
#             'top_bottom_percent_edges': None,
#             'shared_nodes_random_edges_percent': None,
#             'all_nodes_random_edges_percent': None,
#         }
#     },
#     'top_percent0.3_5': {
#         'name': 'svm',
#         'func': {
#             "model": run_node2vec,
#         },
#
#         'edges_selection': {
#             'common_nodes_feat': 'gene',
#             'edges_weight_option': 'jaccard',
#             'mask_edges': True,
#             'self_loop': False,
#             'edges_weight_limit': None,
#             'edges_weight_percent': None,
#             'top_percent_edges': 0.3,
#             'bottom_percent_edges': None,
#             'stochastic_edges': False,
#             'top_bottom_percent_edges': None,
#             'shared_nodes_random_edges_percent': None,
#             'all_nodes_random_edges_percent': None,
#         }
#     },
#     'top_percent0.3_6': {
#         'name': 'svm',
#         'func': {
#             "model": run_node2vec,
#         },
#
#         'edges_selection': {
#             'common_nodes_feat': 'gene',
#             'edges_weight_option': 'jaccard',
#             'mask_edges': True,
#             'self_loop': False,
#             'edges_weight_limit': None,
#             'edges_weight_percent': None,
#             'top_percent_edges': 0.3,
#             'bottom_percent_edges': None,
#             'stochastic_edges': False,
#             'top_bottom_percent_edges': None,
#             'shared_nodes_random_edges_percent': None,
#             'all_nodes_random_edges_percent': None,
#         }
#     },
#     'top_percent0.3_7': {
#         'name': 'svm',
#         'func': {
#             "model": run_node2vec,
#         },
#
#         'edges_selection': {
#             'common_nodes_feat': 'gene',
#             'edges_weight_option': 'jaccard',
#             'mask_edges': True,
#             'self_loop': False,
#             'edges_weight_limit': None,
#             'edges_weight_percent': None,
#             'top_percent_edges': 0.3,
#             'bottom_percent_edges': None,
#             'stochastic_edges': False,
#             'top_bottom_percent_edges': None,
#             'shared_nodes_random_edges_percent': None,
#             'all_nodes_random_edges_percent': None,
#         }
#     },
#     'top_percent0.3_8': {
#         'name': 'svm',
#         'func': {
#             "model": run_node2vec,
#         },
#
#         'edges_selection': {
#             'common_nodes_feat': 'gene',
#             'edges_weight_option': 'jaccard',
#             'mask_edges': True,
#             'self_loop': False,
#             'edges_weight_limit': None,
#             'edges_weight_percent': None,
#             'top_percent_edges': 0.3,
#             'bottom_percent_edges': None,
#             'stochastic_edges': False,
#             'top_bottom_percent_edges': None,
#             'shared_nodes_random_edges_percent': None,
#             'all_nodes_random_edges_percent': None,
#         }
#     },
#     'top_percent0.3_9': {
#         'name': 'svm',
#         'func': {
#             "model": run_node2vec,
#         },
#
#         'edges_selection': {
#             'common_nodes_feat': 'gene',
#             'edges_weight_option': 'jaccard',
#             'mask_edges': True,
#             'self_loop': False,
#             'edges_weight_limit': None,
#             'edges_weight_percent': None,
#             'top_percent_edges': 0.3,
#             'bottom_percent_edges': None,
#             'stochastic_edges': False,
#             'top_bottom_percent_edges': None,
#             'shared_nodes_random_edges_percent': None,
#             'all_nodes_random_edges_percent': None,
#         }
#     },
#     'top_percent0.4_0': {
#         'name': 'svm',
#         'func': {
#             "model": run_node2vec,
#         },
#
#         'edges_selection': {
#             'common_nodes_feat': 'gene',
#             'edges_weight_option': 'jaccard',
#             'mask_edges': True,
#             'self_loop': False,
#             'edges_weight_limit': None,
#             'edges_weight_percent': None,
#             'top_percent_edges': 0.4,
#             'bottom_percent_edges': None,
#             'stochastic_edges': False,
#             'top_bottom_percent_edges': None,
#             'shared_nodes_random_edges_percent': None,
#             'all_nodes_random_edges_percent': None,
#         }
#     },
#     'top_percent0.4_1': {
#         'name': 'svm',
#         'func': {
#             "model": run_node2vec,
#         },
#
#         'edges_selection': {
#             'common_nodes_feat': 'gene',
#             'edges_weight_option': 'jaccard',
#             'mask_edges': True,
#             'self_loop': False,
#             'edges_weight_limit': None,
#             'edges_weight_percent': None,
#             'top_percent_edges': 0.4,
#             'bottom_percent_edges': None,
#             'stochastic_edges': False,
#             'top_bottom_percent_edges': None,
#             'shared_nodes_random_edges_percent': None,
#             'all_nodes_random_edges_percent': None,
#         }
#     },
#     'top_percent0.4_2': {
#         'name': 'svm',
#         'func': {
#             "model": run_node2vec,
#         },
#
#         'edges_selection': {
#             'common_nodes_feat': 'gene',
#             'edges_weight_option': 'jaccard',
#             'mask_edges': True,
#             'self_loop': False,
#             'edges_weight_limit': None,
#             'edges_weight_percent': None,
#             'top_percent_edges': 0.4,
#             'bottom_percent_edges': None,
#             'stochastic_edges': False,
#             'top_bottom_percent_edges': None,
#             'shared_nodes_random_edges_percent': None,
#             'all_nodes_random_edges_percent': None,
#         }
#     },
#     'top_percent0.4_3': {
#         'name': 'svm',
#         'func': {
#             "model": run_node2vec,
#         },
#
#         'edges_selection': {
#             'common_nodes_feat': 'gene',
#             'edges_weight_option': 'jaccard',
#             'mask_edges': True,
#             'self_loop': False,
#             'edges_weight_limit': None,
#             'edges_weight_percent': None,
#             'top_percent_edges': 0.4,
#             'bottom_percent_edges': None,
#             'stochastic_edges': False,
#             'top_bottom_percent_edges': None,
#             'shared_nodes_random_edges_percent': None,
#             'all_nodes_random_edges_percent': None,
#         }
#     },
#     'top_percent0.4_4': {
#         'name': 'svm',
#         'func': {
#             "model": run_node2vec,
#         },
#
#         'edges_selection': {
#             'common_nodes_feat': 'gene',
#             'edges_weight_option': 'jaccard',
#             'mask_edges': True,
#             'self_loop': False,
#             'edges_weight_limit': None,
#             'edges_weight_percent': None,
#             'top_percent_edges': 0.4,
#             'bottom_percent_edges': None,
#             'stochastic_edges': False,
#             'top_bottom_percent_edges': None,
#             'shared_nodes_random_edges_percent': None,
#             'all_nodes_random_edges_percent': None,
#         }
#     },
#     'top_percent0.4_5': {
#         'name': 'svm',
#         'func': {
#             "model": run_node2vec,
#         },
#
#         'edges_selection': {
#             'common_nodes_feat': 'gene',
#             'edges_weight_option': 'jaccard',
#             'mask_edges': True,
#             'self_loop': False,
#             'edges_weight_limit': None,
#             'edges_weight_percent': None,
#             'top_percent_edges': 0.4,
#             'bottom_percent_edges': None,
#             'stochastic_edges': False,
#             'top_bottom_percent_edges': None,
#             'shared_nodes_random_edges_percent': None,
#             'all_nodes_random_edges_percent': None,
#         }
#     },
#     'top_percent0.4_6': {
#         'name': 'svm',
#         'func': {
#             "model": run_node2vec,
#         },
#
#         'edges_selection': {
#             'common_nodes_feat': 'gene',
#             'edges_weight_option': 'jaccard',
#             'mask_edges': True,
#             'self_loop': False,
#             'edges_weight_limit': None,
#             'edges_weight_percent': None,
#             'top_percent_edges': 0.4,
#             'bottom_percent_edges': None,
#             'stochastic_edges': False,
#             'top_bottom_percent_edges': None,
#             'shared_nodes_random_edges_percent': None,
#             'all_nodes_random_edges_percent': None,
#         }
#     },
#     'top_percent0.4_7': {
#         'name': 'svm',
#         'func': {
#             "model": run_node2vec,
#         },
#
#         'edges_selection': {
#             'common_nodes_feat': 'gene',
#             'edges_weight_option': 'jaccard',
#             'mask_edges': True,
#             'self_loop': False,
#             'edges_weight_limit': None,
#             'edges_weight_percent': None,
#             'top_percent_edges': 0.4,
#             'bottom_percent_edges': None,
#             'stochastic_edges': False,
#             'top_bottom_percent_edges': None,
#             'shared_nodes_random_edges_percent': None,
#             'all_nodes_random_edges_percent': None,
#         }
#     },
#     'top_percent0.4_8': {
#         'name': 'svm',
#         'func': {
#             "model": run_node2vec,
#         },
#
#         'edges_selection': {
#             'common_nodes_feat': 'gene',
#             'edges_weight_option': 'jaccard',
#             'mask_edges': True,
#             'self_loop': False,
#             'edges_weight_limit': None,
#             'edges_weight_percent': None,
#             'top_percent_edges': 0.4,
#             'bottom_percent_edges': None,
#             'stochastic_edges': False,
#             'top_bottom_percent_edges': None,
#             'shared_nodes_random_edges_percent': None,
#             'all_nodes_random_edges_percent': None,
#         }
#     },
#     'top_percent0.4_9': {
#         'name': 'svm',
#         'func': {
#             "model": run_node2vec,
#         },
#
#         'edges_selection': {
#             'common_nodes_feat': 'gene',
#             'edges_weight_option': 'jaccard',
#             'mask_edges': True,
#             'self_loop': False,
#             'edges_weight_limit': None,
#             'edges_weight_percent': None,
#             'top_percent_edges': 0.4,
#             'bottom_percent_edges': None,
#             'stochastic_edges': False,
#             'top_bottom_percent_edges': None,
#             'shared_nodes_random_edges_percent': None,
#             'all_nodes_random_edges_percent': None,
#         }
#     },
#     'top_percent0.5_0': {
#         'name': 'svm',
#         'func': {
#             "model": run_node2vec,
#         },
#
#         'edges_selection': {
#             'common_nodes_feat': 'gene',
#             'edges_weight_option': 'jaccard',
#             'mask_edges': True,
#             'self_loop': False,
#             'edges_weight_limit': None,
#             'edges_weight_percent': None,
#             'top_percent_edges': 0.5,
#             'bottom_percent_edges': None,
#             'stochastic_edges': False,
#             'top_bottom_percent_edges': None,
#             'shared_nodes_random_edges_percent': None,
#             'all_nodes_random_edges_percent': None,
#         }
#     },
#     'top_percent0.5_1': {
#         'name': 'svm',
#         'func': {
#             "model": run_node2vec,
#         },
#
#         'edges_selection': {
#             'common_nodes_feat': 'gene',
#             'edges_weight_option': 'jaccard',
#             'mask_edges': True,
#             'self_loop': False,
#             'edges_weight_limit': None,
#             'edges_weight_percent': None,
#             'top_percent_edges': 0.5,
#             'bottom_percent_edges': None,
#             'stochastic_edges': False,
#             'top_bottom_percent_edges': None,
#             'shared_nodes_random_edges_percent': None,
#             'all_nodes_random_edges_percent': None,
#         }
#     },
#     'top_percent0.5_2': {
#         'name': 'svm',
#         'func': {
#             "model": run_node2vec,
#         },
#
#         'edges_selection': {
#             'common_nodes_feat': 'gene',
#             'edges_weight_option': 'jaccard',
#             'mask_edges': True,
#             'self_loop': False,
#             'edges_weight_limit': None,
#             'edges_weight_percent': None,
#             'top_percent_edges': 0.5,
#             'bottom_percent_edges': None,
#             'stochastic_edges': False,
#             'top_bottom_percent_edges': None,
#             'shared_nodes_random_edges_percent': None,
#             'all_nodes_random_edges_percent': None,
#         }
#     },
#     'top_percent0.5_3': {
#         'name': 'svm',
#         'func': {
#             "model": run_node2vec,
#         },
#
#         'edges_selection': {
#             'common_nodes_feat': 'gene',
#             'edges_weight_option': 'jaccard',
#             'mask_edges': True,
#             'self_loop': False,
#             'edges_weight_limit': None,
#             'edges_weight_percent': None,
#             'top_percent_edges': 0.5,
#             'bottom_percent_edges': None,
#             'stochastic_edges': False,
#             'top_bottom_percent_edges': None,
#             'shared_nodes_random_edges_percent': None,
#             'all_nodes_random_edges_percent': None,
#         }
#     },
#     'top_percent0.5_4': {
#         'name': 'svm',
#         'func': {
#             "model": run_node2vec,
#         },
#
#         'edges_selection': {
#             'common_nodes_feat': 'gene',
#             'edges_weight_option': 'jaccard',
#             'mask_edges': True,
#             'self_loop': False,
#             'edges_weight_limit': None,
#             'edges_weight_percent': None,
#             'top_percent_edges': 0.5,
#             'bottom_percent_edges': None,
#             'stochastic_edges': False,
#             'top_bottom_percent_edges': None,
#             'shared_nodes_random_edges_percent': None,
#             'all_nodes_random_edges_percent': None,
#         }
#     },
#     'top_percent0.5_5': {
#         'name': 'svm',
#         'func': {
#             "model": run_node2vec,
#         },
#
#         'edges_selection': {
#             'common_nodes_feat': 'gene',
#             'edges_weight_option': 'jaccard',
#             'mask_edges': True,
#             'self_loop': False,
#             'edges_weight_limit': None,
#             'edges_weight_percent': None,
#             'top_percent_edges': 0.5,
#             'bottom_percent_edges': None,
#             'stochastic_edges': False,
#             'top_bottom_percent_edges': None,
#             'shared_nodes_random_edges_percent': None,
#             'all_nodes_random_edges_percent': None,
#         }
#     },
#     'top_percent0.5_6': {
#         'name': 'svm',
#         'func': {
#             "model": run_node2vec,
#         },
#
#         'edges_selection': {
#             'common_nodes_feat': 'gene',
#             'edges_weight_option': 'jaccard',
#             'mask_edges': True,
#             'self_loop': False,
#             'edges_weight_limit': None,
#             'edges_weight_percent': None,
#             'top_percent_edges': 0.5,
#             'bottom_percent_edges': None,
#             'stochastic_edges': False,
#             'top_bottom_percent_edges': None,
#             'shared_nodes_random_edges_percent': None,
#             'all_nodes_random_edges_percent': None,
#         }
#     },
#     'top_percent0.5_7': {
#         'name': 'svm',
#         'func': {
#             "model": run_node2vec,
#         },
#
#         'edges_selection': {
#             'common_nodes_feat': 'gene',
#             'edges_weight_option': 'jaccard',
#             'mask_edges': True,
#             'self_loop': False,
#             'edges_weight_limit': None,
#             'edges_weight_percent': None,
#             'top_percent_edges': 0.5,
#             'bottom_percent_edges': None,
#             'stochastic_edges': False,
#             'top_bottom_percent_edges': None,
#             'shared_nodes_random_edges_percent': None,
#             'all_nodes_random_edges_percent': None,
#         }
#     },
#     'top_percent0.5_8': {
#         'name': 'svm',
#         'func': {
#             "model": run_node2vec,
#         },
#
#         'edges_selection': {
#             'common_nodes_feat': 'gene',
#             'edges_weight_option': 'jaccard',
#             'mask_edges': True,
#             'self_loop': False,
#             'edges_weight_limit': None,
#             'edges_weight_percent': None,
#             'top_percent_edges': 0.5,
#             'bottom_percent_edges': None,
#             'stochastic_edges': False,
#             'top_bottom_percent_edges': None,
#             'shared_nodes_random_edges_percent': None,
#             'all_nodes_random_edges_percent': None,
#         }
#     },
#     'top_percent0.5_9': {
#         'name': 'svm',
#         'func': {
#             "model": run_node2vec,
#         },
#
#         'edges_selection': {
#             'common_nodes_feat': 'gene',
#             'edges_weight_option': 'jaccard',
#             'mask_edges': True,
#             'self_loop': False,
#             'edges_weight_limit': None,
#             'edges_weight_percent': None,
#             'top_percent_edges': 0.5,
#             'bottom_percent_edges': None,
#             'stochastic_edges': False,
#             'top_bottom_percent_edges': None,
#             'shared_nodes_random_edges_percent': None,
#             'all_nodes_random_edges_percent': None,
#         }
#     },
# }



def run_embedding(copd, config=None, emb_type=None):

    run_and_save_embedding(copd, emb_type=emb_type, config =config)

def run_and_save_embedding(copd, emb_type=None, config=None):
    for name, model in config.items():

        # random.seed(args.seed)
        # np.random.seed(args.seed)
        # torch.manual_seed(args.seed)
        # =====================
        # ==code below is bad, I can fix it easily by create "preprocessing" class and group all preprocess method into it
        # =====================
        # # reassign args of jaccard_coeff function

        copd_geometric_dataset = None
        if model.get('edges_selection', None):
            args.common_nodes_feat = model['edges_selection']['common_nodes_feat']
            # args.stochastic_edges = model['edges_selection']['stochastic_edges']
            args.index = int(name.split('_')[-1]) if re.findall("^[0-9]$",name.split('_')[-1]) else 0
            if args.common_nodes_feat != "no":

                args.stochastic_edges     = model['edges_selection']['stochastic_edges']
                args.mask_edges           = model['edges_selection']['mask_edges']
                args.edges_weight_limit   = model['edges_selection']['edges_weight_limit']
                args.self_loop            = model['edges_selection']['self_loop']
                args.edges_weight_percent = model['edges_selection']['edges_weight_percent']
                args.top_percent_edges    = model['edges_selection']['top_percent_edges']
                args.bottom_percent_edges = model['edges_selection']['bottom_percent_edges']
                args.top_bottom_percent_edges = model['edges_selection']['top_bottom_percent_edges']
                args.shared_nodes_random_edges_percent    = model['edges_selection']['shared_nodes_random_edges_percent']
                args.all_nodes_random_edges_percent    = model['edges_selection']['all_nodes_random_edges_percent']

                # func_kwargs = { 'used_nodes':args.common_nodes_feat,"edges_weight_option":model['edges_selection']['edges_weight_option']}
                _, copd_geometric_dataset, weight_adj = get_config(model['name'], copd, used_nodes=args.common_nodes_feat,
                                                                 edges_weight_option=model['edges_selection']['edges_weight_option'])
            else:
                raise ValueError("error in run_ensemble: common_nodes_feat is None")
        else:
            raise ValueError('func_kwargs is None')

        args.cv = None
        #=====================
        #==get trained model
        #=====================
        emb_func = model['func']['model']

        # func_args= model['func']['args']

        # def run_node2vec(copd, copd_geometric, time_stamp=args.time_stamp):
        # if args.run_embedding == 'node2vec':
        if model['name'] == 'node2vec':
            emb_func(copd, copd_geometric_dataset, weight_adj) # pred must be real prediction
        # elif args.run_embedding == 'gnn':
        elif model['name'] == 'gnn':
            # emb_func(s)
            # gcn_runtime = timer(embedding.GNN(data=copd_geometric_dataset, config=config).run)
            from all_models import embedding
            args.verbose = True
            # args.verbose = False
            # args.hidden = 256
            args.hidden = 64
            # args.hidden = 32
            # args.hidden = 16
            # args.hidden = 8
            # args.epochs = 100
            args.report_performance = True
            print(copd_geometric_dataset.edge_index.shape)

            modules = {
                # "conv1": GCNConv(64, args.hidden, cached=True),
                "conv1": GCNConv(copd_geometric_dataset.num_features, args.hidden, cached=True),
                # "conv2": GCNConv(args.hidden, copd_geometric_dataset.num_classes, cached=True),
                "conv2": GCNConv(args.hidden, 16, cached=True),

                # "Linearin_32": nn.Linear(64, 32),
                "Linear32_16": nn.Linear(32, 16),
                "Linear16_out": nn.Linear(16, copd_geometric_dataset.num_classes),
                "Linear32_out": nn.Linear(32, copd_geometric_dataset.num_classes),
            }

            embedding.GNN(data=copd_geometric_dataset).run(modules)
        else:
            raise ValueError('specifed embedding  is not supported or incorrectly typed')

def run_main():
    percent = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    stoch = [True, False]
    # conditions = ['top_percent_edges', 'bottom_percent_edges', 'top_bottom_percent_edges',
    #               'all_nodes_random_edges_percent', 'shared_nodes_random_edges_percent']
    conditions = ['top_bottom_percent_edges', 'all_nodes_random_edges_percent', 'shared_nodes_random_edges_percent']
    # embedding = 'gnn'
    embedding = 'node2vec'

    # percent = [0.1]
    # stoch = [True, False]
    # stoch = [False]
    # conditions = ["top_percent_edges"]

    embedding_config = {}
    for i in conditions:
        for k in stoch:
            for j in percent:
                if i in ['shared_nodes_random_edges_percent', "shared_nodes_random_edges_percent"]:
                    if k:
                        for index in range(10):  # all of them have the same result.
                            embedding_config[f'{i}{j}_stoch={k}_{index}'] = {}
                            embedding_config[f'{i}{j}_stoch={k}_{index}']['name'] = embedding
                            embedding_config[f'{i}{j}_stoch={k}_{index}']['func'] = {}
                            embedding_config[f'{i}{j}_stoch={k}_{index}']['func']['model'] = run_node2vec
                            embedding_config[f'{i}{j}_stoch={k}_{index}']['edges_selection'] = {}
                            embedding_config[f'{i}{j}_stoch={k}_{index}']['edges_selection']['common_nodes_feat'] = 'gene'
                            embedding_config[f'{i}{j}_stoch={k}_{index}']['edges_selection']['edges_weight_option'] = 'jaccard'
                            embedding_config[f'{i}{j}_stoch={k}_{index}']['edges_selection']['mask_edges'] = True
                            embedding_config[f'{i}{j}_stoch={k}_{index}']['edges_selection']['self_loop'] = False
                            embedding_config[f'{i}{j}_stoch={k}_{index}']['edges_selection']['edges_weight_limit'] = None
                            embedding_config[f'{i}{j}_stoch={k}_{index}']['edges_selection']['edges_weight_percent'] = None
                            embedding_config[f'{i}{j}_stoch={k}_{index}']['edges_selection']['stochastic_edges'] = k
                            # for n in conditions:
                            for n in ['top_percent_edges', 'bottom_percent_edges', 'top_bottom_percent_edges','shared_nodes_random_edges_percent', 'all_nodes_random_edges_percent']:
                                if n == i:
                                    embedding_config[f'{i}{j}_stoch={k}_{index}']['edges_selection'][i] = j
                                else:
                                    embedding_config[f'{i}{j}_stoch={k}_{index}']['edges_selection'][n] = None
                else:
                    if k:
                        repeat = range(10)
                    else:
                        repeat = range(1)

                    for index in repeat:  # all of them have the same result.
                        embedding_config[f'{i}{j}_stoch={k}_{index}'] = {}
                        embedding_config[f'{i}{j}_stoch={k}_{index}']['name'] = embedding
                        embedding_config[f'{i}{j}_stoch={k}_{index}']['func'] = {}
                        embedding_config[f'{i}{j}_stoch={k}_{index}']['func']['model'] = run_node2vec
                        embedding_config[f'{i}{j}_stoch={k}_{index}']['edges_selection'] = {}
                        embedding_config[f'{i}{j}_stoch={k}_{index}']['edges_selection']['common_nodes_feat'] = 'gene'
                        embedding_config[f'{i}{j}_stoch={k}_{index}']['edges_selection']['edges_weight_option'] = 'jaccard'
                        embedding_config[f'{i}{j}_stoch={k}_{index}']['edges_selection']['mask_edges'] = True
                        embedding_config[f'{i}{j}_stoch={k}_{index}']['edges_selection']['self_loop'] = False
                        embedding_config[f'{i}{j}_stoch={k}_{index}']['edges_selection']['edges_weight_limit'] = None
                        embedding_config[f'{i}{j}_stoch={k}_{index}']['edges_selection']['edges_weight_percent'] = None
                        embedding_config[f'{i}{j}_stoch={k}_{index}']['edges_selection']['stochastic_edges'] = k
                        # for n in conditions:
                        for n in ['top_percent_edges', 'bottom_percent_edges', 'top_bottom_percent_edges','shared_nodes_random_edges_percent', 'all_nodes_random_edges_percent']:
                            if n == i:
                                embedding_config[f'{i}{j}_stoch={k}_{index}']['edges_selection'][i] = j
                            else:
                                embedding_config[f'{i}{j}_stoch={k}_{index}']['edges_selection'][n] = None

    # display2screen(embedding_config)

    copd = all_datasets.Copd(path=args.copd_path, data=args.copd_data, time_stamp=args.time_stamp,
                             undirected=not args.directed)
    if args.run_embedding:
        run_embedding(copd, emb_type=args.run_embedding, config=embedding_config)


if __name__ == '__main__':
    run_main()
