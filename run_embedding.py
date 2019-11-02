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

#TODO here>> run all of embbedding with different configuration and save reuslt to data
embedding_config = {

    'top_percent_0.5_0': {
        'name': 'svm',
        'func': {
            "model": run_node2vec,
        },

        'edges_selection': {
            'common_nodes_feat': 'gene',
            'edges_weight_option': 'jaccard',
            'mask_edges': True,
            'self_loop': False,
            'edges_weight_limit': None,
            'edges_weight_percent': None,
            'top_percent_edges': 0.5,
            'stochastic_edges': False,
            'bottom_percent_edges': None,
            'top_bottom_percent_edges': None,
            'shared_nodes_random_edges_percent': None,
            'all_nodes_random_edges_percent': None,
        }
    },

    'top_percent_0.5_1': {
        'name': 'svm',
        'func': {
            "model": run_node2vec,
        },

        'edges_selection': {
            'common_nodes_feat': 'gene',
            'edges_weight_option': 'jaccard',
            'mask_edges': True,
            'self_loop': False,
            'edges_weight_limit': None,
            'edges_weight_percent': None,
            'top_percent_edges': 0.5,
            'stochastic_edges': False,
            'bottom_percent_edges': None,
            'top_bottom_percent_edges': None,
            'shared_nodes_random_edges_percent': None,
            'all_nodes_random_edges_percent': None,
        }
    },

    'top_percent_0.5_2': {
        'name': 'svm',
        'func': {
            "model": run_node2vec,
        },

        'edges_selection': {
            'common_nodes_feat': 'gene',
            'edges_weight_option': 'jaccard',
            'mask_edges': True,
            'self_loop': False,
            'edges_weight_limit': None,
            'edges_weight_percent': None,
            'top_percent_edges': 0.5,
            'stochastic_edges': False,
            'bottom_percent_edges': None,
            'top_bottom_percent_edges': None,
            'shared_nodes_random_edges_percent': None,
            'all_nodes_random_edges_percent': None,
        }
    },

    'top_percent_0.5_3': {
        'name': 'svm',
        'func': {
            "model": run_node2vec,
        },

        'edges_selection': {
            'common_nodes_feat': 'gene',
            'edges_weight_option': 'jaccard',
            'mask_edges': True,
            'self_loop': False,
            'edges_weight_limit': None,
            'edges_weight_percent': None,
            'top_percent_edges': 0.5,
            'stochastic_edges': False,
            'bottom_percent_edges': None,
            'top_bottom_percent_edges': None,
            'shared_nodes_random_edges_percent': None,
            'all_nodes_random_edges_percent': None,
        }
    },

    'top_percent_0.5_4': {
        'name': 'svm',
        'func': {
            "model": run_node2vec,
        },

        'edges_selection': {
            'common_nodes_feat': 'gene',
            'edges_weight_option': 'jaccard',
            'mask_edges': True,
            'self_loop': False,
            'edges_weight_limit': None,
            'edges_weight_percent': None,
            'top_percent_edges': 0.5,
            'stochastic_edges': False,
            'bottom_percent_edges': None,
            'top_bottom_percent_edges': None,
            'shared_nodes_random_edges_percent': None,
            'all_nodes_random_edges_percent': None,
        }
    },

    'top_percent_0.5_5': {
        'name': 'svm',
        'func': {
            "model": run_node2vec,
        },

        'edges_selection': {
            'common_nodes_feat': 'gene',
            'edges_weight_option': 'jaccard',
            'mask_edges': True,
            'self_loop': False,
            'edges_weight_limit': None,
            'edges_weight_percent': None,
            'top_percent_edges': 0.5,
            'stochastic_edges': False,
            'bottom_percent_edges': None,
            'top_bottom_percent_edges': None,
            'shared_nodes_random_edges_percent': None,
            'all_nodes_random_edges_percent': None,
        }
    },

    'top_percent_0.5_6': {
        'name': 'svm',
        'func': {
            "model": run_node2vec,
        },

        'edges_selection': {
            'common_nodes_feat': 'gene',
            'edges_weight_option': 'jaccard',
            'mask_edges': True,
            'self_loop': False,
            'edges_weight_limit': None,
            'edges_weight_percent': None,
            'top_percent_edges': 0.5,
            'stochastic_edges': False,
            'bottom_percent_edges': None,
            'top_bottom_percent_edges': None,
            'shared_nodes_random_edges_percent': None,
            'all_nodes_random_edges_percent': None,
        }
    },

    'top_percent_0.5_7': {
        'name': 'svm',
        'func': {
            "model": run_node2vec,
        },

        'edges_selection': {
            'common_nodes_feat': 'gene',
            'edges_weight_option': 'jaccard',
            'mask_edges': True,
            'self_loop': False,
            'edges_weight_limit': None,
            'edges_weight_percent': None,
            'top_percent_edges': 0.5,
            'stochastic_edges': False,
            'bottom_percent_edges': None,
            'top_bottom_percent_edges': None,
            'shared_nodes_random_edges_percent': None,
            'all_nodes_random_edges_percent': None,
        }
    },

    'top_percent_0.5_8': {
        'name': 'svm',
        'func': {
            "model": run_node2vec,
        },

        'edges_selection': {
            'common_nodes_feat': 'gene',
            'edges_weight_option': 'jaccard',
            'mask_edges': True,
            'self_loop': False,
            'edges_weight_limit': None,
            'edges_weight_percent': None,
            'top_percent_edges': 0.5,
            'stochastic_edges': False,
            'bottom_percent_edges': None,
            'top_bottom_percent_edges': None,
            'shared_nodes_random_edges_percent': None,
            'all_nodes_random_edges_percent': None,
        }
    },

    'top_percent_0.5_9': {
        'name': 'svm',
        'func': {
            "model": run_node2vec,
        },

        'edges_selection': {
            'common_nodes_feat': 'gene',
            'edges_weight_option': 'jaccard',
            'mask_edges': True,
            'self_loop': False,
            'edges_weight_limit': None,
            'edges_weight_percent': None,
            'top_percent_edges': 0.5,
            'stochastic_edges': False,
            'bottom_percent_edges': None,
            'top_bottom_percent_edges': None,
            'shared_nodes_random_edges_percent': None,
            'all_nodes_random_edges_percent': None,
        }
    },
    'bottom_percent_stoch0.5_0': {
        'name': 'svm',
        'func': {
            "model": run_node2vec,
        },

        'edges_selection': {
            'common_nodes_feat': 'gene',
            'edges_weight_option': 'jaccard',
            'mask_edges': True,
            'self_loop': False,
            'edges_weight_limit': None,
            'edges_weight_percent': None,
            'top_percent_edges': None,
            'bottom_percent_edges': 0.5,
            'stochastic_edges': True,
            'top_bottom_percent_edges': None,
            'shared_nodes_random_edges_percent': None,
            'all_nodes_random_edges_percent': None,
        }
    },
    'bottom_percent_stoch0.5_1': {
        'name': 'svm',
        'func': {
            "model": run_node2vec,
        },

        'edges_selection': {
            'common_nodes_feat': 'gene',
            'edges_weight_option': 'jaccard',
            'mask_edges': True,
            'self_loop': False,
            'edges_weight_limit': None,
            'edges_weight_percent': None,
            'top_percent_edges': None,
            'bottom_percent_edges': 0.5,
            'stochastic_edges': True,
            'top_bottom_percent_edges': None,
            'shared_nodes_random_edges_percent': None,
            'all_nodes_random_edges_percent': None,
        }
    },
    'bottom_percent_stoch0.5_2': {
        'name': 'svm',
        'func': {
            "model": run_node2vec,
        },

        'edges_selection': {
            'common_nodes_feat': 'gene',
            'edges_weight_option': 'jaccard',
            'mask_edges': True,
            'self_loop': False,
            'edges_weight_limit': None,
            'edges_weight_percent': None,
            'top_percent_edges': None,
            'bottom_percent_edges': 0.5,
            'stochastic_edges': True,
            'top_bottom_percent_edges': None,
            'shared_nodes_random_edges_percent': None,
            'all_nodes_random_edges_percent': None,
        }
    },
    'bottom_percent_stoch0.5_3': {
        'name': 'svm',
        'func': {
            "model": run_node2vec,
        },

        'edges_selection': {
            'common_nodes_feat': 'gene',
            'edges_weight_option': 'jaccard',
            'mask_edges': True,
            'self_loop': False,
            'edges_weight_limit': None,
            'edges_weight_percent': None,
            'top_percent_edges': None,
            'bottom_percent_edges': 0.5,
            'stochastic_edges': True,
            'top_bottom_percent_edges': None,
            'shared_nodes_random_edges_percent': None,
            'all_nodes_random_edges_percent': None,
        }
    },
    'bottom_percent_stoch0.5_4': {
        'name': 'svm',
        'func': {
            "model": run_node2vec,
        },

        'edges_selection': {
            'common_nodes_feat': 'gene',
            'edges_weight_option': 'jaccard',
            'mask_edges': True,
            'self_loop': False,
            'edges_weight_limit': None,
            'edges_weight_percent': None,
            'top_percent_edges': None,
            'bottom_percent_edges': 0.5,
            'stochastic_edges': True,
            'top_bottom_percent_edges': None,
            'shared_nodes_random_edges_percent': None,
            'all_nodes_random_edges_percent': None,
        }
    },
    'bottom_percent_stoch0.5_5': {
        'name': 'svm',
        'func': {
            "model": run_node2vec,
        },

        'edges_selection': {
            'common_nodes_feat': 'gene',
            'edges_weight_option': 'jaccard',
            'mask_edges': True,
            'self_loop': False,
            'edges_weight_limit': None,
            'edges_weight_percent': None,
            'top_percent_edges': None,
            'bottom_percent_edges': 0.5,
            'stochastic_edges': True,
            'top_bottom_percent_edges': None,
            'shared_nodes_random_edges_percent': None,
            'all_nodes_random_edges_percent': None,
        }
    },
    'bottom_percent_stoch0.5_6': {
        'name': 'svm',
        'func': {
            "model": run_node2vec,
        },

        'edges_selection': {
            'common_nodes_feat': 'gene',
            'edges_weight_option': 'jaccard',
            'mask_edges': True,
            'self_loop': False,
            'edges_weight_limit': None,
            'edges_weight_percent': None,
            'top_percent_edges': None,
            'bottom_percent_edges': 0.5,
            'stochastic_edges': True,
            'top_bottom_percent_edges': None,
            'shared_nodes_random_edges_percent': None,
            'all_nodes_random_edges_percent': None,
        }
    },
    'bottom_percent_stoch0.5_7': {
        'name': 'svm',
        'func': {
            "model": run_node2vec,
        },

        'edges_selection': {
            'common_nodes_feat': 'gene',
            'edges_weight_option': 'jaccard',
            'mask_edges': True,
            'self_loop': False,
            'edges_weight_limit': None,
            'edges_weight_percent': None,
            'top_percent_edges': None,
            'bottom_percent_edges': 0.5,
            'stochastic_edges': True,
            'top_bottom_percent_edges': None,
            'shared_nodes_random_edges_percent': None,
            'all_nodes_random_edges_percent': None,
        }
    },
    'bottom_percent_stoch0.5_8': {
        'name': 'svm',
        'func': {
            "model": run_node2vec,
        },

        'edges_selection': {
            'common_nodes_feat': 'gene',
            'edges_weight_option': 'jaccard',
            'mask_edges': True,
            'self_loop': False,
            'edges_weight_limit': None,
            'edges_weight_percent': None,
            'top_percent_edges': None,
            'bottom_percent_edges': 0.5,
            'stochastic_edges': True,
            'top_bottom_percent_edges': None,
            'shared_nodes_random_edges_percent': None,
            'all_nodes_random_edges_percent': None,
        }
    },
    'bottom_percent0.5_0': {
        'name': 'svm',
        'func': {
            "model": run_node2vec,
        },

        'edges_selection': {
            'common_nodes_feat': 'gene',
            'edges_weight_option': 'jaccard',
            'mask_edges': True,
            'self_loop': False,
            'edges_weight_limit': None,
            'edges_weight_percent': None,
            'top_percent_edges': None,
            'bottom_percent_edges': 0.5,
            'stochastic_edges': False,
            'top_bottom_percent_edges': None,
            'shared_nodes_random_edges_percent': None,
            'all_nodes_random_edges_percent': None,
        }
    },
    'bottom_percent0.5_1': {
        'name': 'svm',
        'func': {
            "model": run_node2vec,
        },

        'edges_selection': {
            'common_nodes_feat': 'gene',
            'edges_weight_option': 'jaccard',
            'mask_edges': True,
            'self_loop': False,
            'edges_weight_limit': None,
            'edges_weight_percent': None,
            'top_percent_edges': None,
            'bottom_percent_edges': 0.5,
            'stochastic_edges': False,
            'top_bottom_percent_edges': None,
            'shared_nodes_random_edges_percent': None,
            'all_nodes_random_edges_percent': None,
        }
    },
    'bottom_percent0.5_2': {
        'name': 'svm',
        'func': {
            "model": run_node2vec,
        },

        'edges_selection': {
            'common_nodes_feat': 'gene',
            'edges_weight_option': 'jaccard',
            'mask_edges': True,
            'self_loop': False,
            'edges_weight_limit': None,
            'edges_weight_percent': None,
            'top_percent_edges': None,
            'bottom_percent_edges': 0.5,
            'stochastic_edges': False,
            'top_bottom_percent_edges': None,
            'shared_nodes_random_edges_percent': None,
            'all_nodes_random_edges_percent': None,
        }
    },
    'bottom_percent0.5_3': {
        'name': 'svm',
        'func': {
            "model": run_node2vec,
        },

        'edges_selection': {
            'common_nodes_feat': 'gene',
            'edges_weight_option': 'jaccard',
            'mask_edges': True,
            'self_loop': False,
            'edges_weight_limit': None,
            'edges_weight_percent': None,
            'top_percent_edges': None,
            'bottom_percent_edges': 0.5,
            'stochastic_edges': False,
            'top_bottom_percent_edges': None,
            'shared_nodes_random_edges_percent': None,
            'all_nodes_random_edges_percent': None,
        }
    },
    'bottom_percent0.5_4': {
        'name': 'svm',
        'func': {
            "model": run_node2vec,
        },

        'edges_selection': {
            'common_nodes_feat': 'gene',
            'edges_weight_option': 'jaccard',
            'mask_edges': True,
            'self_loop': False,
            'edges_weight_limit': None,
            'edges_weight_percent': None,
            'top_percent_edges': None,
            'bottom_percent_edges': 0.5,
            'stochastic_edges': False,
            'top_bottom_percent_edges': None,
            'shared_nodes_random_edges_percent': None,
            'all_nodes_random_edges_percent': None,
        }
    },
    'bottom_percent0.5_5': {
        'name': 'svm',
        'func': {
            "model": run_node2vec,
        },

        'edges_selection': {
            'common_nodes_feat': 'gene',
            'edges_weight_option': 'jaccard',
            'mask_edges': True,
            'self_loop': False,
            'edges_weight_limit': None,
            'edges_weight_percent': None,
            'top_percent_edges': None,
            'bottom_percent_edges': 0.5,
            'stochastic_edges': False,
            'top_bottom_percent_edges': None,
            'shared_nodes_random_edges_percent': None,
            'all_nodes_random_edges_percent': None,
        }
    },
    'bottom_percent0.5_6': {
        'name': 'svm',
        'func': {
            "model": run_node2vec,
        },

        'edges_selection': {
            'common_nodes_feat': 'gene',
            'edges_weight_option': 'jaccard',
            'mask_edges': True,
            'self_loop': False,
            'edges_weight_limit': None,
            'edges_weight_percent': None,
            'top_percent_edges': None,
            'bottom_percent_edges': 0.5,
            'stochastic_edges': False,
            'top_bottom_percent_edges': None,
            'shared_nodes_random_edges_percent': None,
            'all_nodes_random_edges_percent': None,
        }
    },
    'bottom_percent0.5_7': {
        'name': 'svm',
        'func': {
            "model": run_node2vec,
        },

        'edges_selection': {
            'common_nodes_feat': 'gene',
            'edges_weight_option': 'jaccard',
            'mask_edges': True,
            'self_loop': False,
            'edges_weight_limit': None,
            'edges_weight_percent': None,
            'top_percent_edges': None,
            'bottom_percent_edges': 0.5,
            'stochastic_edges': False,
            'top_bottom_percent_edges': None,
            'shared_nodes_random_edges_percent': None,
            'all_nodes_random_edges_percent': None,
        }
    },
    'bottom_percent0.5_8': {
        'name': 'svm',
        'func': {
            "model": run_node2vec,
        },

        'edges_selection': {
            'common_nodes_feat': 'gene',
            'edges_weight_option': 'jaccard',
            'mask_edges': True,
            'self_loop': False,
            'edges_weight_limit': None,
            'edges_weight_percent': None,
            'top_percent_edges': None,
            'bottom_percent_edges': 0.5,
            'stochastic_edges': False,
            'top_bottom_percent_edges': None,
            'shared_nodes_random_edges_percent': None,
            'all_nodes_random_edges_percent': None,
        }
    },
    'bottom_percent0.5_9': {
        'name': 'svm',
        'func': {
            "model": run_node2vec,
        },

        'edges_selection': {
            'common_nodes_feat': 'gene',
            'edges_weight_option': 'jaccard',
            'mask_edges': True,
            'self_loop': False,
            'edges_weight_limit': None,
            'edges_weight_percent': None,
            'top_percent_edges': None,
            'bottom_percent_edges': 0.5,
            'stochastic_edges': False,
            'top_bottom_percent_edges': None,
            'shared_nodes_random_edges_percent': None,
            'all_nodes_random_edges_percent': None,
        }
    },
    'top_bottom_stoch0.5_0': {
        'name': 'svm',
        'func': {
            "model": run_node2vec,
        },

        'edges_selection': {
            'common_nodes_feat': 'gene',
            'edges_weight_option': 'jaccard',
            'mask_edges': True,
            'self_loop': False,
            'edges_weight_limit': None,
            'edges_weight_percent': None,
            'top_percent_edges': None,
            'bottom_percent_edges': None,
            'top_bottom_percent_edges': 0.5,
            'stochastic_edges': True,
            'shared_nodes_random_edges_percent': None,
            'all_nodes_random_edges_percent': None,
        }
    },
    'top_bottom_stoch0.5_1': {
        'name': 'svm',
        'func': {
            "model": run_node2vec,
        },

        'edges_selection': {
            'common_nodes_feat': 'gene',
            'edges_weight_option': 'jaccard',
            'mask_edges': True,
            'self_loop': False,
            'edges_weight_limit': None,
            'edges_weight_percent': None,
            'top_percent_edges': None,
            'bottom_percent_edges': None,
            'top_bottom_percent_edges': 0.5,
            'stochastic_edges': True,
            'shared_nodes_random_edges_percent': None,
            'all_nodes_random_edges_percent': None,
        }
    },
    'top_bottom_stoch0.5_2': {
        'name': 'svm',
        'func': {
            "model": run_node2vec,
        },

        'edges_selection': {
            'common_nodes_feat': 'gene',
            'edges_weight_option': 'jaccard',
            'mask_edges': True,
            'self_loop': False,
            'edges_weight_limit': None,
            'edges_weight_percent': None,
            'top_percent_edges': None,
            'bottom_percent_edges': None,
            'top_bottom_percent_edges': 0.5,
            'stochastic_edges': True,
            'shared_nodes_random_edges_percent': None,
            'all_nodes_random_edges_percent': None,
        }
    },
    'top_bottom_stoch0.5_3': {
        'name': 'svm',
        'func': {
            "model": run_node2vec,
        },

        'edges_selection': {
            'common_nodes_feat': 'gene',
            'edges_weight_option': 'jaccard',
            'mask_edges': True,
            'self_loop': False,
            'edges_weight_limit': None,
            'edges_weight_percent': None,
            'top_percent_edges': None,
            'bottom_percent_edges': None,
            'top_bottom_percent_edges': 0.5,
            'stochastic_edges': True,
            'shared_nodes_random_edges_percent': None,
            'all_nodes_random_edges_percent': None,
        }
    },
    'top_bottom_stoch0.5_4': {
        'name': 'svm',
        'func': {
            "model": run_node2vec,
        },

        'edges_selection': {
            'common_nodes_feat': 'gene',
            'edges_weight_option': 'jaccard',
            'mask_edges': True,
            'self_loop': False,
            'edges_weight_limit': None,
            'edges_weight_percent': None,
            'top_percent_edges': None,
            'bottom_percent_edges': None,
            'top_bottom_percent_edges': 0.5,
            'stochastic_edges': True,
            'shared_nodes_random_edges_percent': None,
            'all_nodes_random_edges_percent': None,
        }
    },
    'top_bottom_stoch0.5_5': {
        'name': 'svm',
        'func': {
            "model": run_node2vec,
        },

        'edges_selection': {
            'common_nodes_feat': 'gene',
            'edges_weight_option': 'jaccard',
            'mask_edges': True,
            'self_loop': False,
            'edges_weight_limit': None,
            'edges_weight_percent': None,
            'top_percent_edges': None,
            'bottom_percent_edges': None,
            'top_bottom_percent_edges': 0.5,
            'stochastic_edges': True,
            'shared_nodes_random_edges_percent': None,
            'all_nodes_random_edges_percent': None,
        }
    },
    'top_bottom_stoch0.5_6': {
        'name': 'svm',
        'func': {
            "model": run_node2vec,
        },

        'edges_selection': {
            'common_nodes_feat': 'gene',
            'edges_weight_option': 'jaccard',
            'mask_edges': True,
            'self_loop': False,
            'edges_weight_limit': None,
            'edges_weight_percent': None,
            'top_percent_edges': None,
            'bottom_percent_edges': None,
            'top_bottom_percent_edges': 0.5,
            'stochastic_edges': True,
            'shared_nodes_random_edges_percent': None,
            'all_nodes_random_edges_percent': None,
        }
    },
    'top_bottom_stoch0.5_7': {
        'name': 'svm',
        'func': {
            "model": run_node2vec,
        },

        'edges_selection': {
            'common_nodes_feat': 'gene',
            'edges_weight_option': 'jaccard',
            'mask_edges': True,
            'self_loop': False,
            'edges_weight_limit': None,
            'edges_weight_percent': None,
            'top_percent_edges': None,
            'bottom_percent_edges': None,
            'top_bottom_percent_edges': 0.5,
            'stochastic_edges': True,
            'shared_nodes_random_edges_percent': None,
            'all_nodes_random_edges_percent': None,
        }
    },
    'top_bottom_stoch0.5_8': {
        'name': 'svm',
        'func': {
            "model": run_node2vec,
        },

        'edges_selection': {
            'common_nodes_feat': 'gene',
            'edges_weight_option': 'jaccard',
            'mask_edges': True,
            'self_loop': False,
            'edges_weight_limit': None,
            'edges_weight_percent': None,
            'top_percent_edges': None,
            'bottom_percent_edges': None,
            'top_bottom_percent_edges': 0.5,
            'stochastic_edges': True,
            'shared_nodes_random_edges_percent': None,
            'all_nodes_random_edges_percent': None,
        }
    },
    'top_bottom_stoch0.5_9': {
        'name': 'svm',
        'func': {
            "model": run_node2vec,
        },

        'edges_selection': {
            'common_nodes_feat': 'gene',
            'edges_weight_option': 'jaccard',
            'mask_edges': True,
            'self_loop': False,
            'edges_weight_limit': None,
            'edges_weight_percent': None,
            'top_percent_edges': None,
            'bottom_percent_edges': None,
            'top_bottom_percent_edges': 0.5,
            'stochastic_edges': True,
            'shared_nodes_random_edges_percent': None,
            'all_nodes_random_edges_percent': None,
        }
    }
}
def run_embedding(copd):

    run_and_save_embedding(copd, emb_type='node2vec', config =embedding_config)

def run_and_save_embedding(copd, emb_type=None, config=None):
    for name, model in config.items():

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
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

                func_kwargs = { 'used_nodes':args.common_nodes_feat,"edges_weight_option":model['edges_selection']['edges_weight_option']}
                _, copd_geometric_dataset, weight_adj = get_config(model['name'], copd, used_nodes=args.common_nodes_feat,
                                                                 edges_weight_option=model['edges_selection'][
                                                                     'edges_weight_option'])
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
        emb_func(copd, copd_geometric_dataset, weight_adj) # pred must be real prediction


if __name__ == '__main__':
    copd = all_datasets.Copd(path=args.copd_path, data=args.copd_data, time_stamp=args.time_stamp,
                             undirected=not args.directed)
    if args.run_embedding:
        run_embedding(copd)