import argparse
import torch
import random
import numpy as np
'''
example command
 --run_gcn  --verbose --t1_t2_alpha 10000 10000  0.3 --log  --plot_loss --epoch 200 --pseudo_label_topk --topk 1 --epoch 400
 python __init__.py --check_condition all --common_nodes_feat gene --cv 3 --report_performance
'''

#--Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='gene_disease', help='specify type of dataset to be used')
parser.add_argument('--time_stamp', type=str, default='07_14_19_46', help='time_stamp version of copd_data')
parser.add_argument('--copd_path', type=str, default=f'data/gene_disease/07_14_19_46/raw/', help='path containing copd_label{time_stamp} dataset')
parser.add_argument('--copd_data', type=str, default='copd_label', help='name of the gene_disease dataset; default = copd_label')
parser.add_argument('--emb_name', type=str, default='no_feat', help='name of embedding type being used eg attentionwalk, bine, node2vec, (gcn, gat, graph sage)')
parser.add_argument('--edges_weight_option', type=str, default='no', help='edges_weight option such as jaccards etc. ')
# parser.add_argument('--emb_path', type=str, default=f"data/gene_disease/07_14_19_46/gene_disease/processed/embedding/", help='name of the gene_disease dataset; default = copd_label')
# parser.add_argument('--emb_path', type=str, default=f"data/gene_disease/07_14_19_46/gene_disease/processed/embedding/", help='name of the gene_disease dataset; default = copd_label')
parser.add_argument('--emb_path', type=str, default=None, help='name of the gene_disease dataset; default = copd_label')
parser.add_argument('--subgraph', action="store_true", help='NOT CURRENTLY COMPATIBLE WITH THE PROGRAM;Use only node in the largest connected component instead of all nodes disconnected graphs')
parser.add_argument('--with_gene', action="store_true", help='plot plot_2d() with genes')
parser.add_argument('--plot_2d_func', type=str, default='tsne', help='plotting.plot_2d() with genes')
parser.add_argument('--arch', type=str, default='gcn', help='architecutre name to be run eg. GAT, GCN ') #todo graph_sage is not supported yet
parser.add_argument('--add_features', action='store_true', help='added_features to the nodes')
# parser.add_argument('--common_nodes_feat', action='store_true', help='use common nodes as feature to nodes')
# parser.add_argument('--common_nodes_feat', action='store_true', help='use common nodes as feature to nodes')
parser.add_argument('--common_nodes_feat', type=str, default='no', help='all => use gene and disease; gene => create edges between gene, disease => create edges between disease')
parser.add_argument('--run_lr', action='store_true', help='run logistic regression with grpah embedding of choice node2vec, bine, gcn,and attentionwalk')
parser.add_argument('--run_mlp', action='store_true', help='run multi-layer perceptron. Input is disease whose features are genes.')
parser.add_argument('--run_gnn', action='store_true', help='run multi-layer perceptron. Input is disease whose features are genes.')
parser.add_argument('--run_node2vec', action='store_true', help='run node2vec on raw copd data.')
parser.add_argument('--run_gcn_on_disease_graph', action='store_true', help='run gcn on disease only graph where edges between diseases are form iff they share at least 1 gene')
parser.add_argument('--th', default=200, help='amount of shared gene requires to form an edge in disease graph')

# -- hyper paramters
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.09, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=0.006, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--split', type=float, default=0.7, help='training split with range=[0,1] ')
parser.add_argument('--weighted_class', default=[1,1,1,1,1], nargs='+', help='list of weighted_class of diseases only in order <0,1,2,3,4,5>')

# -- psuedo_lable specific hyper parrameters
parser.add_argument('--pseudo_label_all', action='store_true', help='psudo_label_all ')
parser.add_argument('--pseudo_label_topk', action='store_true', help='psudo_label_topk add top k most confidence; --topk must be specified')
parser.add_argument('--topk', default="50", help='amount of topk to be label every epoch')
parser.add_argument('--t1_t2_alpha', default=[10000,10000,0.3], nargs='+', help='[T1, T2, af] followed literature in pseudo_label where T1 and T2 are threshold for epcoh which will determine specific alpha value to be used')

# -- not shown to improve the output
parser.add_argument('--pseudo_label_topk_with_replacement', action='store_true', help='psudo_label_topk add top k most confidence; --topk must be specified')


# -- GAT specific hyper parameters
parser.add_argument('--heads', type=int, default=8, help='number of heads to neighbors important for GAT')

# -- GraphSage Specific hyper parameters
parser.add_argument('--aggr', type=str, default='mean', help="aggregator for graphsage: ['add', 'mean', 'max']")

#--------baseline
parser.add_argument('--run_svm', action='store_true', help="run svm baseline")
parser.add_argument('--run_rf', action='store_true', help="run svm random forest")

#-- utilities
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--edges_weight_limit', type=float, default=None, help='accepted range from 0 to 1; lower bound of edges weight that will be added')
parser.add_argument('--edges_weight_percent', type=float, default=None, help='selected edges by percentile. eg 20 percent => select all edges that are less than max value of top 20 percent of the lowest value')
parser.add_argument('--top_percent_edges', type=float, default=None, help='accepted range from 0 to 1; percentage of edges to be selected')
parser.add_argument('--stochastic_edges', action='store_true', help="add k percent edges stocastically")
# parser.add_argument('--ensemble', type=int, default=None, help="apply ensemble to the chosen models n number of times")
parser.add_argument('--ensemble', action='store_true', help="apply ensemble to n number of models with independent config")
parser.add_argument('--mask_edges', action='store_true', help="mask adj matrix with original edges (before edges of the same types are introduced)")
parser.add_argument('--self_loop', action='store_true', help="add self loop")
parser.add_argument('--num_run', type=int, default=1, help='number of run to repeat the experiment')
parser.add_argument('--directed', action='store_true', help="directed == True")
parser.add_argument('--verbose', action="store_true", help='verbose status')
parser.add_argument('--plot_reports', action="store_true", help='plot all of the function related to report including datasets and models')
parser.add_argument('--report_performance', action="store_true", help='report precision recall f1-score pred auc')
parser.add_argument('--plot_all', action="store_true", help='plot loss, notrain, train')
parser.add_argument('--plot_emb', action="store_true", help='plot nodes emb')
parser.add_argument('--plot_loss', action="store_true", help='plot loss')
parser.add_argument('--plot_no_train', action="store_true", help='plot no_train graph ')
parser.add_argument('--plot_train', action="store_true", help='plot train_graph ')
parser.add_argument('--plot_roc', action="store_true", help='plot roc for report performancce')
parser.add_argument('--tuning', action="store_true", help='hyper-paramter tuning')
parser.add_argument('--log', action="store_true",  help='logging training infomation such as accuracy and confusino matrix')
parser.add_argument('--no_cuda', action="store_true",  help='Disable cuda training ')
parser.add_argument('--check_condition', default=None, nargs='+', help='checking for error by applying the same conditions to all models')
# parser.add_argument('--check_condition', action="store_true",  help='checking for error by applying the same conditions to all models')
parser.add_argument('--cv', type=str, default=None,  help='activate crossvalidation input must have between positive integer')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)