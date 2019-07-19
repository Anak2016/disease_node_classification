import argparse
import torch
import random
import numpy as np

#--Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--time_stamp', type=str, default='07_14_19_46', help='time_stamp version of copd_data')
parser.add_argument('--copd_path', type=str, default='data/gene_disease/', help='path containing copd_label{time_stamp} dataset')
parser.add_argument('--copd_data', type=str, default='copd_label', help='name of the gene_disease dataset; default = copd_label')
parser.add_argument('--emb_name', type=str, default='node2vec', help='name of embedding type being used')
parser.add_argument('--emb_path', type=str, default=f"output/gene_disease/embedding/", help='name of the gene_disease dataset; default = copd_label')
parser.add_argument('--subgraph', action="store_true", help='NOT CURRENTLY COMPATIBLE WITH THE PROGRAM;Use only node in the largest connected component instead of all nodes disconnected graphs')
parser.add_argument('--with_gene', action="store_true", help='plot_2d() with genes')
parser.add_argument('--plot_2d_func', type=str, default='tsne', help='plot_2d() with genes')

# -- hyper paramters
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.09, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=0.006, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')

#-- utilities
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--verbose', action="store_true", help='verbose status')
parser.add_argument('--plot', action="store_true", help='plot graph')
parser.add_argument('--tuning', action="store_true", help='hyper-paramter tuning')
parser.add_argument('--log', action="store_true",  help='logging training infomation such as accuracy and confusino matrix')
parser.add_argument('--no_cuda', action="store_true",  help='Disable cuda training ')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)