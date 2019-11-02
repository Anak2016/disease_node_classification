
conda activate pytorch_python3.7
--check_condition svm  --report_performance --common_nodes_feat gene --edges_weight_option jaccard --cv 3 --num_run 3 --mask_edges --top_percent_edges 0 --stochastic_edges --ensemble
--run_node2vec  --report_performance --common_nodes_feat gene --edges_weight_option jaccard --cv 3 --num_run 3 --mask_edges --top_percent_edges 0.05 --stochastic_edges
Computing transition probabilities:   0%|          | 0/2996 [00:00<?, ?it/s]save node2vec emb to data/gene_disease/07_14_19_46/processed/embedding/node2vec/node2vec_emb_fullgraph_common_nodes_feat=gene07_14_19_46_added_edges=disease_no_top_k=0.4_mask=True_stoch2.txt
#todo (final decision, in order )
# > write ensemble
#   :
# > 2 things
#   : check old result whether edges are added correctly
#       >> check "number of added edges {var}" # what is expected output??
#   : use esemble on 3 models with edges added stocastically and compare tis result with the above 2 approach
#       >> consider using built in scikit learn ensemble
#   : writing introduction to the research paper using overleaf
# > speed up the loading process
#   :check what part can be improve
# > for edges_percent, add n percent of edges with highest score
#   : should I use sparse?
#   : how to do this? sorted by value but get index. use index of the first n percent to create a graph
# > once finish with todoo inside of jaccard coeff rerun no_emb, gene, and get performance
# > check if node2vec use edges weight to calculate embedding
# > equalize distrition to be more throughtly distributed from 0-1
# > try stocastically added edges depends on its weighEt.
# > run top 5, 25, 50, 100 percent of added edges.
#   : run it for all models
# > add weight edges
#   : preprocess new data set to match with the existing code. including copd and geometric_dataset
#       >> use new data withon added edges first, (if i have time use onw that have edges)
#       >> test all models with new dataset
# > find results by adding more or les sedge betwee ndisease and check the algorithm performance with respect to differnet embeedign
#       and classifier
# > run gcn on new copd_label data with added dsi and dpi and edge features
# > run pseudo label on the the best one and see decide if I choose try it with other model
# > run cora gcn with node feature on jupyter notebook with trainset = 30 percent and compare if the result is higher than 84.3
# > fix this add node features and feature as graph strucutre RuntimeError: Invalid index in scatterAdd at..\aten\src\TH / generic / THTensorEvenMoreMath.cpp: 523
# > create movin average MA 10, 20 on loss function
# > try my method with Citeseer, Pubmed, NELL
# > figure out how to learn edges weight,
# > try with loss function variation, link prediction,
# > try node_features graph and node_node graph where I use entropy regularization as disagreement
# > create saving path for gcn_on_node_feature_graph with cora
# > check if the implementation is correct
#       :do I mask classes and no_classes correctly??
#       :class distribution between each classes
# > try using feate as graph's strucutre with other bipartite
#      : read literature that learn feature in grpah strucutre.
#      : find bipartite datset to be used
# > what if i remove outlier in node embedding? will it help improve node classification?
# > different run on the same config should have different outcome.
# > label top k non_label with top confidence. iteratively.
# > make gat and graph sage work
# > run 3 times to get average output, and plot them.
# > build logistic regression as in Graph Convolutional Networks
# > How do i make use of bias in GCN? when do i use it to improboe the performance?
# > choose base line for the experiment
#       : what is the standard I should use?
# > consider adding residual for deeper GCN
# 0. create dataset for dataLoader with the followin format
#       >> check if attention_walk, bine, and node2vec preserved order of input nodes.
#              : if it does, we can can concat node embedding directly, if not i have to change the code of
#                   embedding function so that node orders are preserved.
#             :attention_walk, bine, node2vec



# --
# use default setting for all of the deep learning model
# default setting GCN #_hidden_layer = 2, output of hidden layer = 16

# steps to validate
# > use train,val, test to test the resultse.
# > optimal number of hidden units < number of input
#       :sometimes 2 hidden units works best with little data


# get dim of data("train_mask")
# >> print(np.array(list(data('train_mask'))[0][1]).nonzero()[0].shape)