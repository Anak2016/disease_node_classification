
conda activate pytorch_python3.7



#todo (final decision, in order )
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
# > try using feature as graph's strucutre with other bipartite
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