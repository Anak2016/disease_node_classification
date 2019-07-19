#todo
# > build GAT, GrpahSage, logistic regression
# > git ignore dataset
# > apply gcn to karate club. it has no features how could GCN. works at all??
#       : this steps needs to be done, so I can compare result between gcn_emb and node2vec_emb
#       : after seeing how ti works use GCN on copd dataset with no node embedding
# > see if i can do cross validation.
# > FEED THE FILE CREATED ABOVE to attentionWalk.py
# > FEED THE FILE CREATED ABOVE to bine.py.
# 0. create dataset for dataLoader with the followin format
#       >> check if attention_walk, bine, and node2vec preserved order of input nodes.
#              : if it does, we can can concat node embedding directly, if not i have to change the code of
#                   embedding function so that node orders are preserved.
#             :attention_walk, bine, node2vec
# 1. run GCN
# 2. GAT
# 3. GraphSage
# 4. CapsuleGCN



# --
# use default setting for all of the deep learning model
# default setting GCN #_hidden_layer = 2, output of hidden layer = 16

# steps to validate
# > use train,val, test to test the results.
# > optimal number of hidden units < number of input
#       :sometimes 2 hidden units works best with little data


# get dim of data("train_mask")
# >> print(np.array(list(data('train_mask'))[0][1]).nonzero()[0].shape)