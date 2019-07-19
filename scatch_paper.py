import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv  # noqa

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, T.NormalizeFeatures())
data = dataset[0]

# create Data(edge_index, test_mask, train_mask, val_maask, x, y)
# edge_index = [2, num_egdes]
# test_mask = [number of nodes] # number of non zero is == num_test.
# x = [number of nodes,
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, 16, cached=True) #specify input dim and output dim
        self.conv2 = GCNConv(16, dataset.num_classes, cached=True)
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)

    def forward(self):
        x, edge_index = data.x, data.edge_index # this is the input of the Net
        # x = data.x; dim = 2708, 16
        # edge_index = data.edge_index; dim = [2,10556]
        # y = label ; dim = 2708

        # print(x.shape)
        # print(edge_index.shape)
        # exit()
        x = F.relu(self.conv1(x, edge_index)) # conv1(x, edge_index, edge_weight)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


def train():
    model.train()
    optimizer.zero_grad()

    # reg_lambda = 1.0
    # l2_reg = 0
    # for W in mdl.parameters():
    #     l2_reg += *W.norm(2)
    # batch_loss = (1 / N_train) * (y_pred - batch_ys).pow(2).sum() + reg_lambda * l2_reg
    # ## BACKARD PASS
    # batch_loss.backward()  # Use autograd to compute the backward pass. Now w will have gradients

    # print([w for w in model.parameters()]) # weight and bias of each layer

    F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()


def test():
    model.eval()
    logits, accs = model(), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

if __name__ == "__main__":
    best_val_acc = test_acc = 0
    for epoch in range(1, 201):
        train()
        train_acc, val_acc, tmp_test_acc = test()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        print(log.format(epoch, train_acc, best_val_acc, test_acc))
