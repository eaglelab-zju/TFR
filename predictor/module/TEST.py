import torch
import torch.nn as nn

from predictor.module.GNNs import GCN, MLP
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_scatter import scatter
from torch_geometric.nn import MessagePassing


class MPNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, decoder_dropout=0.5, att_softmax=False):
        super(MPNN, self).__init__()
        self.att_layer0 = ATTLayer(in_channels, hidden_channels, att_softmax)
        # self.att_layer1 = ATTLayer(hidden_channels, out_channels)
        # self.conv = SAGEConv(hidden_channels, out_channels)
        self.lin_transform = nn.Linear(hidden_channels, out_channels)
        self.dropout = decoder_dropout

    def forward(self, x, adj):
        x, confidence = self.att_layer0(x, adj)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin_transform(x)
        return x, confidence


class ATTLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, att_softmax=False):
        super().__init__(aggr='mean')  # "Add" aggregation (Step 5).
        self.lin_src = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_dst = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_att = nn.Linear(2 * out_channels, 1, bias=False)
        # self.lin_yh = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_2hh = nn.Linear(2 * in_channels, out_channels, bias=False)
        self.alpha = None
        self.att_softmax = att_softmax
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()
        self.lin_att.reset_parameters()

    def forward(self, x, adj):
        adj, _ = add_self_loops(adj, num_nodes=x.size(0))
        edge_index = adj
        out = self.propagate(edge_index, x=x)
        confidence = self.confidence
        return out, confidence

    def message(self, x_i, x_j, edge_index, size_i):
        # Compute attention coefficients.
        x_i1 = self.lin_src(x_i)
        x_j1 = self.lin_dst(x_j)
        alpha = self.lin_att(torch.cat([F.relu(x_i1), F.relu(x_j1)], dim=-1))
        alpha = F.sigmoid(alpha)
        # alpha = softmax(alpha, edge_index[1], num_nodes=size_i)
        # alpha = F.dropout(alpha, p=self.alpha_dropout, training=self.training)
        self.confidence = scatter(alpha, edge_index[1], dim=0, reduce='mean').detach()
        if self.att_softmax:
            alpha = softmax(alpha, edge_index[1], num_nodes=size_i)
        x_ij = torch.cat([x_i, x_j], dim=-1)
        x_ij = self.lin_2hh(x_ij)
        msg = alpha * x_ij
        return msg

    def aggregate(self, msg, index, dim_size):
        aggr_out = scatter(msg, index, dim=self.node_dim, dim_size=dim_size, reduce=self.aggr)
        return aggr_out

    def update(self, aggr_out):
        return aggr_out



class MLP_TEST_1(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(MLP_TEST_1, self).__init__()
        self.layer_0 = nn.Linear(in_channels, hidden_channels, bias=False)
        self.layer_1 = nn.Linear(in_channels, hidden_channels)
        self.layer_1_1 = nn.Linear(2 * hidden_channels, hidden_channels)
        self.layer_2 = nn.Linear(hidden_channels, hidden_channels)
        self.layer_2_1 = nn.Linear(2 * hidden_channels, hidden_channels)
        self.layer_3 = nn.Linear(hidden_channels, out_channels)
        # self.att_mlp = MLP(in_channels * 2, hidden_channels, 1, 2, dropout)
        self.att_linear = nn.Linear(hidden_channels * 2, 1, bias=False)
        self.dropout = dropout

    def forward(self, x, edge_index):
        row, col = edge_index
        x_0 = self.layer_0(x)
        x_0 = F.dropout(x_0, p=self.dropout, training=self.training)
        x_row = x_0[row]
        x_col = x_0[col]
        val = torch.cat((x_row, x_col), dim=1)
        val = self.att_linear(val)
        val = F.sigmoid(val)
        # val = softmax(val, col, num_nodes=x.size(0))
        val = F.dropout(val, p=self.dropout, training=True)

        x = self.layer_1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(x)
        x_row = x[row]
        x_col = x[col]
        x_row_col = torch.cat((x_row, x_col), dim=1)
        x_row_col = val * x_row_col
        x = scatter(x_row_col, row, dim=0, reduce='mean')
        x = self.layer_1_1(x)

        # x = self.layer_2(x)
        # x = F.dropout(x, p=self.dropout, training=self.training)
        # x = F.relu(x)
        # x_row = x[row]
        # x_col = x[col]
        # x_row_col = torch.cat((x_row, x_col), dim=1)
        # x_row_col = val * x_row_col
        # x = scatter(x_row_col, row, dim=0, reduce='mean')
        # x = self.layer_2_1(x)

        output = self.layer_3(x)
        confidence = scatter(val, row, dim=0, reduce='mean')
        return output, confidence


class MLP_TEST_0(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_layers, n_nodes, dropout=0.5, use_bn=True):
        super(MLP_TEST_0, self).__init__()
        self.input_layer_1 = nn.Linear(in_channels, hidden_channels)
        self.input_layer_2 = nn.Linear(n_nodes, hidden_channels)
        self.MLP = MLP(hidden_channels * 3, hidden_channels * 2, hidden_channels, n_layers, dropout, use_bn)
        # self.MLP_1 = MLP(hidden_channels, hidden_channels, hidden_channels, n_layers, dropout, use_bn)
        # self.MLP_2 = MLP(hidden_channels, hidden_channels, hidden_channels, n_layers, dropout, use_bn)
        self.output_layer = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, adj):
        x = self.input_layer_1(x)
        nei_feats = torch.spmm(adj, x)
        adj = self.input_layer_2(adj)
        # x_feat= self.MLP_1(x)
        # adj_feat = self.MLP_2(adj)
        # input_feats = torch.cat((x_feat, adj_feat), dim=1)
        input_feats = torch.cat((x, nei_feats, adj), dim=1)
        output_feats = self.MLP(input_feats)
        output = self.output_layer(output_feats)
        return output


def gcn_norm(adj):
    adj = add_self_loops(adj)[0]
    row, col = adj.indices()
    deg = degree(col, adj.shape[0], dtype=torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    new_adj = torch.sparse_coo_tensor(adj.indices(), norm, adj.shape)
    return new_adj

def normalize_sparse(mx):
    """Row-normalize sparse COO tensor"""
    rowsum = torch.sparse.sum(mx, dim=1).to_dense()
    r_inv = torch.pow(rowsum, -1)
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.sparse_coo_tensor(
        torch.arange(mx.size(0)).repeat(2, 1).to(mx.device),
        r_inv,
        (mx.size(0), mx.size(0))
    )
    mx = torch.sparse.mm(r_mat_inv, mx)
    return mx

def joint_prob(x, y, num_bins):
    joint_hist = torch.histc(x + num_bins * y, bins=num_bins**2, min=0, max=num_bins**2-1)
    joint_prob = joint_hist / joint_hist.sum()
    return joint_prob.view(num_bins, num_bins)


def marginal_prob(joint_prob):
    x_prob = joint_prob.sum(dim=1)
    y_prob = joint_prob.sum(dim=0)
    return x_prob, y_prob


def mutual_information(x, y, num_bins):
    joint_prob_dist = joint_prob(x, y, num_bins)
    x_prob, y_prob = marginal_prob(joint_prob_dist)

    mi = 0.0
    for i in range(num_bins):
        for j in range(num_bins):
            if joint_prob_dist[i, j] > 0:
                mi += joint_prob_dist[i, j] * torch.log(joint_prob_dist[i, j] / (x_prob[i] * y_prob[j]))
    return mi


class DGI(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_layers, n_classes, dropout=0.5, norm_info=None,
                 act='F.relu', input_layer=False, output_layer=False, bias=True):
        super(DGI, self).__init__()
        self.gcn = GCN(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels,
                        n_layers=n_layers, dropout=dropout, norm_info=norm_info,
                        act=act, input_layer=input_layer,
                        output_layer=output_layer, bias=bias)
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.disc = Discriminator(out_channels, n_classes)

    def forward(self, feature, adj, mask, random_mask, label, random_label):
        h = self.gcn(feature, adj)
        h_1 = h[mask]
        h_2 = h[random_mask]
        ret = self.disc(h_1, h_2, label, random_label)
        return ret

    # Detach the return variables
    def embed(self, seq, adj):
        h_1 = self.gcn(seq, adj)
        return h_1


class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq):
        return torch.mean(seq, 0)


class Discriminator(nn.Module):
    def __init__(self, hidden_channels, n_classes):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(hidden_channels, n_classes, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, h_1, h_2, label, fake_label):
        sc_1 = self.f_k(h_1, label).T
        sc_2 = self.f_k(h_2, fake_label).T
        logits = torch.cat((sc_1, sc_2), 1)
        return logits


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret




