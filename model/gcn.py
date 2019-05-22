import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch, math

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class Generator(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, adj):
        super(Generator, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(num_features=hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, 2 * hidden_size),
            nn.BatchNorm1d(num_features=2*hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2 * hidden_size, output_size),
            nn.Tanh()
        )

    def forward(self, noise):
        output = self.main(noise)
        return output


class Discriminator(nn.Module):

    def __init__(self, input_size, hidden_size, adj, output_size=1):
        super(Discriminator, self).__init__()

        self.adj = adj
        self.gc1 = GraphConvolution(input_size, 2*hidden_size)
        self.block1 = nn.Sequential(
            nn.BatchNorm1d(num_features=2*hidden_size),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.gc2 = GraphConvolution(2*hidden_size, hidden_size)
        self.block2 = nn.Sequential(
            nn.BatchNorm1d(num_features=hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, inputs):
        inputs = inputs.unsqueeze(2)
        x = self.gc1(inputs, self.adj)
        x = self.block1(x)
        x = self.gc2(x, self.adj)
        output = self.block2(x)
        return output.view(-1)
        # return F.log_softmax(output, dim=1)
