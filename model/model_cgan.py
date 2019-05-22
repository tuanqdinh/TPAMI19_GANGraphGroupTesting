import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
img_shape = (65, 65)

class Generator(nn.Module):
    def __init__(self, input_size, label_size, hidden_size, output_size, adj):
        super(Generator, self).__init__()
        self.adj = adj

        self.block_x = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(num_features=hidden_size),
            nn.ReLU()
        )
        self.block_y = nn.Sequential(
            nn.Linear(label_size, hidden_size),
            nn.BatchNorm1d(num_features=hidden_size),
            nn.ReLU()
        )
        self.block1 = nn.Sequential(
            nn.Linear(2 * hidden_size, 8 * hidden_size),
            nn.BatchNorm1d(num_features=8 * hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(8 * hidden_size, output_size),
            )
        self.block2 = nn.Sequential(
            nn.Tanh()
        )

    def forward(self, noise, label):
        x = self.block_x(noise)
        y = self.block_y(label)
        try:
            x = torch.cat([x, y], 1)
        except:
            from IPython import embed; embed()

        output = self.block1(x)
        output = torch.mm(output, self.adj)
        output = self.block2(output)
        return output

class Discriminator(nn.Module):

    def __init__(self, input_size, label_size, hidden_size, output_size, adj):
        super(Discriminator, self).__init__()

        self.adj = adj
        self.block_x = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(num_features=hidden_size),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.block_y = nn.Sequential(
            nn.Linear(label_size, hidden_size),
            nn.BatchNorm1d(num_features=hidden_size),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # self.block0 = nn.Sequential(
        #     nn.Linear(2 * hidden_size, input_size),
        #     nn.BatchNorm1d(num_features=input_size),
        #     nn.LeakyReLU(0.2, inplace=True)
        # )
        self.block1 = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.BatchNorm1d(num_features=hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, input, label):
        # from IPython import embed; embed()
        x = self.block_x(input)
        y = self.block_y(label)
        x = torch.cat([x, y], 1)

        # x = torch.mm(x, self.adj)
        output = self.block1(x)
        return output.view(-1)
