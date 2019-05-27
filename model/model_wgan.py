import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
img_shape = (65, 65)

class Generator(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, adj):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(input_size, 256, normalize=False),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.ReLU()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        return img

class Generator2(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, adj):
        super(Generator, self).__init__()
        self.adj = adj
        self.block1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(num_features=hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, 2 * hidden_size),
            nn.BatchNorm1d(num_features=2*hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2 * hidden_size, output_size),
            )
        self.block2 = nn.Sequential(
            nn.ReLU()
        )

    def forward(self, noise):
        output = self.block1(noise)
        output = torch.mm(output, self.adj)
        output = self.block2(output)
        return output

class Discriminator2(nn.Module):

    def __init__(self, adj, input_size, hidden_size, output_size=1):
        super(Discriminator, self).__init__()

        self.adj = adj
        self.block0 = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.BatchNorm1d(num_features=input_size),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.block1 = nn.Sequential(
            nn.Linear(input_size, 2 * hidden_size),
            nn.BatchNorm1d(num_features=2*hidden_size),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.block2 = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.BatchNorm1d(num_features=hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, inputs):
        x = self.block0(inputs)
        x = torch.mm(x, self.adj)
        x = self.block1(x)
        output = self.block2(x)
        return output.view(-1)
        # return F.log_softmax(output, dim=1)


class Discriminator(nn.Module):
    def __init__(self, adj, input_size, hidden_size, output_size=1):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity
