import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Generator, self).__init__()

        main = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(num_features=hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, 2 * hidden_size),
            nn.BatchNorm1d(num_features=2*hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2 * hidden_size, output_size),
            nn.Sigmoid()
        )
        self.main = main

    def forward(self, noise):
        output = self.main(noise)
        return output


class Discriminator(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(Discriminator, self).__init__()

        main = nn.Sequential(
            nn.Linear(input_size, 2 * hidden_size),
            nn.BatchNorm1d(num_features=2*hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2 * hidden_size, hidden_size),
            nn.BatchNorm1d(num_features=hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, 1),
        )
        self.main = main

    def forward(self, inputs):
        output = self.main(inputs)
        return output.view(-1)
