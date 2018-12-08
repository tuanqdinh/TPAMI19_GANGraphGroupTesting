import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Generator, self).__init__()

        self.output_dim = output_size
        self.input_dim = input_size
        self.hidden_dim = hidden_size

        preprocess = nn.Sequential(
            nn.Linear(self.input_dim, 4*4*4*self.hidden_dim),
            nn.ReLU(True),
        )
        block1 = nn.Sequential(
            nn.ConvTranspose2d(4*self.hidden_dim, 2*self.hidden_dim, 5),
            nn.ReLU(True),
        )
        block2 = nn.Sequential(
            nn.ConvTranspose2d(2*self.hidden_dim, self.hidden_dim, 5),
            nn.ReLU(True),
        )
        deconv_out = nn.ConvTranspose2d(self.hidden_dim, 1, 8, stride=2)

        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.preprocess = preprocess
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        output = self.preprocess(input)
        output = output.view(-1, 4*self.hidden_dim, 4, 4)
        #print output.size()
        output = self.block1(output)
        #print output.size()
        output = output[:, :, :7, :7]
        #print output.size()
        output = self.block2(output)
        #print output.size()
        output = self.deconv_out(output)
        output = self.sigmoid(output)
        #print output.size()
        return output.view(-1, self.output_dim)


class Discriminator(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(Discriminator, self).__init__()
        self.hidden_dim = hidden_size
        main = nn.Sequential(
            nn.Conv2d(1, self.hidden_dim, 5, stride=2, padding=2),
            # nn.Linear(OUTPUT_self.hidden_dim, 4*4*4*self.hidden_dim),
            nn.ReLU(True),
            nn.Conv2d(self.hidden_dim, 2*self.hidden_dim, 5, stride=2, padding=2),
            # nn.Linear(4*4*4*self.hidden_dim, 4*4*4*self.hidden_dim),
            nn.ReLU(True),
            nn.Conv2d(2*self.hidden_dim, 4*self.hidden_dim, 5, stride=2, padding=2),
            # nn.Linear(4*4*4*self.hidden_dim, 4*4*4*self.hidden_dim),
            nn.ReLU(True),
            # nn.Linear(4*4*4*self.hidden_dim, 4*4*4*self.hidden_dim),
            # nn.LeakyReLU(True),
            # nn.Linear(4*4*4*self.hidden_dim, 4*4*4*self.hidden_dim),
            # nn.LeakyReLU(True),
        )
        self.main = main
        self.output = nn.Linear(4*4*4*self.hidden_dim, 1)

    def forward(self, input):
        input = input.view(-1, 1, 28, 28)
        out = self.main(input)
        out = out.view(-1, 4*4*4*self.hidden_dim)
        out = self.output(out)
        return out.view(-1)
