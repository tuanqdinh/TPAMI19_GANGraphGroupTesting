import torch
import torch.nn as nn



# input: batch_size * nc * 64 * 64
# output: batch_size * k * 1 * 1
class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(Encoder, self).__init__()

        self.input_dim = input_dim**2
        self.model_encode = nn.Sequential(
            nn.Linear(input_dim, input_dim//2),
            nn.ReLU(),
            # nn.BatchNorm1d(100)
            nn.Linear(input_dim//2, input_dim//4),
            nn.ReLU(),
            nn.Linear(input_dim//2, embed_dim)
        )

    def forward(self, x):
        z = self.model_encode(x.view(-1, self.input_dim))
        return z

class Decoder(nn.Module):
    def __init__(self, embed_dim, output_dim):
        super(Decoder, self).__init__()

        self.embed_dim = embed_dim
        self.model_encode = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            # nn.BatchNorm1d(100)
            nn.Linear(embed_dim * 2, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, output_dim)
        )

    def forward(self, z):
        x = self.model_encode(z.view(-1, self.embed_dim))
        return x


def grad_norm(m, norm_type=2):
    total_norm = 0.0
    for p in m.parameters():
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.1)
        m.bias.data.fill_(0)
