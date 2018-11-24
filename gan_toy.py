import os, sys
import numpy as np
import sklearn.datasets
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.autograd as autograd
import torch.optim as optim

sys.path.append(os.getcwd())
import tflib as lib
import tflib.plot
from models.toy import Generator, Discriminator
from utils.data_helpers import inf_train_gen

torch.manual_seed(1)

MODE = 'wgan-gp'  # wgan or wgan-gp
DATASET = '8gaussians'  # 8gaussians, 25gaussians, swissroll

FIXED_GENERATOR = False  # whether to hold the generator fixed at real data plus
# Gaussian noise, as in the plots in the paper
LAMBDA = .1  # Smaller lambda seems to help for toy tasks specifically
CRITIC_ITERS = 5  # How many critic iterations per generator iteration
BATCH_SIZE = 256  # Batch size
ITERS = 100000  # how many generator iterations to train for

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.to(device)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

def generate_image(true_dist, frame_index):
    """
    Generates and saves a plot of the true distribution, the generator, and the
    critic.
    """
    N_POINTS = 128
    RANGE = 3

    points = np.zeros((N_POINTS, N_POINTS, 2), dtype='float32')
    points[:, :, 0] = np.linspace(-RANGE, RANGE, N_POINTS)[:, None]
    points[:, :, 1] = np.linspace(-RANGE, RANGE, N_POINTS)[None, :]
    points = points.reshape((-1, 2))

    points_v = autograd.Variable(torch.Tensor(points), volatile=True).to(device)
    disc_map = netD(points_v).cpu().data.numpy()

    noise = torch.randn(BATCH_SIZE, 2).to(device)
    noisev = autograd.Variable(noise, volatile=True)
    true_dist_v = autograd.Variable(torch.Tensor(true_dist).to(device))
    samples = netG(noisev, true_dist_v).cpu().data.numpy()

    plt.clf()

    x = y = np.linspace(-RANGE, RANGE, N_POINTS)
    plt.contour(x, y, disc_map.reshape((len(x), len(y))).transpose())

    plt.scatter(true_dist[:, 0], true_dist[:, 1], c='orange', marker='+')
    if not FIXED_GENERATOR:
        plt.scatter(samples[:, 0], samples[:, 1], c='green', marker='+')

    plt.savefig('tmp/' + DATASET + '/' + 'frame' + str(frame_index[0]) + '.jpg')


# ==================Model======================
netG = Generator().to(device)
netD = Discriminator().to(device)
netD.apply(weights_init)
netG.apply(weights_init)
print(netG)
print(netD)

optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))

data = inf_train_gen(BATCH_SIZE)

for iteration in range(ITERS):
    ############################
    # (1) Update D network
    ###########################
    for p in netD.parameters():  # reset requires_grad
        p.requires_grad = True  # they are set to False below in netG update

    for iter_d in range(CRITIC_ITERS):
        _data = next(data)
        real_data = torch.Tensor(_data).to(device)
        real_data_v = autograd.Variable(real_data)

        netD.zero_grad()

        # train with real
        D_real = netD(real_data_v)
        D_real = D_real.mean()
        D_real.backward()

        # train with fake
        noise = torch.randn(BATCH_SIZE, 2).to(device)

        noisev = autograd.Variable(noise, volatile=True)  # totally freeze netG
        fake = autograd.Variable(netG(noisev, real_data_v).data)
        inputv = fake
        D_fake = netD(inputv)
        D_fake = D_fake.mean()
        (-D_fake).backward()

        # train with gradient penalty
        gradient_penalty = calc_gradient_penalty(netD, real_data_v.data, fake.data)
        gradient_penalty.backward()

        D_cost = D_fake - D_real + gradient_penalty
        Wasserstein_D = D_real - D_fake
        optimizerD.step()

    if not FIXED_GENERATOR:
        ############################
        # (2) Update G network
        ###########################
        for p in netD.parameters():
            p.requires_grad = False  # to avoid computation
        netG.zero_grad()

        _data = next(data)
        real_data = torch.Tensor(_data).to(device)
        real_data_v = autograd.Variable(real_data)

        noise = torch.randn(BATCH_SIZE, 2).to(device)
        noisev = autograd.Variable(noise)
        fake = netG(noisev, real_data_v)
        G = netD(fake)
        G = G.mean()
        (-G).backward()
        G_cost = -G
        optimizerG.step()

    # Write logs and save samples
    lib.plot.plot('tmp/' + DATASET + '/' + 'disc cost', D_cost.cpu().data.numpy())
    lib.plot.plot('tmp/' + DATASET + '/' + 'wasserstein distance', Wasserstein_D.cpu().data.numpy())
    if not FIXED_GENERATOR:
        lib.plot.plot('tmp/' + DATASET + '/' + 'gen cost', G_cost.cpu().data.numpy())
    if iteration % 100 == 99:
        lib.plot.flush()
        generate_image(_data, frame_index = [0])
    lib.plot.tick()
