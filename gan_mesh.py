import os, sys
sys.path.append(os.getcwd())

import numpy as np
import sklearn.datasets
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.autograd as autograd
import torch.optim as optim

import tflib as lib
import tflib.plot
from models.mesh import *
from utils.data_helpers import inf_train_gen
from __init__ import *

torch.manual_seed(1)
# Create model directory
model_path = os.path.join(args.output_path, 'snapshot')

logf = open(os.path.join(args.output_path, "log%s.txt" % strftime("%Y-%m-%d-%H:%M:%S", gmtime())), 'w')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================Model======================
netG = Generator().to(device)
netD = Discriminator().to(device)
netD.apply(weights_init)
netG.apply(weights_init)
print(netG)
print(netD)

optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))

data = inf_train_gen(args.BATCH_SIZE)

for iteration in range(args.ITERS):
    ############################
    # (1) Update D network
    ###########################
    for p in netD.parameters():  # reset requires_grad
        p.requires_grad = True  # they are set to False below in netG update

    for iter_d in range(args.CRITIC_ITERS):
        _data = next(data)
        real_data = torch.Tensor(_data).to(device)
        real_data_v = autograd.Variable(real_data)

        netD.zero_grad()

        # train with real
        D_real = netD(real_data_v)
        D_real = D_real.mean()
        D_real.backward()

        # train with fake
        noise = torch.randn(args.BATCH_SIZE, 2).to(device)

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
    lib.plot.plot('tmp/' + DATASET + '/' + 'gen cost', G_cost.cpu().data.numpy())
    # Print log info
    if iteration % args.log_step == 0:
        lib.plot.flush()
        generate_image(_data, frame_index = [0])
        log('Epoch [{}/{}], Step [{}/{}], MineScore: {:.4f}'
              .format(epoch, args.num_epochs, i, total_step, mine_score.item()))
        # Save the model checkpoints
    lib.plot.tick()

    if (iteration + 1) % args.save_step == 0:
        torch.save(model.state_dict(), os.path.join(model_path, 'coco-{}-{}.pth'.format(epoch + 1, i + 1)))
