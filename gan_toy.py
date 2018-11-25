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
from __init__ import args
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

torch.manual_seed(1)
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
    alpha = torch.rand(args.batch_size, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.to(device)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = args.LAMBDA * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
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

    noise = torch.randn(args.batch_size, 2).to(device)
    noisev = autograd.Variable(noise, volatile=True)
    true_dist_v = autograd.Variable(torch.Tensor(true_dist).to(device))
    samples = netG(noisev, true_dist_v).cpu().data.numpy()

    plt.clf()

    x = y = np.linspace(-RANGE, RANGE, N_POINTS)
    plt.contour(x, y, disc_map.reshape((len(x), len(y))).transpose())

    plt.scatter(true_dist[:, 0], true_dist[:, 1], c='orange', marker='+')
    plt.scatter(samples[:, 0], samples[:, 1], c='green', marker='+')

    plt.savefig('tmp/' + args.dataset + '/' + 'frame' + str(frame_index[0]) + '.jpg')



# Create model directory
model_path = os.path.join(args.output_path, 'snapshot')

# logf = open(os.path.join(args.output_path, "log%s.txt" % strftime("%Y-%m-%d-%H:%M:%S", gmtime())), 'w')


# ==================Model======================
netG = Generator(input_size=args.embed_size, output_size=args.signal_size, hidden_size=args.hidden_size).to(device)
netD = Discriminator(input_size=args.signal_size, hidden_size=args.hidden_size).to(device)
netD.apply(weights_init)
netG.apply(weights_init)
print(netG)
print(netD)

optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))

data = inf_train_gen(args.batch_size)

for iteration in range(args.num_epochs):
    ############################
    # (1) Update D network
    ###########################
    for p in netD.parameters():  # reset requires_grad
        p.requires_grad = True  # they are set to False below in netG update

    for iter_d in range(args.critic_iters):
        _data = next(data)
        real_data = torch.Tensor(_data).to(device)
        real_data_v = autograd.Variable(real_data)

        netD.zero_grad()

        # train with real
        # from IPython import embed; embed()

        D_real = netD(real_data_v)
        D_real = D_real.mean()
        D_real.backward()

        # train with fake
        noise = torch.randn(args.batch_size, 2).to(device)

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

    noise = torch.randn(args.batch_size, 2).to(device)
    noisev = autograd.Variable(noise)
    fake = netG(noisev, real_data_v)
    G = netD(fake)
    G = G.mean()
    (-G).backward()
    G_cost = -G
    optimizerG.step()

    # Write logs and save samples
    lib.plot.plot('tmp/' + args.dataset + '/' + 'disc cost', D_cost.cpu().data.numpy())
    lib.plot.plot('tmp/' + args.dataset + '/' + 'wasserstein distance', Wasserstein_D.cpu().data.numpy())
    lib.plot.plot('tmp/' + args.dataset + '/' + 'gen cost', G_cost.cpu().data.numpy())
    # Print log info
    if iteration % args.log_step == 0:
        lib.plot.flush()
        generate_image(_data, frame_index = [0])
        # log('Epoch [{}/{}], Step [{}/{}], MineScore: {:.4f}'
              # .format(epoch, args.num_epochs, i, total_step, mine_score.item()))
        # Save the model checkpoints
    lib.plot.tick()

    # if (iteration + 1) % args.save_step == 0:
        # torch.save(model.state_dict(), os.path.join(model_path, 'coco-{}-{}.pth'.format(epoch + 1, i + 1)))
