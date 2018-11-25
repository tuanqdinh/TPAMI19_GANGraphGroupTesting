import os, sys
sys.path.append(os.getcwd())

import numpy as np
from time import gmtime, strftime
import sklearn.datasets
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.autograd as autograd
import torch.optim as optim
from torchvision import datasets, transforms

import tflib as lib
import tflib.save_images
import tflib.mnist
import tflib.plot
from models.mesh import *
from utils.data_helpers import inf_train_gen, load_data
from utils.helpers import *
from __init__ import args
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

torch.manual_seed(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create model directory
if args.laplacian:
	print('--- Laplacian --> alpha: {}'.format(args.alpha))
	folder = 'laplacian'
else:
	folder = 'normal'
args.output_path = "{}/{}".format(args.output_path, folder)
model_path = os.path.join(args.output_path, 'snapshot')
sample_path = os.path.join(args.output_path, 'samples')
log_path = os.path.join(args.output_path, 'logs')
mkdir(args.output_path)
mkdir(model_path)
mkdir(sample_path)
mkdir(log_path)
logf = open(os.path.join(log_path, "log%s.txt" % strftime("%Y-%m-%d-%H:%M:%S", gmtime())), 'w')

# ==================Model======================
# custom weights initialization called on netG and netD
def log(msg, console_print=True):
	logf.write(msg + '\n')
	if console_print:
		print(msg)

def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Linear') != -1:
		m.weight.data.normal_(0.0, 0.02)
		m.bias.data.fill_(0)
	elif classname.find('BatchNorm') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)

def calc_gradient_penalty(netD, real_data, fake_data):
	alpha = torch.rand(real_data.size()[0], 1)
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

def generate_image(netG, frame_index, img_size=65):
	noise = torch.randn(args.batch_size, args.embed_size).to(device)
	noisev = autograd.Variable(noise, volatile=True)
	samples = netG(noisev)
	samples = samples.view(args.batch_size, img_size, img_size)
	samples = samples.cpu().data.numpy()
	np.save(
		'{}/samples_{}.npy'.format(sample_path, frame_index),
		samples
	)
	# lib.save_images.save_images(
	# 		samples,
	# 		'{}/samples_{}.png'.format(sample_path, frame_index),
	# 	)

netG = Generator(input_size=args.embed_size, output_size=args.signal_size, hidden_size=args.hidden_size).to(device)
netD = Discriminator(input_size=args.signal_size, hidden_size=args.hidden_size).to(device)
netD.apply(weights_init)
netG.apply(weights_init)
print(netG)
print(netD)

optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))

L, signals = load_data(args.data_path, is_control=False)
lap_matrx = torch.tensor(L, dtype=torch.float32).to(device)
data = torch.tensor(signals, dtype=torch.float32)
train = torch.utils.data.TensorDataset(data)
dataloader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=int(args.num_workers))
# dataloader = torch.utils.data.DataLoader(
# 		datasets.MNIST('data/mnist', train=True, download=True,
# 					   transform=transforms.Compose([
# 						   transforms.ToTensor(),
# 						   transforms.Normalize((0.1307,), (0.3081,))
# 					   ])),
# 						batch_size=args.batch_size, shuffle=True, num_workers=int(args.num_workers))
for epoch in range(args.num_epochs):
	data_iter = iter(dataloader)
	total_step = len(data_iter)
	iteration = 0
	while iteration < total_step:
		############################
		# (1) Update D network
		###########################
		for p in netD.parameters():  # reset requires_grad
			p.requires_grad = True  # they are set to False below in netG update

		iter_d = 0
		while iter_d < args.critic_iters and iteration < total_step - 1:
			iter_d += 1
			_data = next(data_iter); iteration += 1

			real_data = _data[0].to(device)
			# real_data = real_data.view(-1, args.signal_size)
			real_data_v = autograd.Variable(real_data)
			netD.zero_grad()

			# train with real
			# from IPython import embed; embed()
			D_real = netD(real_data_v)
			D_real = D_real.mean()
			# (-D_real).backward()

			# train with fake
			noise = torch.randn(real_data_v.size()[0], args.embed_size).to(device)
			noisev = autograd.Variable(noise, volatile=True)  # totally freeze netG
			fake = autograd.Variable(netG(noisev).data)
			inputv = fake
			D_fake = netD(inputv)
			D_fake = D_fake.mean()
			# D_fake.backward()

			# train with gradient penalty
			gradient_penalty = calc_gradient_penalty(netD, real_data_v.data, fake.data)
			# gradient_penalty.backward()

			D_cost = D_fake - D_real + gradient_penalty
			Wasserstein_D = D_real - D_fake

			D_cost.backward()
			optimizerD.step()

		############################
		# (2) Update G network
		###########################
		for p in netD.parameters():
			p.requires_grad = False  # to avoid computation
		netG.zero_grad()

		_data = next(data_iter); iteration += 1
		real_data = _data[0].to(device)
		# real_data = real_data.view(-1, args.signal_size)
		real_data_v = autograd.Variable(real_data)

		noise = torch.randn(args.batch_size, args.embed_size).to(device)
		noisev = autograd.Variable(noise)
		fake = netG(noisev)
		D_fake = netD(fake)
		G_cost = -D_fake.mean()

		if args.laplacian:
			xl = torch.mm(fake, lap_matrx)
			xlx = torch.mm(xl, fake.t())
			yl = torch.mm(real_data_v, lap_matrx)
			yly = torch.mm(yl, real_data_v.t())
			reg = (xlx.mean().sqrt() - yly.mean().sqrt())**2
			G_cost = G_cost + args.alpha * reg

		G_cost.backward()
		optimizerG.step()

		# Write logs and save samples
		lib.plot.plot(args.output_path + '/gradient_penalty', gradient_penalty.cpu().data.numpy())
		lib.plot.plot(args.output_path + '/disc_cost', D_cost.cpu().data.numpy())
		lib.plot.plot(args.output_path + '/wasserstein_distance', Wasserstein_D.cpu().data.numpy())
		lib.plot.plot(args.output_path + '/gen_cost', G_cost.cpu().data.numpy())
		# Print log info
		if iteration == total_step:
			lib.plot.flush()
			generate_image(netG, frame_index=epoch, img_size=args.img_size)
			log('Epoch [{}/{}], Step [{}/{}], D-cost: {:.4f}, G-cost: {:.4f}, W-distance: {:.4f}'
				  .format(epoch, args.num_epochs, iteration, total_step, D_cost.cpu().data.numpy(), G_cost.cpu().data.numpy(), Wasserstein_D.cpu().data.numpy()))
		lib.plot.tick()
		if iteration == total_step:
			torch.save(netG.state_dict(), os.path.join(model_path, 'mesh-netG-{}-{}.pth'.format(epoch + 1, iteration + 1)))
			torch.save(netD.state_dict(), os.path.join(model_path, 'mesh-netD-{}-{}.pth'.format(epoch + 1, iteration + 1)))
