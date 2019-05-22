import os, sys, time
sys.path.append(os.getcwd())
import numpy as np
from time import gmtime, strftime
import sklearn.datasets

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.autograd as autograd
import torch.optim as optim
from torchvision import datasets, transforms

import tflib as lib
import tflib.save_images
import tflib.mnist
import tflib.plot

from util.helper import *
from __init__ import args

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

################## MAIN #######################
start_time = time.time()

torch.manual_seed(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OFFSET_LAPGAN = 1
# Create model directory
### GET Input
'''
print('Arguments!!! data - model - control')
print('----data : 1 ADNI - 2 ADRC - 3 Simuln')
print('----gan: 1 wgan 2 lsgan')
print('----gan: 0 linear 10 conv 100 gcn')
print('----model: 1 lapgan - 2 baseline - 3 real')
print('----control: 1 AD 2 CN')
'''

name_data = offdata2name(args.off_data)
name_model = offmodel2name(args.off_model)
name_ctrl = offctrl2name(args.off_ctrl)
name_gan = offgan2name(args.off_gan)

current_name = "{}-{}-{}".format(name_data, name_gan, name_model)
print('---- Working on {}'.format(current_name))

from model.model_cgan import Generator, Discriminator

output_folder = "{}/{}/{}".format(name_data, name_model, name_ctrl) # final folder
gan_path = "{}/{}".format(args.result_path, name_gan)
output_path = os.path.join(gan_path, output_folder)
model_path = os.path.join(args.result_path, 'saved_models')
sample_path = gan_path
log_path = os.path.join(output_path, 'logs')

netG_path = os.path.join(model_path, 'mesh-netG-{}.pth'.format(current_name))
netD_path = os.path.join(model_path, 'mesh-netD-{}.pth'.format(current_name))

mkdir(output_path)
mkdir(model_path)
mkdir(sample_path)
mkdir(log_path)
logf = open(os.path.join(log_path, "log%s.txt" % strftime("%Y-%m-%d-%H:%M:%S", gmtime())), 'w')

###------------------ Data -----------------------
if args.off_data == 4: # demo
	data_path = os.path.join(args.data_path, '{}/data_{}.npy'.format(name_data, name_data))
	dict = np.load(data_path).item()
	#####
	if args.off_ctrl == 1:
		signals = dict['ad']
	else:
		signals = dict['cn']
	adj = torch.tensor(dict['A'],  dtype=torch.float32).to(device)
	lap_matrx = torch.tensor(dict['L'],  dtype=torch.float32).to(device)
	####
	args.signal_size = signals.shape[1]
	args.hidden_size = signals.shape[1] // 4
	args.embed_size = signals.shape[1] // 20
else:
	data_path = os.path.join(args.data_path, '{}/data_{}_4k.mat'.format(name_data, name_data))
	A, L, ad = load_data(data_path, is_control=0)
	_, _, cn = load_data(data_path, is_control=1)
	labels = np.zeros((ad.shape[0] + cn.shape[0]))
	labels[0:ad.shape[0]] = 1

	signals = np.concatenate([ad, cn], axis=0)
	lap_matrx = torch.tensor(L, dtype=torch.float32).to(device)
	adj = torch.tensor(A, dtype=torch.float32).to(device)
	# adj = A.to(device).to_dense()

# from IPython import embed; embed()

data = torch.tensor(signals, dtype=torch.float32)
labels = torch.tensor(labels).type(torch.LongTensor)
mu_data = torch.mean(data, dim=0).to(device)
train = torch.utils.data.TensorDataset(data, labels)
dataloader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=int(args.num_workers))

################### Helpers #################################
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
	elif classname.find('Conv') != -1:
		torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)

def calc_gradient_penalty(netD, real_data, fake_data, label):
	alpha = torch.rand(real_data.size()[0], 1)
	alpha = alpha.expand(real_data.size())
	alpha = alpha.to(device)

	interpolates = alpha * real_data + ((1 - alpha) * fake_data)
	interpolates = interpolates.to(device)
	interpolates = autograd.Variable(interpolates, requires_grad=True)

	disc_interpolates = netD(interpolates, label)

	gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
							  grad_outputs=torch.ones(disc_interpolates.size()).to(device),
							  create_graph=True, retain_graph=True, only_inputs=True)[0]

	gradient_penalty = args.LAMBDA * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
	return gradient_penalty

def generate_image(netG, frame_index, nsamples, img_size=65):
	noise = torch.randn(labels.shape[0], args.embed_size)
	real_label = torch.zeros(labels.shape[0], 2)
	real_label = real_label.scatter_(1, labels.view(labels.shape[0], 1), 1)
	with torch.no_grad():
		noisev = autograd.Variable(noise).to(device)
		real_label_v = autograd.Variable(real_label).to(device)
	samples = netG(noisev, real_label_v)
	samples = samples.cpu().data.numpy()
	np.save(
		'{}/samples_{}_{}.npy'.format(sample_path, current_name, frame_index),
		samples
	)
	x = np.mean(samples[:150, :], axis=0)
	y = np.mean(signals[:150, :], axis=0)
	plt.scatter(range(100), x[:100])
	plt.scatter(range(100), y[:100])
	plt.show()

# ==================Model======================
netG = Generator(input_size=args.embed_size, label_size = 2, output_size=args.signal_size, hidden_size=args.hidden_size, adj=adj).to(device)
# netD = Discriminator(input_size=args.batch_size, hidden_size=args.hidden_size * 2, adj=adj).to(device)
netD = Discriminator(input_size=args.signal_size, label_size = 2, output_size=1, hidden_size=args.hidden_size, adj=adj).to(device)

netD.apply(weights_init)
netG.apply(weights_init)

optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))

#####--------------Training----------------------------
if os.path.isfile(netG_path) and False:
	print('Load existing models')
	netG.load_state_dict(torch.load(netG_path))
	netD.load_state_dict(torch.load(netD_path))
else:
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
				_data, _label = next(data_iter); iteration += 1
				real_label = torch.zeros(_label.shape[0], 2)
				real_label = real_label.scatter_(1, _label.view(_label.shape[0], 1), 1)
				# from IPython import embed; embed()

				real_data = _data.to(device)
				real_label = real_label.to(device)
				with torch.no_grad():
					real_data_v = autograd.Variable(real_data)
					real_label_v = autograd.Variable(real_label)

				optimizerD.zero_grad()

				D_real = netD(real_data_v, real_label_v)
				D_real = D_real.mean()
				# train with fake
				noise = torch.randn(real_data_v.size()[0], args.embed_size).to(device)
				with torch.no_grad():
					noisev = autograd.Variable(noise)  # totally freeze netG
				fake = autograd.Variable(netG(noisev, real_label_v).data)
				#### This will prevent gradient on params of G
				inputv = fake
				D_fake = netD(inputv, real_label_v)
				D_fake = D_fake.mean()

				# train with gradient penalty
				gradient_penalty = calc_gradient_penalty(netD, real_data_v.data, fake.data, real_label_v.data)

				D_cost = D_fake - D_real + gradient_penalty
				Wasserstein_D = D_real - D_fake

				D_cost.backward()
				optimizerD.step()

			############################
			# (2) Update G network
			###########################
			for p in netD.parameters():
				p.requires_grad = False  # to avoid computation

			optimizerG.zero_grad()

			_data, _label = next(data_iter); iteration += 1
			real_label = torch.zeros(_label.shape[0], 2)
			real_label = real_label.scatter_(1, _label.view(_label.shape[0], 1), 1)

			real_data = _data.to(device)
			real_label = real_label.to(device)
			with torch.no_grad():
				real_data_v = autograd.Variable(real_data)
				real_label_v = autograd.Variable(real_label)

			noise = torch.randn(real_label_v.shape[0], args.embed_size).to(device)
			with torch.no_grad():
				noisev = autograd.Variable(noise)
			fake = netG(noisev, real_label_v)
			D_fake = netD(fake, real_label_v)

			G_cost = -D_fake.mean()
			if OFFSET_LAPGAN == args.off_model: # laplacian
				xl = torch.mm(fake, lap_matrx)
				xlx = torch.mm(xl, fake.t())
				yl = torch.mm(real_data_v, lap_matrx)
				yly = torch.mm(yl, real_data_v.t())
				reg = (xlx.mean().sqrt() - yly.mean().sqrt())**2
				G_cost = G_cost + args.alpha * reg

			m = real_data_v.mean(dim=0) - mu_data
			mean_norm = torch.sqrt((m * m).mean())
			G_cost += 10 * mean_norm

			G_cost.backward()
			optimizerG.step()

			# Write logs and save samples
			lib.plot.plot(output_path + '/gradient_penalty', gradient_penalty.cpu().data.numpy())
			lib.plot.plot(output_path + '/disc_cost', D_cost.cpu().data.numpy())
			lib.plot.plot(output_path + '/wasserstein_distance', Wasserstein_D.cpu().data.numpy())
			lib.plot.plot(output_path + '/gen_cost', G_cost.cpu().data.numpy())
			# Print log info
			if iteration == total_step:
				log('Epoch [{}/{}], Step [{}/{}], D-cost: {:.4f}, G-cost: {:.4f}'
					  .format(epoch, args.num_epochs, iteration, total_step, D_cost.cpu().data.numpy(), G_cost.cpu().data.numpy()))
	print('save models')
	torch.save(netG.state_dict(), netG_path)
	torch.save(netD.state_dict(), netD_path)

print('Generating samples')
generate_image(netG, frame_index=args.alpha, nsamples=200)
print("Total time: {:4.4f}".format(time.time() - start_time))
