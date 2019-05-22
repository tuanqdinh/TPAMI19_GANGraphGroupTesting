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

current_name = "{}-{}-{}-{}".format(name_data, name_gan, name_model, name_ctrl)
print('---- Working on {}'.format(current_name))

print('Linear')
from model.mlp import *

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

data_path = os.path.join(args.data_path, '{}/data_{}_4k.mat'.format(name_data, name_data))
A, L, signals = load_data(data_path, is_control=not(1 == args.off_ctrl))
lap_matrx = torch.tensor(L, dtype=torch.float32).to(device)
adj = torch.tensor(A, dtype=torch.float32).to(device)
# adj = A.to(device).to_dense()

data = torch.tensor(signals, dtype=torch.float32)
mu_data = torch.mean(data, dim=0).to(device)
train = torch.utils.data.TensorDataset(data)
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

def generate_image(netG, frame_index, nsamples, img_size=65):
	noise = torch.randn(nsamples, args.embed_size).to(device)
	with torch.no_grad():
		noisev = autograd.Variable(noise)
		# noisev = autograd.Variable(noise, volatile=True)
	samples = netG(noisev)
	# samples = samples.view(args.batch_size, img_size, img_size)
	samples = samples.cpu().data.numpy()
	np.save(
		'{}/samples_{}_{}.npy'.format(sample_path, current_name, frame_index),
		samples
	)
	x = np.mean(samples, axis=0)
	y = np.mean(signals, axis=0)
	plt.scatter(range(100), x[:100])
	plt.scatter(range(100), y[:100])
	plt.show()
	# from IPython import embed;embed()

# ==================Model======================
netG = Generator(input_size=args.embed_size, output_size=args.signal_size, hidden_size=args.hidden_size, adj=adj).to(device)
netD = Discriminator(input_size=args.signal_size, hidden_size=args.hidden_size, adj=adj).to(device)

netD.apply(weights_init)
netG.apply(weights_init)

optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))

# Loss function
adversarial_loss = torch.nn.BCELoss()
Tensor = torch.cuda.FloatTensor
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
			_data = next(data_iter); iteration += 1
			real_data = _data[0].to(device)

			with torch.no_grad():
				real_data_v = autograd.Variable(real_data)
				valid_label = autograd.Variable(torch.ones(real_data.shape[0])).to(device)
				fake_label = autograd.Variable(torch.zeros(real_data.shape[0])).to(device)

			optimizerG.zero_grad()
			noise = torch.randn(real_data_v.shape[0], args.embed_size).to(device)
			with torch.no_grad():
				noisev = autograd.Variable(noise)  # totally freeze netG
			fake = autograd.Variable(netG(noisev))
			d_fake = netD(fake)
			g_loss = adversarial_loss(d_fake, valid_label)

			# mean
			m = fake.mean(dim=0) - mu_data
			mean_norm = (m * m).sum()
			g_loss += 100000000 * mean_norm
			from IPython import embed; embed()

			g_loss.backward()
			optimizerG.step()

			optimizerD.zero_grad()
			real_loss = adversarial_loss(netD(real_data_v), valid_label)
			fake_loss = adversarial_loss(netD(fake.detach()), fake_label)

			d_loss = (real_loss + fake_loss)/2

			d_loss.backward()
			optimizerD.step()


			# Write logs and save samples
			lib.plot.plot(output_path + '/disc_cost', d_loss.cpu().data.numpy())
			lib.plot.plot(output_path + '/gen_cost', g_loss.cpu().data.numpy())
			# Print log info
			if iteration == total_step:
				log('Epoch [{}/{}], Step [{}/{}], D-cost: {:.4f}, G-cost: {:.4f}'
					  .format(epoch, args.num_epochs, iteration, total_step, d_loss.cpu().data.numpy(), g_loss.cpu().data.numpy()))
			# lib.plot.tick()
	print('save models')
	torch.save(netG.state_dict(), netG_path)
	torch.save(netD.state_dict(), netD_path)

print('Generating samples')
generate_image(netG, frame_index=args.alpha, nsamples=200)
print("Total time: {:4.4f}".format(time.time() - start_time))
