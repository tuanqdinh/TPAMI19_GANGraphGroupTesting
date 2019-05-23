import os, sys, time
sys.path.append(os.getcwd())
import numpy as np
from time import gmtime, strftime
import sklearn.datasets

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.autograd as autograd
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F

import tflib as lib
import tflib.save_images
import tflib.mnist
import tflib.plot

import model.mmd_utils as util
from model.mmd_kernels import mix_rbf_mmd2
from model.mmd_gan import Encoder, Decoder, weights_init, grad_norm
from util.helper import *
from __init__ import args

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#CUDA_VISIBLE_DEVICES=1,2
################## MAIN #######################
class NetD(nn.Module):
	def __init__(self, encoder, decoder):
		super(NetD, self).__init__()
		self.encoder = encoder
		self.decoder = decoder

	def forward(self, input):
		f_enc_X = self.encoder(input)
		f_dec_X = self.decoder(f_enc_X)

		f_enc_X = f_enc_X.view(input.size(0), -1)
		f_dec_X = f_dec_X.view(input.size(0), -1)
		return f_enc_X, f_dec_X

class NetG(nn.Module):
	def __init__(self, decoder):
		super(NetG, self).__init__()
		self.decoder = decoder

	def forward(self, input):
		output = self.decoder(input)
		return output


class ONE_SIDED(nn.Module):
	def __init__(self):
		super(ONE_SIDED, self).__init__()

		main = nn.ReLU()
		self.main = main

	def forward(self, input):
		output = self.main(-input)
		output = -output.mean()
		return output

############################

start_time = time.time()

torch.manual_seed(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OFFSET_LAPGAN = 1
# Create model directory
### GET Input
'''
print('Arguments!!! data - model - control')
print('----data : 1 ADNI - 2 ADRC - 3 Simuln')
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

if args.off_gan < 10:
	print('Linear')
	# from model.mesh import *
elif args.off_gan < 100:
	print('CONV')
	from model.model_lsgan import *
else:
	print('GCN')
	from model.gcn import Generator, Discriminator

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
data_path = os.path.join(args.data_path, '{}/data_{}_4k.mat'.format(name_data, name_data))
A, L, signals = load_data(data_path, is_control=not(1 == args.off_ctrl))
lap_matrx = torch.tensor(L, dtype=torch.float32).to(device)
adj = A.to(device).to_dense()

data = torch.tensor(signals, dtype=torch.float32)
data = torch.exp(data) # exp -0.1 to 0.1
mu_data = torch.mean(data, dim=0).to(device)
mu2_data = torch.mean(data * data, dim=0).to(device)

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

# sigma for MMD
base = 1.0
sigma_list = [1, 2, 4, 8, 16]
sigma_list = [sigma / base for sigma in sigma_list]

lambda_MMD = 1.0
lambda_AE_X = 8.0
lambda_AE_Y = 8.0
lambda_rg = 16.0

# ==================Model======================
# netG = Generator(input_size=args.embed_size, output_size=args.signal_size, hidden_size=args.hidden_size).to(device)
# netD = Discriminator(input_size=args.signal_size, hidden_size=args.hidden_size).to(device)
# netD = Discriminator(input_size=args.batch_size, hidden_size=args.batch_size * 2, adj=adj).to(device)

G_decoder = Decoder(embed_dim=args.embed_size, output_dim=args.signal_size)
D_encoder = Encoder(input_dim=args.signal_size, embed_dim=args.embed_size)
D_decoder = Decoder(embed_dim=args.embed_size, output_dim=args.signal_size)

netG = NetG(G_decoder).to(device)
netD = NetD(D_encoder, D_decoder).to(device)
one_sided = ONE_SIDED().to(device)

netD.apply(weights_init)
netG.apply(weights_init)
one_sided.apply(weights_init)

optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))

#####--------------Training----------------------------
if os.path.isfile(netG_path) and False:
	print('Load existing models')
	netG.load_state_dict(torch.load(netG_path))
	netD.load_state_dict(torch.load(netD_path))
else:
	for p in netG.parameters():
		p.requires_grad = True
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

				for p in netD.encoder.parameters():
					p.data.clamp_(-0.01, 0.01)

				_data = next(data_iter); iteration += 1

				real_data = _data[0].to(device)
				real = autograd.Variable(real_data)

				netD.zero_grad()

				# real data
				f_enc_real, f_dec_real = netD(real)

				# train with fake
				noise = torch.randn(real.size()[0], args.embed_size).to(device)
				with torch.no_grad():
					noisev = autograd.Variable(noise)  # totally freeze netG
				fake = autograd.Variable(netG(noisev).data)
				f_enc_fake, f_dec_fake = netD(fake)

				# train with gradient penalty
				# gradient_penalty = calc_gradient_penalty(netD, real.data, fake.data)

				### MMD
				# compute biased MMD2 and use ReLU to prevent negative value
				mmd2_D = mix_rbf_mmd2(f_enc_real, f_enc_fake, sigma_list)
				mmd2_D = F.relu(mmd2_D)

				one_side_errD = one_sided(f_enc_real.mean(0) - f_enc_fake.mean(0))

				# compute L2-loss of AE
				L2_AE_X_D = util.match(real.view(args.batch_size, -1), f_dec_real, 'L2')
				L2_AE_Y_D = util.match(fake.view(args.batch_size, -1), f_dec_fake, 'L2')

				D_cost = torch.sqrt(mmd2_D) + lambda_rg * one_side_errD - lambda_AE_X * L2_AE_X_D - lambda_AE_Y * L2_AE_Y_D
				(-D_cost).backward()
				optimizerD.step()

			############################
			# (2) Update G network
			###########################
			for p in netD.parameters():
				p.requires_grad = False  # to avoid computation

			_data = next(data_iter); iteration += 1

			netG.zero_grad()

			real_data = _data[0].to(device)
			real = autograd.Variable(real_data)

			# real data
			f_enc_real_G, f_dec_real_G = netD(real)

			noise = torch.randn(real.size()[0], args.embed_size).to(device)
			# with torch.no_grad():
			noisev = autograd.Variable(noise)  # totally freeze netG
			fake = netG(noisev)
			f_enc_fake_G, f_dec_fake_G = netD(fake)

			### MMD
			# compute biased MMD2 and use ReLU to prevent negative value
			mmd2_G = mix_rbf_mmd2(f_enc_real_G, f_enc_fake_G, sigma_list)
			mmd2_G = F.relu(mmd2_G)

			one_side_errG = one_sided(f_enc_real_G.mean(0) - f_enc_fake_G.mean(0))

			G_cost = torch.sqrt(mmd2_G) + lambda_rg * one_side_errG

			if OFFSET_LAPGAN == args.off_model: # laplacian
				x = fake.mean(dim=0)
				y = real.mean(dim=0)
				diff = x - y
				reg = torch.dot(diff, torch.mv(lap_matrx, diff))
				G_cost = G_cost + 0.1 * reg / lap_matrx.shape[0]

			G_cost.backward()
			optimizerG.step()

			# mean Dim=0 or 1
			m = fake.mean(dim=0) - mu_data
			mean_norm = torch.sqrt((m * m).sum())
			fake2 = fake * fake
			m2 = fake2.mean(dim=0) - mu2_data
			m2_norm = torch.sqrt((m2*m2).sum())

			# Write logs and save samples
			# lib.plot.plot(output_path + '/gradient_penalty', gradient_penalty.cpu().data.numpy())
			lib.plot.plot(output_path + '/disc_cost', D_cost.cpu().data.numpy())
			#lib.plot.plot(output_path + '/wasserstein_distance', Wasserstein_D.cpu().data.numpy())
			lib.plot.plot(output_path + '/gen_cost', G_cost.cpu().data.numpy())
			# Print log info
			# if iteration == total_step:
				# lib.plot.flush()
				# generate_image(netG, frame_index=epoch, img_size=args.img_size, nsamples=args.batch_size)
			log('Epoch [{}/{}], Step [{}/{}], D-cost: {:.4f}, G-cost: {:.4f}, mean-norm: {:.4f}, mean2-norm: {:.4f}'
					  .format(epoch, args.num_epochs, iteration, total_step, D_cost.cpu().data.numpy(), G_cost.cpu().data.numpy(), mean_norm.cpu().data.numpy(), m2_norm.cpu().data.numpy()))
	# lib.plot.tick()
	print('save models')
	torch.save(netG.state_dict(), netG_path)
	torch.save(netD.state_dict(), netD_path)

print('Generating samples')
generate_image(netG, frame_index=args.alpha, nsamples=200)
print("Total time: {:4.4f}".format(time.time() - start_time))
