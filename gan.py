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

from model.model_lsgan import *
from util.helper import *
from __init__ import args

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

################## MAIN #######################

torch.manual_seed(1)
cuda = True if torch.cuda.is_available() else False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OFFSET_LAPGAN = 1
# Create model directory
### GET Input

print('Arguments!!! data - model - control')
print('----data : 1 ADNI - 2 ADRC - 3 Simuln')
print('----model: 1 lapgan - 2 baseline - 3 real')
print('----control: 1 AD 2 CN')

name_data = offdata2name(args.off_data)
name_model = offmodel2name(args.off_model)
name_ctrl = offctrl2name(args.off_ctrl)
print('---- Working on {} data - {} model - {}'.format(name_data, name_model, name_ctrl))

output_folder = "{}/{}/{}".format(name_data, name_model, name_ctrl) # final folder
gan_path = "{}/{}".format(args.result_path, args.gan)
output_path = os.path.join(gan_path, output_folder)
model_path = os.path.join(output_path, 'snapshot')
sample_path = os.path.join(output_path, 'samples')
log_path = os.path.join(output_path, 'logs')
mkdir(output_path)
mkdir(model_path)
mkdir(sample_path)
mkdir(log_path)
logf = open(os.path.join(log_path, "log%s.txt" % strftime("%Y-%m-%d-%H:%M:%S", gmtime())), 'w')

###------------------ Data -----------------------
data_path = os.path.join(args.data_path, '{}/data_{}_4k.mat'.format(name_data, name_data))
if 1 == args.off_ctrl:
	L, signals = load_data(data_path, is_control=False)
else:
	L, signals = load_data(data_path, is_control=True)
lap_matrx = torch.tensor(L, dtype=torch.float32).to(device)
data = torch.tensor(signals, dtype=torch.float32)
train = torch.utils.data.TensorDataset(data)
dataloader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=int(args.num_workers))

################### Helpers #################################
# custom weights initialization called on generator and discriminator
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

def calc_gradient_penalty(discriminator, real_data, fake_data):
	alpha = torch.rand(real_data.size()[0], 1)
	alpha = alpha.expand(real_data.size())
	alpha = alpha.to(device)

	interpolates = alpha * real_data + ((1 - alpha) * fake_data)
	interpolates = interpolates.to(device)
	interpolates = autograd.Variable(interpolates, requires_grad=True)

	disc_interpolates = discriminator(interpolates)

	gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
							  grad_outputs=torch.ones(disc_interpolates.size()).to(device),
							  create_graph=True, retain_graph=True, only_inputs=True)[0]

	gradient_penalty = args.LAMBDA * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
	return gradient_penalty

def generate_image(generator, frame_index, nsamples, img_size=65):
	noise = torch.randn(nsamples, args.embed_size).to(device)
	with torch.no_grad():
		noisev = autograd.Variable(noise)
		# noisev = autograd.Variable(noise, volatile=True)
	samples = generator(noisev)
	# samples = samples.view(args.batch_size, img_size, img_size)
	samples = samples.cpu().data.numpy()
	np.save(
		'{}/samples_{}.npy'.format(sample_path, frame_index),
		samples
	)
	# lib.save_images.save_images(
	# 		samples,
	# 		'{}/samples_{}.png'.format(sample_path, frame_index),
	# 	)

# ==================Model======================
generator = Generator(input_size=args.embed_size, output_size=args.signal_size, hidden_size=args.hidden_size).to(device)
discriminator = Discriminator(input_size=args.signal_size, hidden_size=args.hidden_size).to(device)
discriminator.apply(weights_init)
generator.apply(weights_init)

# !!! Minimizes MSE instead of BCE
adversarial_loss = torch.nn.MSELoss().to(device)

print(generator)
print(discriminator)

optimizer_D = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))
optimizer_G = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.999))

#####--------------Training----------------------------
generator_path = os.path.join(model_path, 'mesh-generator-99.pth')
discriminator_path = os.path.join(model_path, 'mesh-discriminator-99.pth')

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
if os.path.isfile(generator_path):
	print('Load existing models')
	generator.load_state_dict(torch.load(generator_path))
	discriminator.load_state_dict(torch.load(discriminator_path))
else:
	for epoch in range(args.num_epochs):
		data_iter = iter(dataloader)
		total_step = len(data_iter)
		iteration = 0
		while iteration < total_step:
			_data = next(data_iter)[0]
			iteration += 1
			############################
			# (1) Update D network
			###########################
			real_data = _data.to(device)
			real_imgs = autograd.Variable(real_data)
			valid = autograd.Variable(Tensor(real_data.shape[0], 1).fill_(1.0), requires_grad=False)
			fake = autograd.Variable(Tensor(real_data.shape[0], 1).fill_(0.0), requires_grad=False)

			optimizer_G.zero_grad()
			noise = torch.randn(real_data.shape[0], args.embed_size).to(device)
			noisev = autograd.Variable(noise)
			gen_imgs = generator(noisev)
			g_loss = adversarial_loss(discriminator(gen_imgs), valid)
			if OFFSET_LAPGAN == args.off_model: # laplacian
				xl = torch.mm(gen_imgs, lap_matrx)
				xlx = torch.mm(xl, gen_imgs.t())
				yl = torch.mm(real_imgs, lap_matrx)
				yly = torch.mm(yl, real_imgs.t())
				reg = (xlx.mean().sqrt() - yly.mean().sqrt())**2
				g_loss = g_loss + args.alpha * reg

			g_loss.backward()
			optimizer_G.step()

			optimizer_D.zero_grad()
			real_loss = adversarial_loss(discriminator(real_imgs), valid)
			fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
			d_loss = 0.5 * (real_loss + fake_loss)
			d_loss.backward()
			optimizer_D.step()

			log('Epoch [{}/{}], Step [{}/{}], D-cost: {:.4f}, G-cost: {:.4f}'
				  .format(epoch, args.num_epochs, iteration, total_step, d_loss.cpu().data.numpy(), g_loss.cpu().data.numpy()))

			lib.plot.plot(output_path + '/disc_cost', d_loss.cpu().data.numpy())
			lib.plot.plot(output_path + '/gen_cost', g_loss.cpu().data.numpy())
			# Print log info
			if iteration == total_step:
				lib.plot.flush()
			lib.plot.tick()
	print('save models')
	torch.save(generator.state_dict(), os.path.join(model_path, 'mesh-generator-{}.pth'.format(epoch)))
	torch.save(discriminator.state_dict(), os.path.join(model_path, 'mesh-discriminator-{}.pth'.format(epoch)))
print('Generating samples')
generate_image(generator, frame_index=1000, nsamples=1000)
