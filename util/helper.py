import os, shutil
import numpy as np

import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import matplotlib.gridspec as gridspec

import scipy
import scipy.io as spio
from scipy.sparse import csgraph
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE


# ls -1 | wc -l

############## Dataset ########
# Dataset iterator
# train_gen, dev_gen, test_gen = lib.mnist.load(args.batch_size, args.batch_size)
# def inf_train_gen2():
#     while True:
#         for images,targets in train_gen():
#             yield images

# Dataset iterator
def inf_train_gen(data, batch_size):
	ds_size = data.shape[0]
	while True:
		for i in range(ds_size // batch_size):
			start = i * batch_size
			end = (i + 1) * batch_size
			yield data[start:end, :]


def load_data(data_path, is_control=False):
	mat = spio.loadmat(data_path, squeeze_me=True)
	A = mat['A']
	L = csgraph.laplacian(A, normed=False)
	if is_control:
		signals = np.asarray(mat['cn_signals'])
	else:
		signals = np.asarray(mat['ad_signals'])
	signals = normalize(signals, axis=1, norm='l2')
	return L.todense(), signals

def offdata2name(off_data):
	if off_data == 1:
		return 'adni'
	elif off_data == 2:
		return 'adrc'
	else:
		return 'simuln'

def offmodel2name(off_model):
	if off_model == 3:
		return 'real'
	elif off_model == 1:
		return 'lapgan'
	elif off_model == 2:
		return 'baseline'

def offctrl2name(off_ctrl):
	if off_ctrl == 1:
		return 'ad'
	elif off_ctrl == 2:
		return 'cn'

################### Ops ##################
def sample_z(m, n):
	return np.random.normal(size=[m, n], loc = 0, scale = 1)
	# return np.random.uniform(-1., 1., size=[m, n])

def mkdir(name, rm=False):
	if not os.path.exists(name):
		os.makedirs(name)
	elif rm:
		shutil.rmtree(name)
		os.mkdir(name)

################### Plotting #############################
def save_images(samples, im_size, path, idx, n_fig_unit=2):
	fig = plt.figure(figsize=(4, 4))
	gs = gridspec.GridSpec(n_fig_unit, n_fig_unit)
	gs.update(wspace=0.05, hspace=0.05)

	for i, sample in enumerate(samples):
		ax = plt.subplot(gs[i])
		plt.axis('off')
		ax.set_xticklabels([])
		ax.set_yticklabels([])
		ax.set_aspect('equal')
		plt.imshow(sample.reshape(im_size, im_size), cmap='Greys_r')

	plt.savefig(path + '/{}.png'.format(str(idx).zfill(3)),
				bbox__hnches='tight')
	plt.close(fig)
	return fig

def plot_hist_1(data, deg_vec, path, idx):
	x = data / deg_vec
	fig = plt.figure(figsize=(4, 4))
	plt.scatter(np.arange(len(x)), x)
	plt.savefig(path + '/{}.png'.format(str(idx).zfill(3)),
				bbox__hnches='tight')
	plt.close(fig)


def plot_hist_2(data, deg_vec):
	fig = plt.gcf()
	fig.show()
	fig.canvas.draw()
	plt.title("Gaussian Histogram")
	plt.xlabel("Value")
	plt.ylabel("Frequency")
	for node in range(len(deg_vec)):
		try:
			x = data[:, node] / deg_vec[node]
			mu = np.mean(x)
			sig = np.std(x)
			print('Node {:d}: mean:{:.3f}, std: {:.3f}'.format(node, mu, sig))
			plt.hist(x, 20) # Hard-code
			fig.canvas.draw()
			input('Press to continue ...')
		except:
			# from IPython import embed; embed() #os._exit(1)
			print('Exception')
			break

def plot_fig(lmse):
	x = np.arange(len(lmse))
	plt.figure()
	plt.plot(x, results[0], c='r')
	plt.plot(x, results[1], c='b')
	plt.plot(x, results[-1], c='g')

def plot_tnse(fname):
	data = np.load(fname)
	tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
	result = tsne.fit_transform(data)
	vis_x = result[:, 0]
	vis_y = result[:, 1]
	plt.scatter(vis_x, vis_y, cmap=plt.cm.get_cmap("jet", 10))
	plt.title('Laplacian')
	plt.show()

########
def plot_dist(dists, n_pixels, name):
	# Plot the distributions
	# from IPython import embed; embed()
	means = dists[:, 0]
	sds = dists[:, 1]
	hpd_025 = dists[:, 2]
	hpd_975 = dists[:, 3]

	fig = plt.figure(figsize=(20, 7))
	plt.errorbar(range(n_pixels), means, yerr=sds, fmt='-o', linestyle="None")
	plt.scatter(range(n_pixels), hpd_025, c='b')
	plt.scatter(range(n_pixels), hpd_975, c='b')
	fig.tight_layout()
	fig.savefig(name, bbox_inches='tight')
	plt.xlabel('Vertex')
	plt.title('Difference of means distribution')
	plt.show()


def plot_eb(off_data, epsilon = 0.01):
	root_path = '../result'
	fig = plt.figure(figsize=(10, 7))
	name_data = offdata2name(off_data)
	plot_path = "{}/eb/lsgan/plt_{}_{}.png".format(root_path, name_data, epsilon)

	for off_model in range(1, 4):
		name_model = offmodel2name(off_model)
		if off_model == 3:
			file_path = "{}/eb/eval_{}_{}_{}.npy".format(root_path, name_data, name_model, epsilon)
		else:
			file_path = "{}/eb/lsgan/eval_{}_{}_{}.npy".format(root_path, name_data, name_model, epsilon)
		arr = np.load(file_path)
		plt.plot(range(len(arr)), sorted(arr), label=name_model)
	plt.legend()
	plt.xlabel('Node')
	plt.ylabel("e-value")
	plt.title('{} data - e-value with epsilon = {}'.format(name_data, epsilon))
	plt.show()
	fig.tight_layout()
	fig.savefig(plot_path, bbox_inches='tight')
