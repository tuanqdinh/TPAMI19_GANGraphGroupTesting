import tensorflow as tf
import numpy as np
import os, shutil
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.manifold import TSNE


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
