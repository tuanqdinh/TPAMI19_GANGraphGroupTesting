import os, shutil
import numpy as np

import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

import scipy
import scipy.io as spio
from scipy.sparse import csgraph
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
import torch
import scipy.sparse as sp

plt.style.use('ggplot')

image_folder = '../output'

SMALL_SIZE = 8.5
MEDIUM_SIZE = 14
BIGGER_SIZE = 20

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE, titleweight='bold')     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE, labelweight='bold')    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
# plt.rc('text', usetex=True)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

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
	# signals = normalize(signals, axis=1, norm='l2')
	signals = signals/5
	adj = normalize(A + sp.eye(A.shape[0]))
	adj = sparse_mx_to_torch_sparse_tensor(adj)
	return adj, L.todense(), signals

def offdata2name(off_data):
	if off_data == 1:
		return 'adni'
	elif off_data == 2:
		return 'adrc'
	elif off_data == 3:
		return 'simuln'
	elif off_data == 4:
		return 'demo'

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

def offgan2name(off):
	if off== 1:
		return 'wgan'
	elif off == 11:
		return 'wgan-conv'
	elif off == 111:
		return 'wgan-gcn'
	elif off == 2:
		return 'lsgan'
	elif off == 21:
		return 'lsgan-conv'
	elif off == 211:
		return 'lsgan-gcn'

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

def plot_signals(off_data, gan_name, cn):
	root_path = '../result'
	fig = plt.figure(figsize=(10, 7))
	name_data = offdata2name(off_data)
	if cn:
		cn_name = 'cn'
	else:
		cn_name = 'ad'
	plot_path = "{}/{}/plt_{}_{}.png".format(root_path, gan_name, name_data, cn_name)
	data_path = '../data/{}/data_{}_4k.mat'.format(name_data, name_data)
	_, real_signals = load_data(data_path, is_control=cn)
	for off_model in range(1, 4):
		name_model = offmodel2name(off_model)
		if off_model == 3:
			signals = real_signals
			if off_data == 3: # Simuln
				signals = signals[:1000, :]
		else:
			sample_path = "{}/{}/{}/{}".format(root_path, gan_name, name_data, name_model)
			signal_path = os.path.join(sample_path, "{}/samples/samples_1000.npy".format(cn_name))
			signals = np.load(signal_path)[:real_signals.shape[0], :]
		means = np.mean(signals, axis = 0)
		plt.scatter(range(len(means)), means, label=name_model)
	plt.legend()
	plt.xlabel('Node')
	plt.ylabel("Value")
	plt.title('{} signals on {} - {}'.format(cn_name, name_data, gan_name))
	plt.show()
	fig.tight_layout()
	fig.savefig(plot_path, bbox_inches='tight')


def plot_eb(off_data, off_gan, alp, epsilon):
	fig = plt.figure(figsize=(10, 7))
	name_data = offdata2name(off_data)
	name_gan = offgan2name(off_gan)
	root_folder = "../result/eb-same/{}".format(name_gan)
	plot_path = "{}/plt_{}_{}_{}_{}.png".format(root_folder, name_data, name_gan, alp, epsilon)
	for off_model in range(1, 4):
		name_model = offmodel2name(off_model)
		fname = "eval_{}_{}_{}".format(name_data, name_gan, name_model)
		if off_model ==1:
			file_path = "{}/{}_{}_{}.npy".format(root_folder, fname, alp, epsilon)
		elif off_model == 2:
			file_path = "{}/{}_{}.npy".format(root_folder, fname, epsilon)
		else:
			sample_folder = '../result/eb-same/real'
			file_path = "{}/eval_{}_{}.npy".format(sample_folder, name_data,  epsilon)
		arr = np.load(file_path)
		# plt.plot(range(len(arr)), sorted(arr), label=name_model)
		plt.scatter(range(len(arr)), arr, label=name_model)
	plt.legend()
	plt.xlabel('Node')
	plt.ylabel("e-value")
	plt.title('{} data - e-value on {} - {} with epsilon = {}'.format(name_data, name_gan, alp, epsilon))
	plt.show()
	# fig.tight_layout()
	# fig.savefig(plot_path, bbox_inches='tight')


def plot_ttest_alp(off_data, off_gan, alp):
	name_data = offdata2name(off_data)
	name_gan = offgan2name(off_gan)
	root_path = '../result'
	output_path = "{}/ttest/{}".format(root_path, name_gan)
	plot_path = "{}/plt_{}_{}_{}.png".format(output_path, name_data, name_gan, alp)

	fig = plt.figure(figsize=(10, 7))
	if off_data == 4:
		model_list = [2, 3]
	else:
		model_list = [1, 2, 3]

	colors = ['y', 'b', 'r', 'g']
	for off_model in model_list:
		name_model = offmodel2name(off_model)
		if off_model == 1:
			file_path = "{}/{}_{}_{}_{}.npy".format(output_path, name_data, name_gan, name_model, alp)
		else:
			file_path = "{}/{}_{}_{}_0.01.npy".format(output_path, name_data, name_gan, name_model)
		arr = np.load(file_path)
		spio.savemat('/home/tuandinh/Documents/Project/glapGan/data/mesh_all/pvalue_{}_{}.mat'.format(name_model, name_data), {'pvalue_{}'.format(name_model):arr})
		return
		# from IPython import embed; embed()
		# idx = arr < 0.08
		# arr = arr[idx]
		plt.plot(range(len(arr)), sorted(arr), label=name_model,  c=colors[off_model])
		# plt.scatter(range(len(arr)), arr, label=name_model)

	plt.legend()
	plt.xlabel('Node')
	plt.ylabel("p-value")
	plt.title('{} data - BH correction with {} - alpha {}'.format(name_data, name_gan, alp))
	plt.show()
	fig.tight_layout()
	fig.savefig(plot_path, bbox_inches='tight')


def plot_ttest_all(off_data, off_gan):
	name_data = offdata2name(off_data)
	name_gan = offgan2name(off_gan)
	root_path = '../result'
	output_path = "{}/ttest/{}".format(root_path, name_gan)
	plot_path = "{}/plt_{}_{}_all.png".format(output_path, name_data, name_gan)

	fig = plt.figure(figsize=(7, 4))
	if off_data == 4:
		model_list = [2, 3]
	else:
		model_list = [1, 2, 3]

	colors = ['y', 'b', 'r', 'g']
	for off_model in [2, 3]:
		name_model = offmodel2name(off_model)
		file_path = "{}/{}_{}_{}_0.01.npy".format(output_path, name_data, name_gan, name_model)
		arr = np.load(file_path)
		# idx = arr < 0.08
		# arr = arr[idx]
		plt.plot(range(len(arr)), sorted(arr), label=name_model, c=colors[off_model])

	linestyles = ['-', '--', '-.', ':']
	i = 0
	for alp in [0.03, 0.07, 0.11, 0.15]:
		name_model = offmodel2name(1)  # lapgan
		file_path = "{}/{}_{}_{}_{}.npy".format(output_path, name_data, name_gan, name_model, alp)
		arr = np.load(file_path)
		plt.plot(range(len(arr)), sorted(arr), label=name_model + ' ' r'$\alpha=$'+str(alp), linestyle=linestyles[i], c='b')
		i = i + 1

	plt.legend()
	plt.xlabel('Node')
	plt.ylabel("p-value")
	plt.title('{} data - BH correction with {}'.format(name_data, name_gan))
	plt.show()
	fig.tight_layout()
	fig.savefig(plot_path, bbox_inches='tight')


def plot_ttest_zoom(off_data, off_gan):
	name_data = offdata2name(off_data)
	name_gan = offgan2name(off_gan)
	root_path = '../result'
	output_path = "{}/ttest/{}".format(root_path, name_gan)
	plot_path = "{}/plt_{}_{}_zoom.png".format(output_path, name_data, name_gan)

	# fig = plt.figure(figsize=(7, 4))
	fig, ax = plt.subplots() #
	x = range(4225)
	linestyles = ['-', '--', '-.', ':']
	colors = ['y', 'b', 'r', 'g']
	for off_model in [2, 3]:
		name_model = offmodel2name(off_model)
		file_path = "{}/{}_{}_{}_0.01.npy".format(output_path, name_data, name_gan, name_model)
		arr = np.load(file_path)
		ax.plot(x, sorted(arr), label=name_model, c=colors[off_model])

	i = 0
	for alp in [0.03, 0.07, 0.11, 0.15]:
		name_model = offmodel2name(1)  # lapgan
		file_path = "{}/{}_{}_{}_{}.npy".format(output_path, name_data, name_gan, name_model, alp)
		arr = np.load(file_path)
		ax.plot(x, sorted(arr), label=r'$\lambda_L=$'+str(alp), linestyle=linestyles[i], c='b')
		i = i + 1

	plt.legend(loc='lower right')
	plt.xlabel('Node (arbitrary order)')
	plt.ylabel("p-value")
	plt.title('Sorted p values after Benjamini-Hochberg correction')

	axins = zoomed_inset_axes(ax, 3, loc='upper left', bbox_to_anchor=(0.16, 0.9),bbox_transform=ax.figure.transFigure) # zoom-factor: 2.5,
	# axins = inset_axes(ax, 1,1 , loc=2,bbox_to_anchor=(0.2, 0.55))
	for off_model in [2, 3]:
		name_model = offmodel2name(off_model)
		file_path = "{}/{}_{}_{}_0.01.npy".format(output_path, name_data, name_gan, name_model)
		arr = np.load(file_path)
		axins.plot(x, sorted(arr), label=name_model, c=colors[off_model])

	i = 0
	for alp in [0.03, 0.07, 0.11, 0.15]:
		name_model = offmodel2name(1)  # lapgan
		file_path = "{}/{}_{}_{}_{}.npy".format(output_path, name_data, name_gan, name_model, alp)
		arr = np.load(file_path)
		axins.plot(x, sorted(arr), label=r'$\alpha=$'+str(alp), linestyle=linestyles[i], c='b')
		i = i + 1

	x1, x2, y1, y2 = 1100, 2100, 0, 0.08 # specify the limits
	axins.set_xlim(x1, x2) # apply the x-limits
	axins.set_ylim(y1, y2) # apply the y-limits
	axins.set_facecolor((1, 0.75, 0.75))
	mark_inset(ax, axins, loc1=1, loc2=3, linewidth=1, ec="0.5")
	# plt.yticks(visible=False)
	plt.xticks(visible=False)
	plt.grid(False)
	fig.tight_layout()
	fig.savefig(plot_path)
	plt.show()


n = 20
t = np.linspace(0.01, 0.1, num=n)
def get_fdr(r_pvalues, l_pvalues):
	lines = np.zeros((n, 1))
	for k in range(n):
		threshold = t[k]
		l_pred = np.asarray(l_pvalues < threshold, dtype=int)
		r_pred = np.asarray(r_pvalues < threshold, dtype=int)
		l_v = 0
		for i in range(len(r_pred)):
			if r_pred[i] == 1:
				l_v += l_pred[i]
		# l_v = np.sum(abs(l_pred - r_pred))
		# b_v = np.sum(abs(b_pred - r_pred))
		lines[k] = l_v / np.sum(r_pred)
		# from IPython import embed; embed()
	return lines

def plot_fdr(off_data, off_gan):
	name_data = offdata2name(off_data)
	name_gan = offgan2name(off_gan)
	root_path = '../result'
	output_path = "{}/ttest/{}".format(root_path, name_gan)
	plot_path = "{}/recall_{}_{}_all.png".format(output_path, name_data, name_gan)

	fig = plt.figure(figsize=(10, 7))
	b_pvalues = np.load("{}/{}_{}_{}_0.01.npy".format(output_path, name_data, name_gan, offmodel2name(2)))
	r_pvalues = np.load("{}/{}_{}_{}_0.01.npy".format(output_path, name_data, name_gan, offmodel2name(3)))

	b_line = get_fdr(r_pvalues, b_pvalues)
	plt.plot(t, b_line, label='WGAN (baseline)', c='r')
	linestyles = ['-', '--', '-.', ':']
	colors = ['y', 'b', 'r', 'g']
	i = 0
	for alp in [0.05, 0.1, 0.15]:
		name_model = offmodel2name(1)  # lapgan
		file_path = "{}/{}_{}_{}_{}.npy".format(output_path, name_data, name_gan, name_model, alp)
		l_pvalues = np.load(file_path)
		l_line = get_fdr(r_pvalues, l_pvalues)
		plt.plot(t, l_line, label=r'$\lambda_L=$'+str(alp), linestyle=linestyles[i], c='b')
		i = i + 1
		# from IPython import embed; embed()

	plt.legend()
	plt.xlabel('p-value threshold')
	plt.ylabel("Sensitivity")
	plt.grid(b=True)
	plt.title('Sensitivity of t-test with generated data');
	plt.show()
	fig.tight_layout()
	fig.savefig(plot_path, bbox_inches='tight')
