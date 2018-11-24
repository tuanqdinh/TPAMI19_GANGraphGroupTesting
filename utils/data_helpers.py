import numpy as np
import scipy
from scipy import stats
import scipy.io as spio
import os, random

##################### Dataset ##############################
# Dataset iterator
def inf_train_gen(BATCH_SIZE):
    scale = 2.
    centers = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1. / np.sqrt(2), 1. / np.sqrt(2)),
        (1. / np.sqrt(2), -1. / np.sqrt(2)),
        (-1. / np.sqrt(2), 1. / np.sqrt(2)),
        (-1. / np.sqrt(2), -1. / np.sqrt(2))
    ]
    centers = [(scale * x, scale * y) for x, y in centers]
    while True:
        dataset = []
        for i in range(BATCH_SIZE):
            point = np.random.randn(2) * .02
            center = random.choice(centers)
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        dataset = np.array(dataset, dtype='float32')
        dataset /= 1.414  # stdev
        yield dataset


def load_mat(fname, ad):
	mat = spio.loadmat(fname, squeeze_me=True)
	L = mat['L']
	if ad == 1:
		signals = np.asarray(mat['ad_signals'])
	else:
		signals = np.asarray(mat['cn_signals'])
	return [L.todense(), signals]

def load_dataset(config, n_nodes):
	data_folder = '../data/'
	lh = dataset % 10 # left brain
	# dim = dataset // 10 #nodes
	d_name = data_folder + 'mesh/fake_data_4k_normalized.mat'
	#'mesh/smooth_test/data/data_4k_normalized.mat'
	lap_matrix, data = load_mat(d_name, lh)
	# x = np.linalg.norm(data, 1, axis=1)
	# y = np.argsort(x)
	# data = data[y[:120], :]
	# data = data[1:120, :]
	data_size = np.shape(data)[0]
	return data, data_size, lap_matrix

############ Utils ##########################
def gen_data(data, ds_size, batch_size):
    while True:
        for i in range(int(ds_size / batch_size)):
            start = i * batch_size
            end = (i + 1) * batch_size
            yield data[start:end, :]

def get_batch(data, batch_size, ds):
	# for i in range(int(ds.size / BATCH_SIZE)):
	#     start = i * BATCH_SIZE
	#     end = (i + 1) * BATCH_SIZE
	#     yield ds.images[start:end], ds.labels[start:end]
	sig = data
	batch = np.outer(np.ones([batch_size, 1]), sig)
	noise = np.random.uniform(0, 0.05, [batch_size, len(data)])
	X_mb =  batch + noise
	return X_mb
