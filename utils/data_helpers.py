import numpy as np
import scipy
import scipy.io as spio
from scipy.sparse import csgraph
from sklearn.preprocessing import normalize
import torch
from torch.utils import data

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
