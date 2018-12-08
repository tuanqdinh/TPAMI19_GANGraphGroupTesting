from scipy import stats
import scipy
import scipy.io as spio
import numpy as np
import os, sys
from scipy.stats import ttest_ind
from statsmodels.sandbox.stats.multicomp import multipletests


def get_t_stats(data1, data2, pvalue=True):
	# For each voxel
	twosample_results = ttest_ind(data1, data2)
	if pvalue:
		return twosample_results[1]
	else:
		return twosample_results[0]

def load_mat(fname):
	mat = spio.loadmat(fname, squeeze_me=True)
	A = mat['A']
	ad_signals = np.asarray(mat['ad_signals'])
	cn_signals = np.asarray(mat['cn_signals'])
	data_mean = np.asarray(mat['data_mean'])
	data_range = np.asarray(mat['data_range'])
	return [A, ad_signals, cn_signals, data_mean, data_range]

def get_bh_correction(A, data_ad, data_cn, pvalue=True, correction=False):
	p_values = [get_t_stats(data_ad[:, i], data_cn[:, i]) for i in range(n_voxels)]
	if correction:
		reject, pvals_corrected, alphacSidak, alphacBonf = multipletests(p_values, alpha=0.2, method='fdr_bh')
		return pvals_corrected
	else:
		return p_values

def cut(data):
	x = np.linalg.norm(data, 1, axis=1)
	y = np.argsort(x)
	return y

def re_scale(data, data_mean, data_range):
	return data * data_range + data_mean

if __name__ == '__main__':
	value = 'p'
	pvalue = True
	smooth = False
	n_points = 270
	output = 'output2'
	eta = sys.argv[1]
	correction = True

	# Load data
	[A, ad_signals, cn_signals, data_mean, data_range] = 	load_mat('../fake_data_4k_normalized.mat')#load_mat('data/data_4k_normalized.mat')
	n_voxels = A.shape[0]
	ad_signals = re_scale(ad_signals, data_mean, data_range)
	cn_signals = re_scale(cn_signals, data_mean, data_range)

	# ad_ind = cut(ad_signals)
	# cn_ind = cut(cn_signals)
	ad_ind = np.arange(n_voxels)
	cn_ind = np.arange(n_voxels)


	lap_ad_name = 'rebuttal_ad41_011_{:1}.npy'.format(eta)
	lap_cn_name = 'rebuttal_cn42_011_{:1}.npy'.format(eta)
	lap_ad_data = np.load('data/' + lap_ad_name)
	lap_cn_data = np.load('data/' + lap_cn_name)
	lap_ad_data = lap_ad_data[:n_points, :]
	lap_cn_data = lap_cn_data[:n_points, :]
	lap_ad_data = re_scale(lap_ad_data, data_mean, data_range)
	lap_cn_data = re_scale(lap_cn_data, data_mean, data_range)

	p_values_lap = get_bh_correction(A, lap_ad_data, lap_cn_data, correction)
	np.save('{}/rebuttal_{}_values_lap_{}.npy'.format(output, value, eta), p_values_lap)
