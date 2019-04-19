'''
	Save pvalues and rejects to files
'''
import numpy as np
import scipy.io as spio
import pandas as pd
import multiprocessing
import os, sys, time
from util.helper import *
from util.mypool import MyPool
from scipy.stats import ttest_ind
from statsmodels.sandbox.stats.multicomp import multipletests
import argparse

'''
print('Arguments!!! data - model - control')
print('----data : 1 ADNI - 2 ADRC - 3 Simuln')
print('----gan: 1 wgan - 2 lsgan')
print('----model: 1 lapgan - 2 baseline - 3 real')
'''
parser = argparse.ArgumentParser()
parser.add_argument('--off_data', type=int, default=1)
parser.add_argument('--off_model', type=int, default=2)
parser.add_argument('--off_gan', type=int, default=1)
parser.add_argument('--alpha', type=float, default=0.01)
parser.add_argument('--path_result', type=str, default='result', help='result path')
parser.add_argument('--path_data', type=str, default='data/', help='path for data')

args = parser.parse_args()

OFFSET_REAL = 3
draws = 2000
n_init= 200000
cores = 2
fraction = 100


def get_t_stats(data1, data2, pvalue=True):
	# For each voxel
	twosample_results = ttest_ind(data1, data2)
	if pvalue:
		return twosample_results[1]
	else:
		return twosample_results[0]

name_data = offdata2name(args.off_data)
name_model = offmodel2name(args.off_model)
name_gan = offgan2name(args.off_gan)

name_data_gan_model = "{}-{}-{}".format(name_data, name_gan, name_model)
print('---- Working on {}'.format(name_data_gan_model))
path_real_data = os.path.join(args.path_data, '{}/data_{}_4k.mat'.format(name_data, name_data))
path_generated_sample = os.path.join(args.path_result, name_gan)

path_ttest = os.path.join(args.path_result, "ttest/{}".format(name_gan))
path_saved_pvalue = os.path.join(path_ttest, "pvalue_{}_{}.npy".format(name_data_gan_model, args.alpha))
path_saved_reject = os.path.join(path_ttest, "reject_{}_{}.npy".format(name_data_gan_model, args.alpha))

mkdir(path_ttest)

def _load_signals():
	if OFFSET_REAL == off_model: # real data
		if off_data == 4:
			cn_path = 'data/demo/data_demo_100_{}.npy'.format('cn')
			ad_path = 'data/demo/data_demo_100_{}.npy'.format('ad')
			ad_signals = np.load(ad_path)
			cn_signals = np.load(cn_path)
			from IPython import embed; embed()
		else:
			_, _, ad_signals = load_data(path_real_data, is_control=False)
			_, _, cn_signals = load_data(path_real_data, is_control=True)
			if off_data == 3: # Simuln
				ad_signals = ad_signals[:1000, :]
				cn_signals = cn_signals[:1000, :]
	else:
		ad_path = os.path.join(path_generated_sample, 'samples_{}-{}_{}.npy'.format(name_data_gan_model, 'ad', args.alpha))
		cn_path = os.path.join(path_generated_sample, 'samples_{}-{}_{}.npy'.format(name_data_gan_model, 'cn', args.alpha))
		if off_data == 4:
			ad_signals = np.load(ad_path)
			cn_signals = np.load(cn_path)
		else:
			_, _, real_ad_signals = load_data(path_real_data, is_control=False)
			_, _,  real_cn_signals = load_data(path_real_data, is_control=True)
			ad_signals = np.load(ad_path)[:real_ad_signals.shape[0], :]
			cn_signals = np.load(cn_path)[:real_cn_signals.shape[0], :]

	return ad_signals, cn_signals


ad_signals, cn_signals = _load_signals()

if os.path.isfile(path_saved_pvalue):
	print('Exists!!')
else:
	n_pixels = ad_signals.shape[1]
	num_cores = multiprocessing.cpu_count() - 3
	start_time = time.time()
	def par_best(j):
		print('\nVertex {}\n'.format(j))
		ad = ad_signals[:, j]
		cn = cn_signals[:, j]
		return get_t_stats(ad, cn)
	with MyPool(num_cores) as p:
		pvalues = p.map(par_best, range(n_pixels))

	p_values = np.asarray(pvalues)
	rejects, pvals_corrected, _, _ = multipletests(p_values, alpha=0.2, method='fdr_bh')
	print("Total time: {:4.4f}".format(time.time() - start_time))

	np.save(path_saved_pvalue, pvals_corrected)
	np.save(path_saved_reject, rejects)
print('\nFinish ------')
