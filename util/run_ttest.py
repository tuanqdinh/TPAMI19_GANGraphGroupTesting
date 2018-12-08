import numpy as np
import scipy.io as spio
import pandas as pd
import multiprocessing
import os, sys, time
from helper import *
from mypool import MyPool
from scipy.stats import ttest_ind
from statsmodels.sandbox.stats.multicomp import multipletests

OFFSET_REAL = 3
draws = 2000
n_init= 200000
cores = 2
fraction = 100


print('Arguments!!! data - model - control')
print('----data : 1 ADNI - 2 ADRC - 3 Simuln')
print('----gan: 1 wgan - 2 lsgan')
print('----model: 1 lapgan - 2 baseline - 3 real')

### GET Input
if len(sys.argv) < 4:
	print('Not enough arguments!!! data - model')
	os._exit(0)
elif len(sys.argv) == 4:
	off_data = int(sys.argv[1])
	off_gan = int(sys.argv[2])
	alp = float(sys.argv[3])
	print('Plotting')
	plot_ttest(off_data, off_gan, alp)
	os._exit(0)

off_data = int(sys.argv[1])
off_gan = int(sys.argv[2])
alp = float(sys.argv[3])
off_model = int(sys.argv[4])
name_data = offdata2name(off_data)
name_model = offmodel2name(off_model)
name_gan = offgan2name(off_gan)

print('---- Working on {} data - {} model - {} gan'.format(name_data, name_model, name_gan))

root_path = '../result'
sample_path = "{}/{}".format(root_path, name_gan)
output_path = "{}/ttest/{}".format(root_path, name_gan)
mkdir(output_path)
file_path = "{}/{}_{}_{}_{}.npy".format(output_path, name_data, name_gan, name_model, alp)
### variables
def _load_signals():
	data_path = '../data/{}/data_{}_4k.mat'.format(name_data, name_data)
	if OFFSET_REAL == off_model: # real data
		if off_data == 4:
			cn_path = '../data/demo/data_demo_100_{}.npy'.format('cn')
			ad_path = '../data/demo/data_demo_100_{}.npy'.format('ad')
			ad_signals = np.load(ad_path)
			cn_signals = np.load(cn_path)
			from IPython import embed; embed()
		else:
			_, _,  ad_signals = load_data(data_path, is_control=False)
			_,_,  cn_signals = load_data(data_path, is_control=True)
			if off_data == 3: # Simuln
				ad_signals = ad_signals[:1000, :]
				cn_signals = cn_signals[:1000, :]
	else:
		ad_path = os.path.join(sample_path, 'samples_{}_{}_{}_{}_{}.npy'.format(name_data, name_gan, name_model, 'ad', alp))
		cn_path = os.path.join(sample_path, 'samples_{}_{}_{}_{}_{}.npy'.format(name_data, name_gan, name_model, 'cn', alp))
		if off_data == 4:
			ad_signals = np.load(ad_path)
			cn_signals = np.load(cn_path)
		else:
			_, _, real_ad_signals = load_data(data_path, is_control=False)
			_, _,  real_cn_signals = load_data(data_path, is_control=True)
			ad_signals = np.load(ad_path)[:real_ad_signals.shape[0], :]
			cn_signals = np.load(cn_path)[:real_cn_signals.shape[0], :]

	return ad_signals, cn_signals


# sample_path = "{}/{}/{}/{}".format(root_path, name_gan, name_data, name_model)
ad_signals, cn_signals = _load_signals()
# lap_cn_data = re_scale(lap_cn_data, data_mean, data_range)

def get_t_stats(data1, data2, pvalue=True):
	# For each voxel
	twosample_results = ttest_ind(data1, data2)
	if pvalue:
		return twosample_results[1]
	else:
		return twosample_results[0]

def par_best(j):
	print('\nVertex {}\n'.format(j))
	ad = ad_signals[:, j]
	cn = cn_signals[:, j]
	return get_t_stats(ad, cn)

if os.path.isfile(file_path):
	print('Exists!!')
else:
	n_pixels = ad_signals.shape[1]
	num_cores = multiprocessing.cpu_count() - 3
	start_time = time.time()
	with MyPool(num_cores) as p:
		pvalues = p.map(par_best, range(n_pixels))

	p_values = np.asarray(pvalues)
	_, pvals_corrected, alphacSidak, alphacBonf = multipletests(p_values, alpha=0.2, method='fdr_bh')
	print("Total time: {:4.4f}".format(time.time() - start_time))

	np.save(file_path, pvals_corrected)
print('\nFinish ------')
