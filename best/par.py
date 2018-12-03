# from multiprocessing import Pool
import multiprocessing
import multiprocessing.pool

from functools import partial
import numpy as np
import scipy.io as spio
import time
import pymc3 as pm
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
plt.style.use('ggplot')
import os, sys
sys.path.append('../utils/')
from helpers import load_data

print('Running on PyMC3 v{}'.format(pm.__version__))

class NoDaemonProcess(multiprocessing.Process):
	# make 'daemon' attribute always return False
	def _get_daemon(self):
		return False
	def _set_daemon(self, value):
		pass
	daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(multiprocessing.pool.Pool):
	Process = NoDaemonProcess


def trace_quantiles(x):
	return pd.DataFrame(pm.quantiles(x, [1, 5, 25, 50, 75, 95, 99]))

'''
	@input: 2 groups
	@output: dist of difference and hpd
'''
def best(y1, y2, label1='group1', label2='group2'):
	y = pd.DataFrame(dict(value=np.r_[y1, y2], group=np.r_[[label1]*len(y1), [label2]*len(y2)]))
	# y.hist('value', by='group');

	##-------PRIORS- specified our probabilistic model
	# Means
	mu_m = y.value.mean()
	mu_s = y.value.std() * 2 # only here is different
	with pm.Model() as model:
		group1_mean = pm.Normal('group1_mean', mu_m, sd=mu_s)
		group2_mean = pm.Normal('group2_mean', mu_m, sd=mu_s)
	# Stds
	sigma_low = 1
	sigma_high = 10
	with model:
		group1_std = pm.Uniform('group1_std', lower=sigma_low, upper=sigma_high)
		group2_std = pm.Uniform('group2_std', lower=sigma_low, upper=sigma_high)
	# Nu
	with model:
		nu = pm.Exponential('nu_minus_one', 1/29.) + 1
	# pm.kdeplot(np.random.exponential(30, size=10000), shade=0.5);

	with model:
		lam1 = group1_std**-2
		lam2 = group2_std**-2
		group1 = pm.StudentT(label1, nu=nu, mu=group1_mean, lam=lam1, observed=y1)
		group2 = pm.StudentT(label2, nu=nu, mu=group2_mean, lam=lam2, observed=y2)

	####---- calculating the comparisons of interest
	with model:
		diff_of_means = pm.Deterministic('difference of means', group1_mean - group2_mean)
		diff_of_stds = pm.Deterministic('difference of stds', group1_std - group2_std)
		effect_size = pm.Deterministic('effect size',
									   diff_of_means / np.sqrt((group1_std**2 + group2_std**2) / 2))
   ####---- Fit Model
	with model:
		#M.sample(iter=110000, burn=10000)
	   trace = pm.sample(2000, cores=2)

	# from IPython import embed; embed()
	t1 = pm.summary(trace, varnames=['difference of means'])
	t2 = pm.summary(trace, stat_funcs=[trace_quantiles], varnames=['difference of means'])
	value1 = t1[['mean', 'sd', 'hpd_2.5', 'hpd_97.5']].values[0]
	value2 = t2.values[0]
	con = np.concatenate([value1, value2])
	return con

def run_main():
	fraction = 100
	if sys.argv[1] == 'True':
		print('Laplacian')
		reg_folder = 'laplacian'
	else:
		print('Normal')
		reg_folder = 'normal'
	dists_name = 'eb_{}_{}.npy'.format(reg_folder, fraction)
	# from IPython import embed; embed()
	if os.path.isfile(dists_name):
		dists = np.load(dists_name)
		n_pixels = dists.shape[0]
	else:
		# data preparation
		# data_path = '../data/adni_data_4k.mat'
		# _, ad_signals = load_data(data_path, is_control=False)
		# _, cn_signals = load_data(data_path, is_control=True)
		ad_path = "../results/{}/ad/samples/samples_1000.npy".format(reg_folder)
		cn_path = "../results/{}/cn/samples/samples_1000.npy".format(reg_folder)

		ad_signals = np.load(ad_path)
		cn_signals = np.load(cn_path)

		nrows, n_pixels = np.shape(ad_signals)
		n_pixels = int(n_pixels * fraction/100) # for testing

		def par_best(j):
			print('Vertex {}'.format(j))
			ad = ad_signals[:, j]
			cn = cn_signals[:, j]
			t = best(ad, cn)
			return t

		num_cores = multiprocessing.cpu_count()
		start_time = time.time()
		# func = partial(f, ad_signals, cn_signals)
		with MyPool(num_cores) as p:
			data = p.map(par_best, range(n_pixels))

		print("Total time: {:4.4f}".format(time.time() - start_time))

		dists = np.asarray(data)
		np.save(dists_name, dists)

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
	fig.savefig('dists_{}_{}.png'.format(reg_folder, fraction), bbox_inches='tight')

	plt.xlabel('Vertex')
	plt.title('Difference of means distribution')
	plt.show()



if __name__ == '__main__':
	fraction = 100
	dists_name = 'eb_real_{}.npy'.format(fraction)
	# from IPython import embed; embed()
	if os.path.isfile(dists_name):
		dists = np.load(dists_name)
		n_pixels = dists.shape[0]
	else:
		# data preparation
		data_path = '../data/sim/fake_data_4k.mat'
		mat = spio.loadmat(data_path, squeeze_me=True)
		ad_signals = mat['f_ad_signals']
		cn_signals = mat['f_cn_signals']

		nrows, n_pixels = np.shape(ad_signals)
		n_pixels = int(n_pixels * fraction/100) # for testing

		def par_best(j):
			print('Vertex {}'.format(j))
			ad = ad_signals[:, j]
			cn = cn_signals[:, j]
			t = best(ad, cn)
			return t

		num_cores = multiprocessing.cpu_count()
		start_time = time.time()
		# func = partial(f, ad_signals, cn_signals)
		with MyPool(num_cores) as p:
			data = p.map(par_best, range(n_pixels))

		print("Total time: {:4.4f}".format(time.time() - start_time))

		dists = np.asarray(data)
		np.save(dists_name, dists)

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
	fig.savefig('dists_{}_{}.png'.format(reg_folder, fraction), bbox_inches='tight')

	plt.xlabel('Vertex')
	plt.title('Difference of means distribution')
	plt.show()
