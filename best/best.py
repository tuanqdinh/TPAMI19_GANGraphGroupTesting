import numpy as np
import pymc3 as pm
import pandas as pd

print('Running on PyMC3 v{}'.format(pm.__version__))

def trace_sd(x):
	return pd.Series(np.std(x, 0), name='sd')

def trace_quantiles(x):
	return pd.DataFrame(pm.quantiles(x, [1, 5, 25, 50, 75, 95, 99]))

'''
	@input: 2 groups
	@output: dist of difference and hpd
'''
def best(y1, y2, label1='group1', label2='group2', draws = 2000, n_init=200000, cores=2):
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
		diff_of_means = pm.Deterministic('diff_means', group1_mean - group2_mean)
		# diff_of_stds = pm.Deterministic('difference of stds', group1_std - group2_std)
		# effect_size = pm.Deterministic('effect size', diff_of_means / np.sqrt((group1_std**2 + group2_std**2) / 2))
   ####---- Fit Model
	with model:
		#M.sample(iter=110000, burn=10000)
	   trace = pm.sample(draws=draws, cores=cores, n_init=n_init, progressbar=False)

	return trace

def _best_summary(trace):
	# from IPython import embed; embed()
	t1 = pm.summary(trace, varnames=['diff_means'])
	t2 = pm.summary(trace, stat_funcs=[trace_quantiles], varnames=['diff_means'])
	value1 = t1[['mean', 'sd', 'hpd_2.5', 'hpd_97.5']].values[0]
	value2 = t2.values[0]
	con = np.concatenate([value1, value2])
	return con

def plot_posterior(pm, trace):
	#### Plot the stochastic parameters of the model
	 pm.plot_posterior(trace, varnames=['group1_mean','group2_mean', 'group1_std', 'group2_std', 'nu_minus_one'], color='#87ceeb');
	 pm.plot_posterior(trace, varnames=['difference of means','difference of stds', 'effect size'], ref_val=0, color='#87ceeb');

	 ####  plots the potential scale reduction parameter
	 pm.forestplot(trace, varnames=['group1_mean', 'group2_mean']);
	 pm.forestplot(trace, varnames=['group1_std', 'group2_std', 'nu_minus_one']);
	 plt.show()

def test():
	drug = (101,100,102,104,102,97,105,105,98,101,100,123,105,103,100,95,102,106,
			109,102,82,102,100,102,102,101,102,102,103,103,97,97,103,101,97,104,
			96,103,124,101,101,100,101,101,104,100,101)
	placebo = (99,101,100,101,102,100,97,101,104,101,102,102,100,105,88,101,100,
			   104,100,100,100,101,102,103,97,101,101,100,101,99,101,100,100,
			   101,100,99,101,100,102,99,100,99)

	y1 = np.array(drug)
	y2 = np.array(placebo)
	v = best(y1, y2) #are (mean, sd, mc_error, hpd_2.5, hpd_97.5, n_eff and Rhat.)
	fraction = 100
	quant_name = 'quantiles_{}.npy'.format(fraction)
	dists_name = 'dists_{}.npy'.format(fraction)
	if os.path.isfile(quant_name):
		quantiles = np.load(quant_name)
		dists = np.load(dists_name)
		n_pixels = dists.shape[0]
	else:
		NUM_QUANTILES = 7
		NUM_SUMMARY = 4
		# data preparation
		data_path = '../data/adni_data_4k.mat'
		_, ad_signals = load_data(data_path, is_control=False)
		_, cn_signals = load_data(data_path, is_control=True)

		nrows, n_pixels = np.shape(ad_signals)
		# n_pixels = int(n_pixels * fraction/100) # for testing
		quantiles = np.zeros((n_pixels, NUM_QUANTILES))
		dists = np.zeros((n_pixels, NUM_SUMMARY))
		for j in range(n_pixels):
			ad = [ad_signals[i][j] for i in range(nrows)]
			cn = [cn_signals[i][j] for i in range(nrows)]
			t1, t2 = best(ad, cn)
			quantiles[j, :] = t1
			dists[j, :] = t2

		np.save(quant_name, quantiles)
		np.save(dists_name, dists)

	# Plot the distributions
	means = dists[:, 0]
	sds = dists[:, 1]
	hpd_025 = dists[:, 2]
	hpd_975 = dists[:, 3]

	fig = plt.figure(figsize=(20, 5))
	plt.errorbar(range(n_pixels), means, yerr=sds, fmt='-o', linestyle="None")
	plt.scatter(range(n_pixels), hpd_025, c='b')
	plt.scatter(range(n_pixels), hpd_975, c='b')
	fig.tight_layout()
	fig.savefig('dists_{}.png'.format(fraction), bbox_inches='tight')

	plt.xlabel('Vertex')
	plt.title('Difference of means distribution')
	plt.show()
