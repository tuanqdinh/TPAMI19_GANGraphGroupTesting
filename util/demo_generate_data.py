
import numpy as np
import numpy as np
import scipy.io as spio
import scipy
import multiprocessing
import os, sys, time
from util.mypool import MyPool
from scipy.stats import ttest_ind
from statsmodels.sandbox.stats.multicomp import multipletests

### a random matrix
SIZE = 4000
INIT_PTS = 2000
DIFF_PTS = 2000
A = np.random.uniform(size=(SIZE, SIZE))
p = 0.9
for i in range(SIZE):
    for j in range(SIZE):
        if A[i, j] < p:
            A[i, j] = 0
            A[j, i] = 0
###
inds = np.random.randint(low = 0, high=SIZE, size=INIT_PTS)
signal = np.random.uniform(low=0, high=0.1, size=SIZE)
signal[inds] = p


### diffuse
A2 = np.multiply(A, A)
A3 = np.multiply(A2, A)
signal = np.dot(A3, signal)
signal = signal / np.max(signal)
ad_signal = np.copy(signal)
cn_signal = np.copy(signal)
idx = np.random.randint(low = 0, high=SIZE, size=DIFF_PTS)
for i in inds: ##############-----------------------DIFF
    ad_signal[i] = ad_signal[i] + 0.1

# from IPython import embed; embed()


ad_signals = []
cn_signals = []
for i in range(200):
    noise = np.random.normal(loc=0.01, scale=0.1, size=SIZE)
    ad = ad_signal + noise
    cn = cn_signal + noise
    ad_signals.append(ad)
    cn_signals.append(cn)
ad_signals = np.asarray(ad_signals)
cn_signals = np.asarray(cn_signals)

L = scipy.sparse.csgraph.laplacian(A, normed=False)
data_path = 'data/demo/data_demo.npy'
dict = {}
dict['ad'] = ad_signals
dict['cn'] = cn_signals
dict['A'] = A
dict['L'] = L
np.save(data_path, dict)

#### t-test
def get_t_stats(data1, data2, pvalue=True):
	# For each voxel
	twosample_results = ttest_ind(data1, data2)
	if pvalue:
		return twosample_results[1]
	else:
		return twosample_results[0]

# else:
n_pixels = ad_signals.shape[1]
num_cores = multiprocessing.cpu_count() - 3
def par_best(j):
	# print('\nVertex {}\n'.format(j))
	ad = ad_signals[:, j]
	cn = cn_signals[:, j]
	return get_t_stats(ad, cn)
with MyPool(num_cores) as p:
	pvalues = p.map(par_best, range(n_pixels))

p_values = np.asarray(pvalues)
rejects, pvals_corrected, _, _ = multipletests(p_values, alpha=0.2, method='fdr_bh')
print('#-Reject: ', np.sum(rejects))
