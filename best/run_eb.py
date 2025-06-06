import numpy as np
import scipy.io as spio
import pandas as pd
import multiprocessing
import os, sys, time
sys.path.append('../util/')
from helper import *
from mypool import MyPool
from best import best

OFFSET_REAL = 3
draws = 2000
n_init= 200000
cores = 2
fraction = 100
list_nodes = [] # global


print('Arguments!!! data - model - control')
print('----data : 1 ADNI - 2 ADRC - 3 Simuln')
print('----gan: 1 wgan - 2 lsgan')
print('----alp: 0.01')
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

print('---- Working on {} data - {} model - {} - {}'.format(name_data, name_model, name_gan, alp))

root_path = '../result'
sample_path = "{}/{}".format(root_path, name_gan)
if off_gan == 1:
	output_path = "{}/eb-same/{}/{}_{}_{}_{}".format(root_path, name_gan, name_data, name_gan, name_model, alp)
else:
	output_path = "{}/eb-same/{}/{}_{}_{}".format(root_path, name_gan, name_data, name_gan, name_model)
### variables
def _load_signals():
	data_path = '../data/{}/data_{}_4k.mat'.format(name_data, name_data)
	if OFFSET_REAL == off_model: # real data
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


ad_signals, cn_signals = _load_signals()
if os.path.exists(output_path):
	for node in os.listdir(output_path):
		t = node.split('.')[0]
		list_nodes.append(int(t[4:]))
	print('Exists {} nodes'.format(len(list_nodes)))
else:
	mkdir(output_path)


def par_best_summary(j):
	print('Vertex {}'.format(j))
	ad = ad_signals[:, j]
	cn = cn_signals[:, j]
	try:
		trace = best(ad, cn)
		t = _best_summary(trace)
	except:
		t = np.ones([len(ad)])
		print('Error at ' + str(j))
	return t

def par_best(j):
	if j in list_nodes:
		return
	print('\nVertex {}\n'.format(j))
	ad = ad_signals[:, j]
	cn = cn_signals[:, j]
	try:
		trace = best(ad, cn, draws=draws, cores=cores, n_init=n_init)
		trace_name = os.path.join(output_path, 'node{}.npy'.format(j))
		np.save(trace_name, trace['diff_means'])
	except:
		print('Error at ' + str(j))

if len(list_nodes) == ad_signals.shape[1]:
	print('Nothing left!!')
else:
	n_pixels = ad_signals.shape[1]
	n_pixels = int(n_pixels * fraction/100) # for testing
	# from IPython import embed; embed()
	num_cores = multiprocessing.cpu_count()
	start_time = time.time()
	# par_best(0)
	with MyPool(num_cores) as p:
		p.map(par_best, range(n_pixels))
	print("Total time: {:4.4f}".format(time.time() - start_time))
	# np.save(dists_name, dists)
print('\nFinish ------')
