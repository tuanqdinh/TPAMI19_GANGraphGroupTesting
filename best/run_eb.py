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

def _load_signals(off_data, off_model, sample_path):
	if OFFSET_REAL == off_model: # real data
		data_path = '../data/{}/data_{}_4k.mat'.format(name_data, name_data)
		_, ad_signals = load_data(data_path, is_control=False)
		_, cn_signals = load_data(data_path, is_control=True)
		if off_data == 3: # Simuln
			ad_signals = ad_signals[:1000, :]
			cn_signals = cn_signals[:1000, :]
	else:
		data_path = '../data/{}/data_{}_4k.mat'.format(name_data, name_data)
		_, real_ad_signals = load_data(data_path, is_control=False)
		_, real_cn_signals = load_data(data_path, is_control=True)
		ad_path = os.path.join(sample_path, "ad/samples/samples_1000.npy")
		cn_path = os.path.join(sample_path, "cn/samples/samples_1000.npy")
		ad_signals = np.load(ad_path)[:real_ad_signals.shape[0], :]
		cn_signals = np.load(cn_path)[:real_cn_signals.shape[0], :]
	return ad_signals, cn_signals

### GET Input
if len(sys.argv) < 3:
	print('Not enough arguments!!! data - model')

print('Arguments!!! data - model - control')
print('----data : 1 ADNI - 2 ADRC - 3 Simuln')
print('----model: 1 lapgan - 2 baseline - 3 real')

off_data = int(sys.argv[1])
off_model = int(sys.argv[2])
name_data = offdata2name(off_data)
name_model = offmodel2name(off_model)
print('---- Working on {} data - {} model'.format(name_data, name_model))
### variables

root_path = '../result/'
gan = 'lsgan-conv'

sample_path = "{}/{}/{}/{}".format(root_path, gan, name_data, name_model)
ad_signals, cn_signals = _load_signals(off_data, off_model, sample_path)

output_path = "{}/eb-same/{}/{}_{}".format(root_path, gan, name_data, name_model)
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
	num_cores = multiprocessing.cpu_count() - 3
	start_time = time.time()
	with MyPool(num_cores) as p:
		p.map(par_best, range(n_pixels))
	print("Total time: {:4.4f}".format(time.time() - start_time))
	# np.save(dists_name, dists)
print('\nFinish ------')
