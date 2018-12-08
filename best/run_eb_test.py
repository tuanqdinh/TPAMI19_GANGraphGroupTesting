import numpy as np
import scipy.io as spio
import pandas as pd
import multiprocessing
import os, sys, time
sys.path.append('../util/')
from helper import *
from mypool import MyPool
from best import best

draws = 2000
n_init= 200000
cores = 2
fraction = 100
list_nodes = [] # global



off_data = 4
name_data = offdata2name(off_data)

root_path = '../result'
### variables
cn_path = '../data/demo/data_demo_100_{}.npy'.format('cn')
ad_path = '../data/demo/data_demo_100_{}.npy'.format('ad')
ad_signals = np.load(ad_path)
cn_signals = np.load(cn_path)


output_path = "{}/eb-same/{}/".format(root_path, name_data)
if os.path.exists(output_path):
	for node in os.listdir(output_path):
		t = node.split('.')[0]
		list_nodes.append(int(t[4:]))
	print('Exists {} nodes'.format(len(list_nodes)))
else:
	mkdir(output_path)

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
