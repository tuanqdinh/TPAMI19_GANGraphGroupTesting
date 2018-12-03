import numpy as np
import os, sys
sys.path.append('../util/')
from helper import *

def count(x, epsilon):
	if abs(x) <= epsilon:
		return 1
	else:
		return 0

def cal(arr, epsilon):
	x = [count(x, epsilon) for x in arr]
	return np.sum(x)


epsilon = 0.01

if len(sys.argv) < 3:
	off_data = int(sys.argv[1])
	plot_eb(off_data, epsilon)
else:
	# print('Arguments!!! data - model - control')
	# print('----data : 1 ADNI - 2 ADRC - 3 Simuln')
	# print('----model: 1 lapgan - 2 baseline - 3 real')

	off_data = int(sys.argv[1])
	off_model = int(sys.argv[2])
	name_data = offdata2name(off_data)
	name_model = offmodel2name(off_model)

	print('---- Working on {} data - {} model'.format(name_data, name_model))
	root_path = '../result'
	sample_path = "{}/eb/lsgan/{}_{}".format(root_path, name_data, name_model)

	file_path = "{}/eb/lsgan/eval_{}_{}_{}.npy".format(root_path, name_data, name_model, epsilon)
	# plot_path = "{}/eb/lsgan/plt_{}_{}_{}.png".format(root_path, name_data, name_model, epsilon)
	n_pixels = 4225
	results = np.zeros(n_pixels)
	for i in range(n_pixels):
		arr = np.load('{}/node{}.npy'.format(sample_path, i))
		results[i] = cal(arr, epsilon) / n_pixels

	np.save(file_path, results)

# from IPython import embed; embed()
# Plot the distributions
