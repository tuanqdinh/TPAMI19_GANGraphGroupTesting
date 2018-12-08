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
	return np.sum(x) / len(arr)

epsilon = 0.01
n_pixels = 100

if len(sys.argv) == 4:
	off_data = int(sys.argv[1])
	off_gan = int(sys.argv[2])
	alp = float(sys.argv[3])
	plot_eb(off_data, off_gan, alp, epsilon)
else:
	off_data = int(sys.argv[1])
	off_gan = int(sys.argv[2])
	alp = float(sys.argv[3])
	off_model = int(sys.argv[4])
	name_data = offdata2name(off_data)
	name_model = offmodel2name(off_model)
	name_gan = offgan2name(off_gan)

	print('---- Working on {} data - {} model'.format(name_data, name_model))
	root_folder = "../result/eb-same/{}".format(name_gan)
	sample_folder = "{}_{}_{}".format(name_data, name_gan, name_model)
	fname = "eval_{}_{}_{}".format(name_data, name_gan, name_model)
	if off_model == 1:
		sample_path = "{}/{}_{}".format(root_folder, sample_folder, alp)
		file_path = "{}/{}_{}_{}.npy".format(root_folder, fname, alp, epsilon)
	elif off_model == 2:
		sample_path = os.path.join(root_folder, sample_folder)
		file_path = "{}/{}_{}.npy".format(root_folder, fname, epsilon)
	else:
		sample_folder = '../result/eb-same/real'
		sample_path = "{}/{}_real".format(sample_folder, name_data)
		file_path = "{}/eval_{}_{}.npy".format(sample_folder, name_data, epsilon)

	results = np.zeros(n_pixels)
	for i in range(n_pixels):
		fname = '{}/node{}.npy'.format(sample_path, i)
		if os.path.exists(fname):
			arr = np.load(fname)
			results[i] = cal(arr, epsilon)
		else:
			print(i, 'not exists')
			results[i] = 0
			# from IPython import embed; embed()
	np.save(file_path, results)
