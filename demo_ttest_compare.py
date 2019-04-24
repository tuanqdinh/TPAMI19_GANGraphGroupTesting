


### compare # of difference between real data and faked
'''
	Save pvalues and rejects to files
'''
import numpy as np
import os, sys, time
from util.helper import *
import argparse

'''
print('Arguments!!! data - model - control')
print('----data : 1 ADNI - 2 ADRC - 3 Simuln')
print('----gan: 1 wgan - 2 lsgan')
print('----model: 1 lapgan - 2 baseline - 3 real')
'''
parser = argparse.ArgumentParser()
parser.add_argument('--off_data', type=int, default=1)
parser.add_argument('--off_gan', type=int, default=1)
parser.add_argument('--alpha', type=float, default=0.03)
parser.add_argument('--path_result', type=str, default='result', help='result path')

args = parser.parse_args()

name_data = offdata2name(args.off_data)
name_gan = offgan2name(args.off_gan)
path_ttest = os.path.join(args.path_result, "ttest/{}".format(name_gan))

pvalues = []
rejects = []
for i in range(3):
    off_model = i + 1
    name_model = offmodel2name(off_model)
    name_data_gan_model = "{}-{}-{}".format(name_data, name_gan, name_model)
    path_saved_pvalue = os.path.join(path_ttest, "pvalue_{}_{}.npy".format(name_data_gan_model, args.alpha))
    path_saved_reject = os.path.join(path_ttest, "reject_{}_{}.npy".format(name_data_gan_model, args.alpha))
    pvalue = np.load(path_saved_pvalue)
    reject = np.load(path_saved_reject)
    pvalues.append(pvalue)
    rejects.append(reject)
    print("Model: ", name_model, " Rejects: ", sum(reject))
    print("P-pvalue: ", np.mean(pvalue))

def compare_region(main_rejects, rej):
	idx = main_rejects == 1
	m = main_rejects[idx]
	r = rej[idx]
	return sum(m == r)


print('Compare')
print('Baseline vs Real: ', compare_region(rejects[-1], rejects[1]))
print('LapGAN vs Real: ', compare_region(rejects[-1], rejects[0]))

print('\nFinish ------')
