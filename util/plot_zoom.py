import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import sys
import csv
import scipy.io as spio
plt.style.use('ggplot')

# image_folder = '/home/tuandinh/Dropbox/'
image_folder = 'imgs/'

SMALL_SIZE = 8.5
MEDIUM_SIZE = 9.5
BIGGER_SIZE = 11

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE, titleweight='bold')     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE, labelweight='bold')    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
# plt.rc('text', usetex=True)
# matplotlib.rc('axes',edgecolor='k')
mesh_folder = '.'
def plot_pvalue(alpha, raw=False):
	value = 'p'
	output = 'output2'
	smooth = False

	if raw:
		title = '2-sample t-test with Benjamini/Hochberg correction'
		raw_name = 'raw'
	else:
		title = 'Smooth permutation test with holm-sidak correction'
		raw_name = 'smooth'
	save_name = '{}/plt_{}values_bh.png'.format(image_folder, value)

	# real
	p_values_real = np.load('{}/{}_values_real.npy'.format(output, value))
	p_values_van = np.load('{}/{}_values_van.npy'.format(output, value))
	p_values_lap = np.load('{}/rebuttal_{}_values_lap_{}.npy'.format(output, value, alpha))

	n_idx = 20
	p_van = np.sort(p_values_van)[0:-1:n_idx]
	p_lgan = np.sort(p_values_lap)[0:-1:n_idx]
	p_real = np.zeros((p_values_real.shape[0], len(p_van)))
	for i in range(p_real.shape[0]):
		p_real[i, :] = np.sort(p_values_real[i, :])[0:-1:n_idx]

	fig, ax = plt.subplots() # create a new figure with a default 111 subplot
	x = np.arange(len(p_van)) * n_idx
	# fig = plt.figure(figsize=(5, 3.75))
	ax.set_title(title)
	ax.plot(x, p_lgan, label=r'GL-GAN', c='b')
	ax.plot(x, p_van, label=r'baseline', c='r')
	linestyles = ['-', '--', '-.', ':', (0, (3, 10, 1, 10))]
	for i in range(p_real.shape[0]):
		linestyle = linestyles[i]
		color = 'g'
		ax.plot(x, p_real[i, :], label='GT' + str(120 + i*20), linestyle=linestyle, color=color)
	plt.xlabel('Node (arbitrary order)')
	plt.ylabel("p value")
	plt.legend(loc='lower right')

	axins = zoomed_inset_axes(ax, 3, loc='upper left', bbox_to_anchor=(0.15, 0.9),bbox_transform=ax.figure.transFigure) # zoom-factor: 2.5,
	# axins = inset_axes(ax, 1,1 , loc=2,bbox_to_anchor=(0.2, 0.55))
	axins.plot(x, p_lgan, label=r'GLapGAN', c='b')
	axins.plot(x, p_van, label=r'baseline', c='r')
	for i in range(p_real.shape[0]):
		linestyle = linestyles[i]
		color = 'g'
		axins.plot(x, p_real[i, :], label='GT' + str(120 + i*20), linestyle=linestyle, color=color)

	x1, x2, y1, y2 = 1400, 2400, 0, 0.08 # specify the limits
	axins.set_xlim(x1, x2) # apply the x-limits
	axins.set_ylim(y1, y2) # apply the y-limits
	axins.set_facecolor((1, 0.75, 0.75))
	# axins.set_xscale('log')
	# axins.set_xlim([0, 5])

	mark_inset(ax, axins, loc1=1, loc2=3, fc="g", linewidth=1, ec="0.5")
	# plt.yticks(visible=False)
	plt.xticks(visible=False)
	plt.grid(False)
	fig.tight_layout()
	# fig.savefig(save_name)
	plt.show()


if __name__ == '__main__':
	t = bool(sys.argv[2])
	plot_pvalue(sys.argv[1], True)
