from helper import *
import numpy as np
from sklearn.metrics import roc_curve

def convert_
def plot_ttest(off_data, off_gan, alp):
	name_data = offdata2name(off_data)
	name_gan = offgan2name(off_gan)
	root_path = '../result'
	output_path = "{}/roc/{}".format(root_path, name_gan)
	plot_path = "{}/plt_{}_{}_{}.png".format(output_path, name_data, name_gan, alp)

	fig = plt.figure(figsize=(10, 7))
	for off_model in range(1, 4):
		name_model = offmodel2name(off_model)
		file_path = "{}/{}_{}_{}_{}.npy".format(output_path, name_data, name_gan, name_model, alp)
		arr = np.load(file_path)
		# idx = arr < 0.08
		# arr = arr[idx]
		plt.plot(range(len(arr)), sorted(arr), label=name_model)
	plt.legend()
	plt.xlabel('Node')
	plt.ylabel("p-value")
	plt.title('{} data - BH correction with {} - alpha {}'.format(name_data, name_gan, alp))
	# plt.show()
	fig.tight_layout()
	fig.savefig(plot_path, bbox_inches='tight')

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
