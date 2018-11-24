import tensorflow as tf
import numpy as np
from numpy import linalg
import scipy.io as sio
import sys
from model_aaai import LGAN
from lib.helpers import *
from lib.data_helpers import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

flags = tf.app.flags
# Setup
flags.DEFINE_integer("epoch", 3, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_float("prob", 0.9, "Probability of an edge being noise")
flags.DEFINE_float("alpha", 0.01, "Regularization tradeoff")
# Training
flags.DEFINE_integer("train_size", 5000, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 8, "The size of batch images [64]")
flags.DEFINE_integer("image_size", 28, "The size of image to use")
# Folders
flags.DEFINE_string("output_dir", "../output", "Output Dir")
flags.DEFINE_string("checkpoint_dir", "../output/checkpoint", "Checkpoint Dir")
flags.DEFINE_string("sample_dir", "../output/syn_sample", "Output sample dir")
flags.DEFINE_string("input_dir", "../output/real_sample", "Input sample dir")
flags.DEFINE_string("mesh_dir", "../output/mesh_sample", "Mesh sample dir")
# Data
flags.DEFINE_integer("dataset", 4, "1-mnist, 2-random graph, 3-12-wrap, 4-12-mesh, 5-12-sync-mesh")
flags.DEFINE_integer("metrics", 0, "1-LMSE, 2-SNR, 3-both")
flags.DEFINE_integer("plot", 0, "")

flags.DEFINE_integer("n_modes", 2, "")
flags.DEFINE_integer("dim_u_k", 20, "")
flags.DEFINE_integer("nc", 5, "")
flags.DEFINE_integer("m", 50, "")

FLAGS = flags.FLAGS

# GPU
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

if __name__ == '__main__':

	lap = (int(sys.argv[1]) == 1)
	FLAGS.alpha = float(sys.argv[2]) ### here
	data_ad = (sys.argv[3].lower() == 'ad')
	if data_ad:
		FLAGS.dataset = 41
	else:
		FLAGS.dataset = 42

	d_name = 'rebuttal_{}'.format(sys.argv[3])
	n_samples = 20000

	FLAGS.checkpoint_dir = "{}_{}_{}_{}/".format(FLAGS.checkpoint_dir, d_name, int(lap), FLAGS.alpha)
	model_types = [False, True, lap] # Ambient, Wasserstein-GP, Laplacian
	type_name = '{}{}{}'.format(int(model_types[0]), int(model_types[1]), int(model_types[2]))
	config_name = '{}'.format(FLAGS.alpha)
	model_name = '{}{}_{}_{}.ckpt'.format(d_name, FLAGS.dataset, type_name, config_name)
	sample_file = '../data/mesh/smooth_test/data/{}{}_{}_{}.npy'.format(d_name, FLAGS.dataset, type_name, config_name)

	data, data_size, lap_matrix = load_dataset(FLAGS, 1)
	print("Datasize:", data_size)

	dim_signal = np.shape(data)[1]
	try:
		with tf.Session(config=config) as sess:
			mygan = LGAN(sess, dim_x=dim_signal, dim_z=100, model_name=model_name, prob=FLAGS.prob, options=model_types, alpha=FLAGS.alpha,  batch_size=FLAGS.batch_size) # Negative here
			print("== Training {} ====".format(model_name))
			mygan.train(FLAGS, data, lap_matrix)
			print("== Sampling ====")
			samples = mygan.generate_sample(FLAGS.checkpoint_dir, n_samples)
			np.save(sample_file, samples)

	except KeyboardInterrupt:
		print('KeyboardInterrupt stopping ....')
		sess.close()
