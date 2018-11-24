import tensorflow as tf
import numpy as np
import os
import time
from lib.data_helpers import *
from lib.metrics import *
from lib.helpers import *

class LGAN:
	def __init__(self, sess, model_name, dim_x, dim_z, prob, options, alpha, batch_size):
		self.sess = sess
		self.prob = prob
		self.alpha = alpha
		self.batch_size = batch_size
		self.model_types = options
		self.model_name = model_name
		self.global_step = 0

		self.dim_x = dim_x
		self.dim_z = dim_z # Noise size 500 - 1000
		self.dim_h = dim_z * 2
		self.dim_h2 = self.dim_h * 2
		self.build_model()

	def generator(self, z, reuse=False, net_type='dc'):
		with tf.variable_scope("generator") as scope:
			if reuse:
				scope.reuse_variables()
			h1 = tf.layers.dense(inputs=z, units=self.dim_h, activation=tf.nn.relu, name='g_h1')
			# b1 = tf.layers.batch_normalization(inputs=h1, training=True, name='g_b1')
			d1 = tf.layers.dropout(inputs=h1, rate=0.5, noise_shape=None, seed=None, training=True, name='g_d1')
			h2 = tf.layers.dense(inputs=d1, units=self.dim_h2, activation=tf.nn.relu, name='g_h2')
			# b2 = tf.layers.batch_normalization(inputs=h2, training=True, name='g_b2')
			d2 = tf.layers.dropout(inputs=h2, rate=0.5, noise_shape=None, seed=None, training=True, name='g_d2')
			logits = tf.layers.dense(inputs=d2, units=self.dim_x, activation=tf.nn.tanh, name='g_h3')
			# this makes values => [0, 1]
			return logits

	def discriminator(self, x, reuse=False, net_type='dc'):
		with tf.variable_scope("discriminator") as scope:
			if reuse:
				scope.reuse_variables()
			h1 = tf.layers.dense(inputs=x, units=self.dim_h2, activation=tf.nn.relu, name='d_h1')
			d1 = tf.layers.dropout(inputs=h1, rate=0.5, noise_shape=None, seed=None, training=False, name='d_d1')
			logits = tf.layers.dense(inputs=d1, units=1, activation=None, name='d_h3')
			return logits

	def build_model(self):
		self.Z = tf.placeholder(tf.float32, shape=[None, self.dim_z])
		self.X = tf.placeholder(tf.float32, shape=[None, self.dim_x])
		self.X_hat = self.generator(self.Z)
		# Model Variants
		# (1) Architecture: vanilla, DC
		# (2) Loss function: vanilla, LS
		# (3) Regularization: yes/no
		# (4) Lossy: Ambient or not
		############ Ambient ############
		print("== Model Configurtion ====")
		if self.model_types[0]: # ambient gan
			print('[--- Architecture: Ambient GAN')
			self.Y_hat = block_pixels(self.X_hat, self.prob)
			self.Y = block_pixels(self.X, self.prob)
		else:
			print('[--- Architecture: FC')
			self.Y_hat = self.X_hat
			self.Y = self.X
		###########
		self.D_real = self.discriminator(self.Y)
		self.D_fake = self.discriminator(self.Y_hat, reuse=True)
		########### Loss ################
		if self.model_types[1]:
			print('[--- Loss: Wasserstein-GP')
			self.D_loss_real = tf.reduce_mean(self.D_real)
			self.D_loss_fake = tf.reduce_mean(self.D_fake)
			self.D_loss = self.D_loss_fake - self.D_loss_real
			self.G_loss = -self.D_loss_fake

			# WGAN gradient penalty
			epsilon = tf.random_uniform(
				shape=[self.batch_size, 1],
				minval=0.,
				maxval=1.
			)
			self.epsilon = epsilon
			interpolates = self.epsilon * self.Y + ((1 - self.epsilon) * self.Y_hat)
			disc_interpolates = self.discriminator(interpolates, reuse=True)
			gradients = tf.gradients(disc_interpolates, [interpolates])[0]
			slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
			gradient_penalty = tf.reduce_mean((slopes-1)**2)
			self.D_loss += LAMBDA*gradient_penalty
		###### Regularization ##################
		if self.model_types[2]: # laplacian
			print('[--- Reg: Laplacian -- Alpha: {}'.format(self.alpha))
			self.lap_mat = tf.placeholder(tf.float32, shape=[self.dim_x, self.dim_x])
			# diff = self.Y - self.Y_hat
			# For smooth purpose
			diff = self.Y_hat
			l1 = tf.matmul(diff, self.lap_mat)
			l2 = l1 * diff
			l3 = tf.reduce_sum(l2, axis=1)
			self.G_loss_lmse = tf.reduce_mean(l3)
			self.G_loss = self.G_loss + self.alpha * self.G_loss_lmse
		###### trainable variables ###################
		t_vars = tf.trainable_variables()
		self.d_vars = [var for var in t_vars if 'd_' in var.name]
		self.g_vars = [var for var in t_vars if 'g_' in var.name]
		# Saver
		self.saver = tf.train.Saver(max_to_keep=1)


	def train(self, config, data, lap_matrix):
		data_size = len(data)
		data = gen_data(data, data_size, config.batch_size)

		D_solver = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(self.D_loss, var_list=self.d_vars)
		G_solver = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(self.G_loss, var_list=self.g_vars)
		try:
			tf.global_variables_initializer().run()
		except:
			tf.initialize_all_variables().run()

		# Directory
		if self.load(config.checkpoint_dir):
			print("[---An existing model FOUND in checkpoint!---]")
			print("[---Global epoch: {}]".format(self.global_step))
		else:
			mkdir(config.output_dir)
			mkdir(config.checkpoint_dir)
			mkdir(config.sample_dir, True)
			mkdir(config.input_dir, True)
			print("[---Model is NOT found in checkpoint! Initializing a new one---]")

		batch_idxs = min(data_size, config.train_size) // config.batch_size
		print("batch_size: {}".format(batch_idxs))
		counter = 0
		start_time = time.time()
		# from IPython import embed; embed()
		for epoch in range(config.epoch):
			for it in range(batch_idxs):
				for it_critic in range(5):
					X_mb = get_batch(data, config.batch_size, config.dataset)
					d_run = [D_solver, self.D_loss]
					# eps = np.random.uniform(
					# 		size=[self.batch_size, 1],
					# 		low=0.,
					# 		high=1 - 1/(epoch + 1.0)
					# 	)
					d_feed = {self.X: X_mb,
								self.Z: sample_z(config.batch_size, self.dim_z)}
					_, errD = self.sess.run(d_run, feed_dict=d_feed)
				# Generator update
				X_mb = get_batch(data, config.batch_size, config.dataset)
				g_run = [G_solver, self.G_loss]
				if self.model_types[2] or config.metrics == 1 or config.metrics == 3:
					g_feed = {self.X: X_mb,
							self.lap_mat: lap_matrix,
							self.Z: sample_z(config.batch_size, self.dim_z)}
				else:
					g_feed = {self.Z: sample_z(config.batch_size, self.dim_z)}
				_, errG = self.sess.run(g_run, feed_dict=g_feed)

				counter += 1
				if np.mod(counter, 20) == 1:
					print("Epoch: [{:2d}] [{:4d}/{:4d}] time: {:4.4f}, d_loss: {:.8f}, g_loss: {:.8f}".format(
						epoch, it, batch_idxs, time.time() - start_time, errD, errG))
					# synthetic samples
					# sam_feed = {self.X: X_mb, self.Z: sample_z(config.batch_size, self.dim_z)}
					# samples, inputs = self.sess.run([self.X_hat, self.X], feed_dict=sam_feed)
					# evaluation
					# if config.metrics == 1 or config.metrics == 3: # lmse
					# 	diff = np.mean(samples, axis=0) - np.mean(X_mb, axis=0)
					# 	lmse = np.dot(np.dot(np.transpose(diff), lap_matrix), diff) / self.dim_x
					# 	lmse = np.asscalar(lmse)
					# 	lsme_list.append(lmse)
					# 	# print('LMSE = {}'.format(lmse))
					# if config.metrics == 2 or config.metrics == 3: # snr
					# 	x = np.mean(scipy.stats.signaltonoise(samples, axis=1))
					# 	y = np.mean(scipy.stats.signaltonoise(inputs, axis=1))
					# 	linf = y - x
					# 	linf_list.append(linf)
						# print('SNR-Diff(XG)= {}'.format(linf))
					if config.plot == 1: # save image
						idx = counter // 100
						save_images(inputs[:4, :], config.image_size, config.input_dir, idx)
						save_images(samples[:4, :], config.image_size, config.sample_dir, idx)
					elif config.plot == 2: # save plot
						idx = counter // 100
						plot_hist_1(samples[0], data, config.sample_dir, idx)
			# add more checkpoint
			self.save(config.checkpoint_dir, epoch + self.global_step)

		# saver = tf.train.Saver()
		# saver.save(self.sess, self.model_name)

	def generate_sample(self, checkpoint_dir, mb_size):
		# use the former session
		if self.load(checkpoint_dir):
			print('==== Loading model ....')
			syn_sample = self.generator(self.Z, reuse=True)
			samples = self.sess.run(syn_sample,
							   feed_dict={self.Z: sample_z(mb_size, self.dim_z)})
			return samples
		else:
			print('==== NO model FOUND ====')
			return None

	def save(self, checkpoint_dir, step):
		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)

		self.saver.save(self.sess,
						os.path.join(checkpoint_dir, self.model_name),
						global_step=step)

	def load(self, checkpoint_dir):
		print(" [*] Reading checkpoints...")
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name_arr = ckpt.model_checkpoint_path.split('/')[-1]
			ckpt_name, step = ckpt_name_arr.split('-')
			if ckpt_name == self.model_name: # with ckpt
				self.saver.restore(self.sess, ckpt.model_checkpoint_path)
				self.global_step = int(step)
				return True
		return False
# end class
