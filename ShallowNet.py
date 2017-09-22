from __future__ import division
import os
import random
import math
import time
import numpy as np
import tensorflow as tf
from nets import *

class ShallowNet(object):
    def __init__(self):
	pass

    def train(self, opt):
		self.opt = opt
		self.build_train_graph()
		self.collect_summaries()
		with tf.name_scope("parameter_count"):
			parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in
				                             tf.trainable_variables()])
		self.saver = tf.train.Saver([var for var in tf.trainable_variables()] +
									[self.global_step],
									max_to_keep = 100,
									keep_checkpoint_every_n_hours = 0.3)
		sv = tf.train.Supervisor(logdir = opt.checkpoint_dir,
								 save_summaries_secs = 0,
								 saver = None)
		# All variables in the graph will be initialized within sv.managed_session(). 
		# Then Checkpoint the model and add summaries.
		with sv.managed_session() as sess:
			print('Trainable variables: ')
			for var in tf.trainable_variables():
				print(var.name)
			print("parameter_count = ", sess.run(parameter_count))
			if opt.continue_train:
				print("Resume training from previous checkpoint")
				checkpoint = tf.train.latest_checkpoint(opt.checkpoint_dir)
				self.saver.restore(sess,checkpoint)
			
			init_time = time.time()
			# Update training process in the loop.
			# Every step denotes processing one batch for once.
			for step in range(1, opt.max_steps):
				start_time = time.time()
				fetches = {
					"train": self.train_op,
					"global_step": self.global_step,
					"incr_global_step": self.incr_global_step
				}

				if step % opt.summary_freq == 0:
					fetches["loss"] = self.total_loss
					fetches["summary"] = sv.summary_op

				# Globle step increases after feeding a batch. It's identical to the value of step.
				results = sess.run(fetches)
				gs = results["global_step"]

				if step % opt.summary_freq == 0:
					sv.summary_writer.add_summary(results["summary"], gs)
					current_epoch = math.ceil(gs / opt.steps_per_epoch)
					step_cur_epoch = gs - (current_epoch - 1) * opt.steps_per_epoch
					print("Epoch: [%2d] [%5d/%5d] speed: %4.4f/it loss: %.3f time: %6.1f"
						  % (current_epoch, step_cur_epoch, opt.steps_per_epoch,
						  	 time.time() - start_time, results["loss"], time.time() - init_time))

				if step % opt.save_latest_freq == 0:
					self.save(sess, opt.checkpoint_dir, 'latest')

				if step % opt.steps_per_epoch == 0:
					self.save(sess, opt.checkpoint_dir, gs)
	
    def build_train_graph(self):
		opt = self.opt
		with tf.name_scope("load_data"):
			seed = random.randint(0, 2 ** 31 - 1)
			
			# Load the list of training files into queues.
			file_path_list = self.format_file_list(opt.data_root, 'dataset_train')
			image_path_queue = tf.train.string_input_producer(
				file_path_list['image_path_list'],
				seed = seed,
				shuffle = True)
			
			pose_path_queue = tf.train.string_input_producer(
				file_path_list['pose_path_list'],
				seed = seed,
				shuffle = True)
				
			# Load images.
			image_reader = tf.WholeFileReader()
			_, img_contents = image_reader.read(image_path_queue)
					
			# Some pre-processing steps.
			image_decoded = tf.image.decode_png(img_contents)
			tgt_image = tf.image.resize_images(image_decoded, [opt.img_height, opt.img_width])
			tgt_image = self.preprocess_image(tgt_image)			
			tgt_image = tf.reshape(tgt_image, [opt.img_height, opt.img_width, 3])
			
			# Load camera pose.
			pose_reader = tf.TextLineReader()
			_, pose_contents = pose_reader.read(pose_path_queue)
			record_temp = []
			for i in range(7):
				record_temp.append([1.])

			pose_vec = tf.decode_csv(pose_contents,
									 record_defaults = record_temp)
			pose_vec = tf.stack(pose_vec)
			cam_pose = tf.reshape(pose_vec, [1, 7])
			
			# Form training batches
			# Two contents below become 3-dim tensors, in which the 1st dim denotes the index of batch.
			tgt_image, cam_pose = tf.train.batch([tgt_image, cam_pose], 
												 batch_size = opt.batch_size)
			
		with tf.name_scope("predict_pose"):
			pred_pose, pose_net_endpoints = pose_prediction(tgt_image, is_training = True)

		with tf.name_scope("compute_loss"):
			tra_loss = 0
			rot_loss = 0
			total_loss = 0

			# Define loss function.
			tra_loss = tf.reduce_sum(tf.abs(cam_pose[:,:,:3] - pred_pose[:,:,:3]))
			#tra_loss = tf.reduce_sum((pred_pose[:, :, 7] + tf.scalar_mul(pred_pose[:, :, 7], tf.abs(cam_pose[:,:,:3] - pred_pose[:,:,:3]))))
			rot_loss = opt.weight * tf.reduce_sum(tf.abs(cam_pose[:,:,3:] - pred_pose[:,:,3:]))
			#rot_loss = tf.reduce_sum((pred_pose[:, :, 8] + tf.scalar_mul(pred_pose[:, :, 8], tf.abs(cam_pose[:,:,3:] - pred_pose[:,:,3:7]))))
			total_loss += tra_loss + rot_loss

		with tf.name_scope("train_op"):
			train_vars = [var for var in tf.trainable_variables()]
			op = tf.train.AdamOptimizer(opt.learning_rate, opt.beta)

			# Gradient descent.
			self.grads_and_vars = op.compute_gradients(total_loss,
													  var_list = train_vars)
			self.train_op = op.apply_gradients(self.grads_and_vars)

			# One global step is defined as training all images for one time.
			self.global_step = tf.Variable(0,
										   name = 'global_step',
										   trainable = False)
			self.incr_global_step = tf.assign(self.global_step,
											  self.global_step + 1)
		self.tra_loss = tra_loss
		self.rot_loss = rot_loss
		self.total_loss = total_loss
		self.pred_pose = pred_pose
		self.tgt_image = tgt_image
		self.opt.steps_per_epoch = int(len(file_path_list['image_path_list']) // opt.batch_size)
			
    def format_file_list(self, data_root, train_or_test):
		
		# Obtain the array of frames' name from the train (or test) list.
		with open(data_root + '/%s.txt' % train_or_test, 'r') as f:
			frames = f.readlines()
		image_path = [l.split(' ')[0] for l in frames[3:]]
		
		# Yield the name of image and pose file (generated in the pose_per_frame.py) per frame.
		image_list = [os.path.join(data_root, image_path[i])
						   for i in range(len(frames) - 3)]
		pose_list = [os.path.join(data_root, image_path[i]).replace('.png', '.txt')
						   for i in range(len(frames) - 3)]
		
		all_list = {}
		all_list['image_path_list'] = image_list
		all_list['pose_path_list'] = pose_list
		return all_list

    def collect_summaries(self):
		opt = self.opt
		tf.summary.scalar("total_loss", self.total_loss)
		tf.summary.scalar("tra_loss", self.tra_loss)
		tf.summary.scalar("rot_loss", self.rot_loss)
		tf.summary.image('image', self.tgt_image)
		tf.summary.histogram("tx", self.pred_pose[:, :, 0])
		tf.summary.histogram("ty", self.pred_pose[:, :, 1])
		tf.summary.histogram("tz", self.pred_pose[:, :, 2])
		tf.summary.histogram("rw", self.pred_pose[:, :, 3])
		tf.summary.histogram("rx", self.pred_pose[:, :, 4])
		tf.summary.histogram("ry", self.pred_pose[:, :, 5])
		tf.summary.histogram("rz", self.pred_pose[:, :, 6])
		for var in tf.trainable_variables():
			tf.summary.histogram(var.op.name + "/values", var)
		for grad, var in self.grads_and_vars:
			tf.summary.histogram(var.op.name + "/gradients", grad)
		
    def preprocess_image(self, image):
		# Assuming input image is uint8
		image = tf.image.convert_image_dtype(image, dtype = tf.float32)
		return image * 2. - 1.

    def save(self, sess, checkpoint_dir, step):
		model_name = 'model'
		print(" [*] Saving checkpoint to %s..." % checkpoint_dir)
		if step == 'latest':
			self.saver.save(sess, 
							os.path.join(checkpoint_dir, model_name + '.latest'))
		else:
			self.saver.save(sess,
							os.path.join(checkpoint_dir, model_name),
							global_step = step)

    def setup_predict(self,
					  img_height,
					  img_width,
					  mode,
					  batch_size = 1):
		self.img_height = img_height
		self.img_width = img_width
		self.mode = mode
		self.batch_size = batch_size
		if self.mode == 'pose':
			self.build_test_graph()

    def build_test_graph(self):
	    input_uint8 = tf.placeholder(tf.uint8, [self.batch_size,
                                     self.img_height, self.img_width, 3],
                                     name='raw_input')
	    input_mc = self.preprocess_image(input_uint8)
	    with tf.name_scope("pose_prediction"):
	    	pred_pose, endpoints = pose_prediction(input_mc)
	    self.inputs = input_uint8
	    self.pred_pose = pred_pose
	    self.epts = endpoints

    def predict(self, inputs, sess, mode = 'pose'):
    	fetches = {}
    	if mode == 'pose':
    		fetches['pose'] = self.pred_pose
    	results = sess.run(fetches, feed_dict = {self.inputs: inputs})
    	return results
