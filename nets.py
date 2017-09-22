from __future__ import division
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import utils
import numpy as np

def pose_prediction(inputs, is_training = True):
	batch_norm_params = {'is_training': is_training}
	H = inputs.get_shape()[1].value
	W = inputs.get_shape()[2].value
	
	with tf.variable_scope('shallow_net') as sc:
		end_point_collection = sc.original_name_scope + '_end_points'
		with slim.arg_scope([slim.conv2d],
							normalizer_fn = slim.batch_norm,
							weights_regularizer = slim.l2_regularizer(0.05),
							normalizer_params = batch_norm_params,
							activation_fn = tf.nn.relu,
							outputs_collections = end_point_collection):
			# cnv1 to cnv_pred of pose prediction
			cnv1 = slim.conv2d(inputs, 16, [7, 7], stride = 2, scope = 'cnv1')
			cnv2 = slim.conv2d(cnv1, 32, [5, 5], stride = 2, scope = 'cnv2')
			cnv3 = slim.conv2d(cnv2, 64, [3, 3], stride = 2, scope = 'cnv3')
			cnv4 = slim.conv2d(cnv3, 128, [3, 3], stride = 2, scope = 'cnv4')
			cnv5 = slim.conv2d(cnv4, 256, [3, 3], stride = 2, scope = 'cnv5')
			cnv6 = slim.conv2d(cnv5, 256, [3, 3], stride = 2, scope = 'cnv6')
			cnv7 = slim.conv2d(cnv6, 256, [3, 3], stride = 2, scope = 'cnv7')
			dropout = slim.dropout(cnv7, 0.5, scope='dropout8')
			
			# pose_pred = slim.fully_connected(dropout, 7, activation_fn=None, scope='pred')
			pose_pred = slim.conv2d(dropout, 7, [1, 1], scope = 'pred',
					stride = 1, normalizer_fn = None, activation_fn = None)
			
			pose_avg = tf.reduce_mean(pose_pred, [1, 2])
			pose_final =  tf.reshape(pose_avg, [-1, 1, 7]) #0.01*
			
		end_points = utils.convert_collection_to_dict(end_point_collection)
		
		return pose_final, end_points
			
