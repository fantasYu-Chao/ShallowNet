from __future__ import division
import os
import scipy.misc
import numpy as np
import tensorflow as tf
from ShallowNet import ShallowNet

flags = tf.app.flags
flags.DEFINE_integer("batch_size", 1, "The size of a sample batch.")
flags.DEFINE_integer("img_height", 180, "Image height.")
flags.DEFINE_integer("img_width", 320, "Image width.")
flags.DEFINE_string("data_root", None, "The root path of the data.")
flags.DEFINE_string("output_dir", None, "Output directory.")
flags.DEFINE_string("ckpt_file", None, "Checkpoint file")
FLAGS = flags.FLAGS

def main():
	sn = ShallowNet()
	sn.setup_predict(FLAGS.img_height,
					 FLAGS.img_width,
					 'pose',
					 FLAGS.batch_size)
	saver = tf.train.Saver([var for var in tf.trainable_variables()])

	if not os.path.exists(FLAGS.output_dir):
		os.makedirs(FLAGS.output_dir)

	with open(FLAGS.data_root + '/dataset_test.txt', 'r') as f:
		frames = f.readlines()

		image_path = [x.split(' ')[0] for x in frames[3:]]

	with tf.Session() as sess:
		saver.restore(sess, FLAGS.ckpt_file)
		for i in range(len(image_path)):
			if i % 100 == 0:
				print('Prediction progress: [%d/%d]' % (i, len(image_path)))
			curr_img = scipy.misc.imread(os.path.join(FLAGS.data_root, image_path[i]))
			curr_img = scipy.misc.imresize(curr_img, (FLAGS.img_height, FLAGS.img_width))

			predict = sn.predict(curr_img[None, :, :, :], sess, mode = 'pose')
			predict_pose = predict['pose'][0]
			predict_pose = np.array(predict_pose)

			tx_pre = predict_pose[0,0]
			ty_pre = predict_pose[0,1]
			tz_pre = predict_pose[0,2]
			qw_pre = predict_pose[0,3]
			qx_pre = predict_pose[0,4]
			qy_pre = predict_pose[0,5]
			qz_pre = predict_pose[0,6]

			with open(FLAGS.output_dir + '/pred_pose.txt', 'a') as f:
				f.write('%s %f %f %f %f %f %f %f\n' % (image_path[i], tx_pre, ty_pre, tz_pre, \
					                                   qw_pre, qx_pre, qy_pre, qz_pre))



main()