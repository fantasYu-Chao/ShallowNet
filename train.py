from __future__ import division
import os
import pprint
import random
import numpy as np
import tensorflow as tf
from ShallowNet import ShallowNet


flags = tf.app.flags
flags.DEFINE_string("data_root", "", "The root path of the data.")
flags.DEFINE_string("checkpoint_dir", "./checkpoints/",
                    "Directory name to save the checkpoints.")
flags.DEFINE_float("learning_rate", 0.0003, "Learning rate.") #0.0002
flags.DEFINE_float("beta", 0.9, "Momentum term.")
flags.DEFINE_float("weight", 600, "Weight for proportion between transformation and rotation.")
flags.DEFINE_integer("batch_size", 16, "The size of a sample batch.")
flags.DEFINE_integer("img_height", 180, "Image height.")
flags.DEFINE_integer("img_width", 320, "Image width.")
flags.DEFINE_integer("max_steps", 180000,
		     "Maximum number of training iterations.") # 30000
flags.DEFINE_integer("summary_freq", 100, "Logging every log_freq iterations.")
flags.DEFINE_integer("save_latest_freq", 5000,
		     "Save the latest model every save_latest_freq iterations \
		     (overwrites the previous latest model).")
flags.DEFINE_boolean("continue_train", False,
		     "Continue training from previous checkpoint")
FLAGS = flags.FLAGS

def main(_):
    seed = 8964
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    pp = pprint.PrettyPrinter()
    pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
	    os.makedirs(FLAGS.checkpoint_dir)

    sn = ShallowNet()
    sn.train(FLAGS)

if __name__ == '__main__':
    tf.app.run()
