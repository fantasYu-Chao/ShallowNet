# ==================================================
# ShallowNet implemented on TensorFlow
# Author: Chao 
# 01/08/2017
# 
# Note: the variable dataset_dir should be the path
# where the dataset is stored, while the data_root
# should be the path where the list is stored.
# ==================================================

# Prerequisites
This codebase was developed and tested with Tensorflow 1.0, Ubuntu 14.04.


# Prepare txt file for camera pose (for both train and test)
# dataset_dir is the root path of the training dataset
python pose_per_frame.py --dataset_dir=/home/USR/Downloads/DATASET --dataset_name=KingsCollege


# Train the network
# data_root is the path of a ".txt" format listing
python train.py --data_root=/home/USR/Downloads/DATASET/KingsCollege


# Test the network (retrieve the pre-trained model by modifying '0000')
python predict.py --data_root=/home/USR/Downloads/DATASET/KingsCollege --output_dir=./pred_pose/ --ckpt_file=./checkpoints/model-0000

# Evaluate the prediction (compare the predicted pose with groundtruth pose)
python evaluate.py --data_root=/home/USR/Downloads/DATASET/KingsCollege --output_dir=./pred_pose/

# ======================================================
# You can start a tensorboard session 
# and visualize the training progress by opening http://0.0.0.0:8888 on your browser. 
# ======================================================
tensorboard --logdir=./checkpoints --port=8888
