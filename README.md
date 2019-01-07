# ShallowNet implemented on TensorFlow
Author: Chao 
01/08/2017


# Prerequisites
This codebase was developed and tested with Tensorflow 1.0, Ubuntu 14.04.


# Prepare txt file for camera pose (for both train and test)
dataset_dir is the root path of the training dataset.

Example:
python pose_per_frame.py --dataset_dir=/home/USR/Downloads/DATASET --dataset_name=KingsCollege


# Train the network
data_root is the path of a ".txt" format listing.

Example:
python train.py --data_root=/home/USR/Downloads/DATASET/KingsCollege


# Test the network
Rretrieve the pre-trained model by modifying '0000'.

Example:
python predict.py --data_root=/home/USR/Downloads/DATASET/KingsCollege --output_dir=./pred_pose/ --ckpt_file=./checkpoints/model-0000


# Evaluate the prediction
Compare the predicted pose with groundtruth pose.

Example:
python evaluate.py --data_root=/home/USR/Downloads/DATASET/KingsCollege --output_dir=./pred_pose/

# Tensorboard
You can start a tensorboard session and visualize the training progress by opening http://0.0.0.0:8888 on your browser. 

Example:
tensorboard --logdir=./checkpoints --port=8888
