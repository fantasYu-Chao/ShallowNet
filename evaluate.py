from __future__ import division
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=str, required=True, help="Specify the root path of the data.")
parser.add_argument("--output_dir", type=str, required=True, help="Specify the output directory.")
args = parser.parse_args()

def main():

    with open(args.data_root + '/dataset_test.txt', 'r') as f:
        frames = f.readlines()

    image_path = [x.split(' ')[0] for x in frames[3:]]
    tx_gt = [np.array(x.split(' ')[1]).astype(np.float32) for x in frames[3:]]
    ty_gt = [np.array(x.split(' ')[2]).astype(np.float32) for x in frames[3:]]
    tz_gt = [np.array(x.split(' ')[3]).astype(np.float32) for x in frames[3:]]
    qw_gt = [np.array(x.split(' ')[4]).astype(np.float32) for x in frames[3:]]
    qx_gt = [np.array(x.split(' ')[5]).astype(np.float32) for x in frames[3:]]
    qy_gt = [np.array(x.split(' ')[6]).astype(np.float32) for x in frames[3:]]
    qz_gt = [np.array(x.split(' ')[7]).astype(np.float32) for x in frames[3:]]

    with open(args.output_dir + '/pred_pose.txt', 'r') as f:
        frames = f.readlines()

	#image_path = [x.split(' ')[0] for x in frames[:]]
    tx_pre = [np.array(x.split(' ')[1]).astype(np.float32) for x in frames[:]]
    ty_pre = [np.array(x.split(' ')[2]).astype(np.float32) for x in frames[:]]
    tz_pre = [np.array(x.split(' ')[3]).astype(np.float32) for x in frames[:]]
    qw_pre = [np.array(x.split(' ')[4]).astype(np.float32) for x in frames[:]]
    qx_pre = [np.array(x.split(' ')[5]).astype(np.float32) for x in frames[:]]
    qy_pre = [np.array(x.split(' ')[6]).astype(np.float32) for x in frames[:]]
    qz_pre = [np.array(x.split(' ')[7]).astype(np.float32) for x in frames[:]]

    error_rate = []
    for i in range(len(image_path)):
    	error_rate.append((abs((tx_pre[i] - tx_gt[i]) / tx_gt[i]) + abs((ty_pre[i] - ty_gt[i]) / ty_gt[i]) +
    					  abs((tz_pre[i] - tz_gt[i]) / tz_gt[i]) + abs((qw_pre[i] - qw_gt[i]) / qw_gt[i]) +
    					  abs((qx_pre[i] - qx_gt[i]) / qx_gt[i]) + abs((qy_pre[i] - qy_gt[i]) / qy_gt[i]) +
    					  abs((qz_pre[i] - qz_gt[i]) / qz_gt[i])) / 7)

    error = sum(error_rate) / len(image_path)

    print("Error rate =", error)


main()