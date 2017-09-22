from __future__ import division
import argparse
import os
import numpy as np
import math

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str, required=True, help="the root path of the data")
parser.add_argument("--dataset_name", type=str, required=True, help="the name of the dataset")
args = parser.parse_args()

def main():
	data_root = os.path.join(args.dataset_dir, args.dataset_name)

	# dataset_train.txt
	with open(data_root + '/dataset_train.txt', 'r') as f:
		next(f)
		next(f)
		next(f)
		
		for line in f:
			image_path, x, y, z, w, p, q, r = line.split()
			tx = float(x)
			ty = float(y)
			tz = float(z)
			qw = float(w)
			qx = float(p)
			qy = float(q)
			qz = float(r)
			
			per_frame_dir = data_root + '/' + image_path.replace('.png', '.txt')
			with open(per_frame_dir, 'w') as f:
				f.write('%f,%f,%f,%f,%f,%f,%f' % (tx, ty, tz, qw, qx, qy, qz) )

	# dataset_test.txt
	with open(data_root + '/dataset_test.txt', 'r') as f:
		next(f)
		next(f)
		next(f)

		for line in f:
			image_path, x, y, z, w, p, q, r = line.split()
			tx = float(x)
			ty = float(y)
			tz = float(z)
			qw = float(w)
			qx = float(p)
			qy = float(q)
			qz = float(r)
			
			per_frame_dir = data_root + '/' + image_path.replace('.png', '.txt')
			with open(per_frame_dir, 'w') as f:
				f.write('%f,%f,%f,%f,%f,%f,%f' % (tx, ty, tz, qw, qx, qy, qz) )

main()

def Quaternion_to_EulerianAngle(w, p, q, r):
	psqr = p * p

	t0 = +2.0 * (r * w + p * q)
	t1 = +1.0 - 2.0 * (w * w + psqr)
	pitch = math.degrees(math.atan2(t0, t1))

	t2 = +2.0 * (r * p - q * w)
	t2 = 1 if t2 > 1 else t2
	t2 = -1 if t2 < -1 else t2
	roll = math.degrees(math.asin(t2))

	t3 = +2.0 * (r * q + w *p)
	t4 = +1.0 - 2.0 * (psqr + q * q)
	yaw = math.degrees(math.atan2(t3, t4))

	return pith, roll, yaw
