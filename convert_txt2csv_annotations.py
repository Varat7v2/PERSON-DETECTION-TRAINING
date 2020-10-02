import os, sys
import glob 
import numpy as np
from tqdm import tqdm
import cv2
import tensorflow as tf
import dataset_config as config
import pandas as pd

bbox = list()
bboxes = list()
label = list()
labels = list()

for annotation_file in tqdm(glob.glob(config.ANNOTATION_PATH+'/*.txt')):
	image_name = annotation_file.split('/')[-1].split('.')[0]
	img = cv2.imread(config.IMAGES_PATH + '/' + image_name + '.jpg')

	# norm_img = tf.image.per_image_standardization(img)
	# print(norm_img)
	# sys.exit(0)

	file = open(annotation_file, 'r') 
	lines = file.readlines()
	lines = [np.asarray(line.rstrip().split(' '), dtype=np.uint).tolist() for line in lines[1:]]

	for line in lines:
		if config.TFRECORD and line[0] == 1:
			width = line[3] - line[1]
			height = line[4] - line[2]
			iclass = line[0]
			bbox = [image_name+'.jpg', width, height, 'person']
			bbox.extend(line[1:])
			bboxes.append(bbox)
		else:
			if line[0] == 1:
				box = line[1:]
				cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0,0,255), 2)
				bbox.append(box)

if config.TFRECORD:
	df = pd.DataFrame(bboxes, columns=['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax'])
	df.to_csv('{}/data/{}.csv'.format(config.DATASET, config.DATASET), index=False)
else:
	cv2.imwrite(config.OUTPUT_PATH + '/' + image_name + '_processd.jpg', img)
	bboxes.append(np.asarray(bbox))