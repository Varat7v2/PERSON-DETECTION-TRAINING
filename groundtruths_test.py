import pandas as pd
import cv2
import os, sys, glob
from tqdm import tqdm

import dataset_config as config

GROUNDTRUTHS_ANNOTS = '{}/data/{}_{}.csv'.format(config.DATASET, config.DATASET, config.GROUNDTRUTHS_TEST)
# IMAGES_PATH = '{}/Images'.format(config.DATASET)
if config.DATASET == 'widerPerson':
	IMAGES_PATH = config.IMAGES_PATH
if config.DATASET == 'cityPerson' or 'eurocityPerson':
	IMAGES_PATH = config.IMAGES_CONVERTED
OUTPUT_PATH = '{}/{}_processed'.format(config.DATASET, config.GROUNDTRUTHS_TEST)

if not os.path.exists(GROUNDTRUTHS_ANNOTS):
        os.makedirs(GROUNDTRUTHS_ANNOTS)

if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

groundtruths = pd.read_csv(GROUNDTRUTHS_ANNOTS)
GT_group = groundtruths.groupby('filename')

for x in tqdm(GT_group.groups):
	gt_group = GT_group.get_group(x)
	img =gt_group['filename'].tolist()[0]
	xmins = gt_group['xmin'].tolist()
	ymins = gt_group['ymin'].tolist()
	xmaxs = gt_group['xmax'].tolist()
	ymaxs = gt_group['ymax'].tolist()
	frame = cv2.imread(IMAGES_PATH + '/' + img)
	for xmin, ymin, xmax, ymax in zip(xmins, ymins, xmaxs, ymaxs):
		cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

	cv2.imwrite(OUTPUT_PATH + '/' + img, frame)

