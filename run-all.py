#!/bin/bash

import dataset_config as config
import sys, os

# if config.DATASET == 'widerPerson':
# 	# CSV GENERATION FROM ANNOTATIONS TEXT FILE FOR TFRECORD PREP
# 	os.system("python convert_txt2csv_annotations.py")

# if config.DATASET == 'cityPerson':
# 	# CSV GENERATION FROM ANNOTATIONS TEXT FILE FOR TFRECORD PREP
# 	os.system("python convert_json2csv_annotations.py")

# # SPLITTING DATA INTO TRAIN AND TEST SET BY GROUPS
# os.system("python myTfrecord_split.py")

# GENERATE TRAIN AND TEST TFRECORD FROM SPLITTED CSV FILE
os.system("python generate_tfrecord.py --images_path={} --csv_input={}  --output_path={}"
	.format(config.IMAGES_PATH, config.TRAIN_CSV, config.TRAIN_TFRECORD))
os.system("python generate_tfrecord.py --images_path={} --csv_input={}  --output_path={}"
	.format(config.IMAGES_PATH, config.VALID_CSV, config.VALID_TFRECORD))
os.system("python generate_tfrecord.py --images_path={} --csv_input={}  --output_path={}"
	.format(config.IMAGES_PATH, config.TEST_CSV, config.TEST_TFRECORD))

# # TEST GROUNDTRUTHS
# os.system("python groundtruths_test.py".format(config.GROUNDTRUTHS_TEST))