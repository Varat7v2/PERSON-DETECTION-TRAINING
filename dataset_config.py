import os, sys, glob

DATASET = 'eurocityPerson'
GROUNDTRUTHS_TEST = 'test'
# MODEL_PATH = 'models/frozen_graph_15143.pb'

ANNOTATION_PATH = '{}/Annotations'.format(DATASET)
IMAGES_PATH = '{}/Images'.format(DATASET)
IMAGE_FORMAT = 'jpg'
IMAGES_CONVERTED = '{}/Images_conv'.format(DATASET)
OUTPUT_PATH = '{}/processed_images'.format(DATASET)
TFRECORD = True
CSV_TO_SPLIT = '{}/data/{}.csv'.format(DATASET, DATASET)
TRAIN_CSV = '{}/data/{}_train.csv'.format(DATASET, DATASET)
VALID_CSV = '{}/data/{}_valid.csv'.format(DATASET, DATASET)
TEST_CSV = '{}/data/{}_test.csv'.format(DATASET, DATASET)
TRAIN_TFRECORD = '{}/data/{}_train.record'.format(DATASET, DATASET)
VALID_TFRECORD = '{}/data/{}_valid.record'.format(DATASET, DATASET)
TEST_TFRECORD = '{}/data/{}_test.record'.format(DATASET, DATASET)

SERIALIZE_GROUNDTRUTHS = '{}/data/ground_truth_dict.pkl'.format(DATASET)
SERIALIZE_PREDICTIONS = '{}/data/model_pred_dict.pkl'.format(DATASET)
