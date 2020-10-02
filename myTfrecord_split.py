import numpy as np
import pandas as pd
import os, sys
from tqdm import tqdm
import dataset_config as config

np.random.seed(1)
full_labels = pd.read_csv(config.CSV_TO_SPLIT)

# PRINT THE STATISTICS OF DATASETS
# print(full_labels.describe().transpose())

grouped = full_labels.groupby('filename')
grouped.apply(lambda x: len(x)).value_counts()

# SPLIT EACH FILE INTO A GROUP IN A LIST
gb = full_labels.groupby('filename')
grouped_list = [gb.get_group(x) for x in gb.groups]

print('Total no. of groups: {}'.format(len(grouped_list)))

TRAIN_SPLIT = 0.8
TEST_SPLIT = 0.4 #60% of splitted validation dataset
# First validation spit from entire dataset
TRAIN_INDEX = np.random.choice(len(grouped_list), size=int(TRAIN_SPLIT*len(grouped_list)), replace=False)
VALID_INDEX_1 = np.setdiff1d(list(range(len(grouped_list))), TRAIN_INDEX)
# Second test split from validation dataset
VALID_INDEX_2 = np.random.choice(len(VALID_INDEX_1), size=int(TEST_SPLIT*len(VALID_INDEX_1)), replace=False)
TEST_INDEX = np.setdiff1d(list(range(len(VALID_INDEX_1))), VALID_INDEX_2)


print( 'Total train groups: {}, Total validation groups: {}, Total test groups: {}'
	.format(len(TRAIN_INDEX), len(VALID_INDEX_2), len(TEST_INDEX)))

# SPLIT INTO TRAIN AND TEST SAMPLES
train = pd.concat([grouped_list[i] for i in TRAIN_INDEX])
valid = pd.concat([grouped_list[i] for i in VALID_INDEX_2])
test = pd.concat([grouped_list[i] for i in TEST_INDEX])

print('Train samples: {}, Validation samples: {}, Test samples: {}'
	.format(len(train), len(valid), len(test)))

# SAVE TO CSV FILE
train.to_csv(config.TRAIN_CSV, index=None)
valid.to_csv(config.VALID_CSV, index=None)
test.to_csv(config.TEST_CSV, index=None)

print('Successfully splitted dataset to train, validation and test!')