#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# echo $DIR

if $DATASET == 'widerPerson':
	# CSV GENERATION FROM ANNOTATIONS TEXT FILE FOR TFRECORD PREP
	python convert_txt2csv_annotations.py

if $DATASET == 'cityPerson':
	# CSV GENERATION FROM ANNOTATIONS TEXT FILE FOR TFRECORD PREP
	python convert_json2csv_annotations.py

python convert_json2csv_annotations.py

# SPLITTING DATA INTO TRAIN AND TEST SET BY GROUPS
python myTfrecord_split.py

# GENERATE TRAIN AND TEST TFRECORD FROM SPLITTED CSV FILE
python generate_tfrecord.py --images_path=config.IMAGES_PATH --csv_input=config.TRAIN_CSV  --output_path=config.TRAIN_TFRECORD
python generate_tfrecord.py --images_path=config.IMAGES_PATH --csv_input=config.VALID_CSV  --output_path=config.VALID_TFRECORD
python generate_tfrecord.py --images_path=config.IMAGES_PATH --csv_input=config.TEST_CSV  --output_path=config.TEST_TFRECORD