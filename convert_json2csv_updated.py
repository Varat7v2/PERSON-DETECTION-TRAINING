import json 
import os, sys
import glob
from tqdm import tqdm
import cv2
import pandas as pd
import dataset_config as config

# to save csv file all together
csv_file = list()

if not os.path.exists(config.IMAGES_CONVERTED):
        os.makedirs(config.IMAGES_CONVERTED)
        
for mydir in os.listdir(config.ANNOTATION_PATH):
	# # to save csv file for train and test separately
	# csv_file = list()
	for subdir in os.listdir(os.path.join(config.ANNOTATION_PATH, mydir)):
		for jsonfile in tqdm(glob.glob(os.path.join(config.ANNOTATION_PATH, mydir, subdir) + '/*.json')):
			city = jsonfile.split('/')[-2]
			if config.DATASET == 'cityPerson':
				img_png = jsonfile.split('/')[-1].split('.')[0][:-18] + '_leftImg8bit.png'
			if config.DATASET == 'eurocityPerson':
				img_png = jsonfile.split('/')[-1].split('.')[0] + '.png'
			
			img_jpg = img_png.split('.')[0]+'.jpg'
			frame = cv2.imread(os.path.join(config.IMAGES_PATH, mydir, subdir) + '/' + img_png)
			# frame = cv2.imread(config.IMAGES_PATH + '/' + img_jpg)
			# frame_height, frame_width, _ = frame.shape
			# aspect_ratio = frame_height / frame_width
			# frame_resized = cv2.resize(frame, (640, int(640*aspect_ratio)), interpolation = cv2.INTER_AREA)
			# frame_resized = cv2.resize(frame, (300, 300), interpolation = cv2.INTER_AREA)
			# resized_height, resized_width, _ = frame_resized.shape
			# # print(resized_height, resized_width)
			# # sys.exit(0)
			cv2.imwrite(config.IMAGES_CONVERTED + '/' + img_jpg, frame)
			
			# Opening JSON file 
			f = open(jsonfile) 
			data = json.load(f)
			f.close()

			for idict in data['objects']:
				bbox = idict['bbox']
				label = idict['label']
				if label == 'pedestrian' #and ((bbox[1]+bbox[3])-bbox[1]) > (1/10 * data['imgHeight']):
					mydict = dict()
					mydict['filename'] = img_jpg
					
					# ORIGINAL IMAGE PROPERTIES
					mydict['xmin'] = bbox[0]
					mydict['ymin'] = bbox[1]
					mydict['xmax'] = bbox[0] + bbox[2]
					mydict['ymax'] = bbox[1] + bbox[3]
					mydict['width'] = mydict['xmax'] - mydict['xmin']
					mydict['height'] = mydict['ymax'] - mydict['ymin']

					# # RESIZED IMAGES PROPERTIES
					# mydict['width'] = resized_width
					# mydict['height'] = resized_height
					# mydict['xmin'] = int(bbox[0] / frame_width * resized_width)
					# mydict['ymin'] = int(bbox[1] / frame_height * resized_height)
					# mydict['xmax'] = int((bbox[0] + bbox[2]) / frame_width * resized_width)
					# mydict['ymax'] = int((bbox[1] + bbox[3]) / frame_height * resized_height)
					
					mydict['class'] = 'person' #label
					# cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0,255,0), 2)
					# cv2.putText(frame, label, (bbox[0]-20, bbox[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

					csv_file.append(mydict)
			
	df = pd.DataFrame.from_dict(csv_file)
	df = df[['filename', 'width',	'height', 'class',	'xmin',	'ymin',	'xmax',	'ymax']]
	# df.to_csv('{}.csv'.format(mydir), index=False)
	df.to_csv('{}/data/{}.csv'.format(config.DATASET, config.DATASET), index=False)
	print('Successfully converted to csv!')