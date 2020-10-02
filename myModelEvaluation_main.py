import os, glob, sys
import tensorflow as tf
import numpy as np
import cv2
from tqdm import tqdm
import random
import pandas as pd
import pickle
import dataset_config as config

from mymodel_evaluation import myMODEL_EVALUATION

SERIALIZE_DICT_FIRST = True
PATH_MODEL_300x300 = config.MODEL_PATH

def main():
    model_evaluator = myMODEL_EVALUATION(PATH_MODEL_300x300)
    if not os.path.exists(config.OUTPUT_PATH):
        os.makedirs(config.OUTPUT_PATH)
    
    if SERIALIZE_DICT_FIRST:
        test_annotations = pd.read_csv(config.TEST_ANNOTATIONS_PATH)
        test_filenames = test_annotations['filename']

        gb = test_annotations.groupby('filename')
        grouped_list = [gb.get_group(x) for x in gb.groups]

        gt_dict = dict()
        pred_dict = dict()
        image_results = dict()

        for image in tqdm(grouped_list):
            img = image.iloc[0]['filename']
            name = img.split('/')[-1]
            frame = cv2.imread(os.path.join(config.TEST_IMAGES_PATH, img))

            # frame_width
            gt_boxes = list()
            pred_boxes_updated = list()
            gt_boxes_updated = list()
            pred_boxes_leftover = list()
            gt_boxes_leftover = list()
            
            for i in range(len(image['filename'])):
                xmin = image.iloc[i]['xmin']
                ymin = image.iloc[i]['ymin']
                xmax = image.iloc[i]['xmax']
                ymax = image.iloc[i]['ymax']
                box_width = xmax - xmin
                box_height = ymax - ymin
                box_area = box_width * box_height

                # TODO: GIVE CONSTRAINT ON BOX ARED TO REMOVE SMALLER BOXES
                # if box_area > 1/10 * 
                gt_boxes.append([xmin, ymin, xmax, ymax])
                # cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,255,0), 8)
                # cv2.putText(frame, '{}'.format(i), (xmin, ymin-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            pred_boxes = model_evaluator.detect_frozen_graph(frame, name)
            # print(gt_boxes)
            # print(pred_boxes)

            for gtidx, gt_box in enumerate(gt_boxes):
                ious = list()
                # flag = False
                if len(pred_boxes) > 0:
                    for predidx, pred_box in enumerate(pred_boxes):
                        ious.append(model_evaluator.calc_iou(gt_box, pred_box))
            
                    if max(ious) > 0.5:
                        match_index = np.argmax(ious)
                        match_box = pred_boxes[match_index]
                        gt_boxes_updated.append(gt_box)
                        # gt_boxes.pop(gtidx)
                        pred_boxes_updated.append(match_box)
                        pred_boxes.pop(match_index)
                        # flag = True
                    else:
                        gt_boxes_leftover.append(gt_box)    
                else:
                    gt_boxes_leftover.append(gt_box)

            gt_boxes_updated.extend(gt_boxes_leftover)
            pred_boxes_set = set(tuple(elem) for elem in pred_boxes)
            pred_boxes_updated_set = set(tuple(elem) for elem in pred_boxes_updated)
            set_diff = pred_boxes_set.difference(pred_boxes_updated_set)
            pred_boxes_updated.extend(list(list(elem) for elem in set_diff))

            print('Ground truths:')
            print(gt_boxes_updated)
            print('Predictions:')
            print(pred_boxes_updated)

            for gtidx, gtbox in enumerate(gt_boxes_updated):
                cv2.rectangle(frame, (gtbox[0], gtbox[1]), (gtbox[2], gtbox[3]), (0,255,0), 1)
                # cv2.putText(frame, '{}'.format(gtidx), (gtbox[0], gtbox[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            for predidx, predbox in enumerate(pred_boxes_updated):
                cv2.rectangle(frame, (predbox[0], predbox[1]), (predbox[2], predbox[3]), (255,0,0), 1)
                # cv2.putText(frame, '{}'.format(predidx), (predbox[2]-20, predbox[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            

            # cv2.putText(frame, '{}'.format(predidx), (predbox[2]-20, predbox[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            # cv2.putText(frame, '{}'.format(predidx), (predbox[2]-20, predbox[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imwrite(config.OUTPUT_PATH+'/'+name, frame)

            image_results[name] = model_evaluator.get_single_image_results(gt_boxes_updated, pred_boxes_updated)
            # print(image_results)

            gt_dict[img] = gt_boxes_updated
            pred_dict[img] = pred_boxes_updated

        model_evaluator.serialize_dict(gt_dict, config.SERIALIZE_GROUNDTRUTHS)
        model_evaluator.serialize_dict(pred_dict, config.SERIALIZE_PREDICTIONS)

    # else:
    #     gt_dict = model_evaluator.deserialize_dict('data/ground_truth_dict.pkl')
    #     pred_dict = model_evaluator.deserialize_dict('data/model_pred_dict.pkl')

    #     for img, gt_boxs in tqdm(gt_dict.items()):
    #         name = img.split('/')[-1]
    #         frame = cv2.imread(os.path.join(config.TEST_IMAGES_PATH,img))
    #         for gtidx, gtbox in enumerate(gt_boxs):
    #             cv2.rectangle(frame, (gtbox[0], gtbox[1]), (gtbox[2], gtbox[3]), (0,255,0), 4)
    #             # cv2.putText(frame, '{}'.format(gtidx), (gtbox[0], gtbox[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    #         cv2.imwrite(config.OUTPUT_PATH+'/'+img, frame)
        
    #     for img, pred_boxs in tqdm(pred_dict.items()):
    #         frame = cv2.imread(os.path.join(config.OUTPUT_PATH,img))
    #         for predidx, predbox in enumerate(pred_boxs):
    #             cv2.rectangle(frame, (predbox[0], predbox[1]), (predbox[2], predbox[3]), (255,0,0), 3)
    #             # cv2.putText(frame, '{}'.format(predidx), (predbox[2]-20, predbox[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    #         cv2.imwrite(config.OUTPUT_PATH+'/'+img, frame)

    # CALCULATE PRECISION AND RECALL
    precision, recall = model_evaluator.calc_precision_recall(image_results)
    # f1_score = 2*(precision*recall)/(precision+recall)
    print('Precision= {:.2f}, Recall={:.2f}'.format(precision*100, recall*100))
    # print('F1-score= {:.2f}'.format(f1_score*100))

if __name__ == '__main__':
    main()

# READ IMAGE WITH TENSORFLOW
# Tensorflow Image Decode
# IMG_WIDTH, IMG_HEIGHT = 640, 480
# def decode_img(img):
#   # convert the compressed string to a 3D uint8 tensor
#   img = tf.image.decode_jpeg(img, channels=3)
#   # Use `convert_image_dtype` to convert to floats in the [0,1] range.
#   img = tf.image.convert_image_dtype(img, tf.float32)
#   # resize the image to the desired size.
#   return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])

# for dir in os.listdir(config.TEST_IMAGES_PATH): #(Uncomment for oringinal test datasets)
#     for img_path in tqdm(glob.glob(os.path.join(config.TEST_IMAGES_PATH,dir)+'/*.jpg')):
#         filenames.append(img_path)
#         # img = tf.io.read_file(img_path)
#         # img = decode_img(tf.io.read_file(img_path))
#         frame = cv2.imread(img_path)

#         im_height, im_width, im_channel = frame.shape
#         # image = cv2.flip(image, 1)