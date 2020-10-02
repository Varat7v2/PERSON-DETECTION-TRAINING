#!/bin/bash

# MODEL TRAINING
/home/ubuntu/jenab/anaconda3/envs/cvtf114/bin/python models/research/object_detection/legacy/train.py \
--logtostderr --clone_on_cpu=False \
--train_dir=/home/ubuntu/varat/PERSON_DETECTION/ckpt \
--pipeline_config_path=/home/ubuntu/varat/PERSON_DETECTION/data/ssd_mobilenetv2_person.config

# MODEL EVALUATION
/home/ubuntu/jenab/anaconda3/envs/cvtf114/bin/python models/research/object_detection/legacy/eval.py --logtostderr --clone_on_cpu=False \
--checkpoint_dir=/home/ubuntu/varat/PERSON_DETECTION/ckpt \
--eval_dir=/home/ubuntu/varat/PERSON_DETECTION/eval_dir \
--pipeline_config_path=/home/ubuntu/varat/PERSON_DETECTION/data/ssd_mobilenetv2_person.config

# # EXPORT FROZON GRAPH
# python object_detection/export_inference_graph.py \
#     --input_type image_tensor \
#     --pipeline_config_path /home/ubuntu/varat/PERSON_DETECTION/data/ssd_mobilenetv2_person.config \
#     --trained_checkpoint_prefix /home/ubuntu/varat/PERSON_DETECTION/ckpt/model.ckpt-15134 \
#     --output_directory /home/ubuntu/varat/PERSON_DETECTION/converted_models/frozen_graph_15143

# # EXPORT TFLITE SSD GRAPH
# python object_detection/export_tflite_ssd_graph.py \
#     --pipeline_config_path /home/ubuntu/varat/PERSON_DETECTION/data/ssd_mobilenetv2_person.config \
#     --trained_checkpoint_prefix /home/ubuntu/varat/PERSON_DETECTION/ckpt/model.ckpt-100000 \
#     --output_directory /home/ubuntu/varat/PERSON_DETECTION/converted_models/tflite_model

# # CONVERT TO TFLITE
# tflite_convert \
# --graph_def_file=/home/ubuntu/varat/PERSON_DETECTION/converted_models/tflite_model/tflite_graph.pb \
# --output_file=/home/ubuntu/varat/PERSON_DETECTION/converted_models/tflite_model/person_detection.tflite \
# --output_format=TFLITE \
# --input_arrays=normalized_input_image_tensor \
# --input_shapes=1,300,300,3 \
# --inference_type=FLOAT \
# --output_arrays="TFLite_Detection_PostProcess,TFLite_Detection_PostProcess:1,TFLite_Detection_PostProcess:2,TFLite_Detection_PostProcess:3" \
# --allow_custom_ops

