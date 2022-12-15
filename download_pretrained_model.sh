#!/usr/bin/env bash

# dowload and unpack the pre-trained SSD Resnet 50 640x640 model
cd /home/workspace/experiments/pretrained_model/
wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz
tar -xvzf ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz
rm -rf ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz
