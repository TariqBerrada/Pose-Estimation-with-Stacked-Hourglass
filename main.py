import sys
import os
import numpy as np
import cv2

from hourglass import HourglassNet
from mpii_datagen import MPIIDataGen
from data_process import *

def inference(model_json, model_weights, img, threshold = 0.1, num_stack = 2, num_class = 16, tiny = False):
    if tiny:
        xnet = HourglassNet(num_classes = num_class, num_stacks = num_stack, num_channels = 128, inres = (192, 192), outres = (48, 48))
    else:
        xnet = HourglassNet(num_classes = num_class, num_stacks = num_stack, num_channels = 256, inres = (256, 256), outres = (64, 64))

    xnet.load_model(model_json, model_weights)

    out, scale = xnet.inference_file(img)

    keypoints = post_process_heatmap(out[0, :, :, :])
    #ignore_kps = ['plevis', 'thorax', 'head_top']
    ignore_kps = []
    kp_keys = MPIIDataGen.get_kp_keys()
    
    mkps = []
    for i, keypoint in enumerate(keypoints):
        if kp_keys[i] in ignore_kps:
            conf = 0.0
        else:
            conf = keypoint[2]
        mkps.append((4*scale[1]*keypoint[0], 4*scale[0]*keypoint[1], conf))

    frame = render_connections(cv2.imread(img), mkps, kp_keys, connections)
    frame = render_joints(frame, mkps, threshold)
    
    return frame

model_json = './trained_models/hg_s2_b1/net_arch.json'
model_weights = './trained_models/hg_s2_b1/weights_epoch96.h5'
confidenc_th = 0.1
input_image = './input/images/yoga.jpg'

connections = [(0, 1), (1, 2), (2, 6), (3, 6), (3, 4), (4, 5), (6, 7), (7, 8), (8, 9), (8, 12), (12, 11), (11, 10), (8, 13), (13, 14), (14, 15)]

frame = inference(model_json = model_json, model_weights = model_weights, img = input_image)
cv2.imwrite('output/images/yoga.jpg', frame)
cv2.imshow('output', frame)
cv2.waitKey()
