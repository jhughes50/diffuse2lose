#!/usr/bin/env python3

import cv2
import numpy as np
import glob

# colors are bgr
CLASS_LUT = [
    np.array([255,0,0]),   # road
    np.array([0,255,0]),   # tree
    np.array([0,0,255]),   # building
    np.array([0,100,0]),   # grass
    np.array([255,255,0]), # car
    np.array([255,0,255]), # human
    np.array([100,100,0])  # gravel
]

def convert_img(path):
    label_color = cv2.imread(path).astype(np.int)
    label_index = np.ones(label_color.shape[:2], dtype=np.uint8)*255

    for ind, color in enumerate(CLASS_LUT):
        label_index[np.linalg.norm(label_color-color, axis=2) < 50] = ind
    return label_index

def viz_label(label):
    label_color = np.zeros((*label.shape, 3), dtype=np.uint8)
    for ind, color in enumerate(CLASS_LUT):
        label_color[label == ind, :] = np.array([color[0], color[1], color[2]])
    return label_color

if __name__ == '__main__':
    for label_img_path in glob.glob('labels/*.png'):
        filename = label_img_path.split('/')[-1].split('.')[0]
        print(filename)

        label = convert_img(label_img_path)
        label_viz = viz_label(label)
        cv2.imwrite('labels/'+filename+'_labelTrainIds.png', label)
        cv2.imwrite('labels/'+filename+'_labelTrainIds_viz.png', label_viz)
