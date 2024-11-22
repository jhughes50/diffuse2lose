"""
    Authors: Jason Hughes and Tenzi Zhouga
    Date: November 2024

    Test script for to debug data loader
"""

import torch
import cv2
from dataloader import OccupancyDataset

if __name__ == "__main__":
    dataset = OccupancyDataset("/home/jason/data")
    print(len(dataset))
    for i, ds in enumerate(dataset):
        print("here")
        img, label = ds
        
        vimg = dataset.preprocessor.makeOccupancyViz(img.numpy())
        cv2.imwrite("viz%i.png" %i, vimg)
        
        if i == 10:
            break

