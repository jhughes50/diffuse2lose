"""
    CIS 6800 Final Project

    Proprocess the data to turn it into an occupancy grid
"""

import os 
import numpy as np

class OccupancyPreProcessor:

    def __init__(self) -> None:
        
        self.class_map = [
            np.array([255,0,0]),   # road
            np.array([0,255,0]),   # tree
            np.array([0,0,255]),   # building
            np.array([0,100,0]),   # grass
            np.array([255,255,0]), # car
            np.array([255,0,255]), # human
            np.array([100,100,0])  # gravel
        ]


    def color2Class(self, img : np.ndarray) -> np.ndarray:
        mat = np.zeros((img.shape[0], img.shape[1]))
        for i, cls in enumerate(self.class_map_):
            cls_mask = np.all(img == cls, axis=-1)
            mat += ((i+1) + mat)
