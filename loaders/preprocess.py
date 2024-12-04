"""
    CIS 6800 Final Project

    Proprocess the data to turn it into an occupancy grid
"""

import os 
import yaml
import numpy as np

class OccupancyPreProcessor:

    def __init__(self) -> None:
        
        with open('/home/jason/loaders/class_map.yaml','r') as f:
            self.class_map_ = yaml.safe_load(f)

        with open('/home/jason/loaders/occupancy.yaml','r') as f:
            self.occupancy_ = yaml.safe_load(f)



    def color2Class(self, img : np.ndarray) -> np.ndarray:
        mat = np.zeros((img.shape[0], img.shape[1]))
        for cls in self.class_map_.keys():
            mask = np.all(img == self.class_map_[cls]['code'], axis=-1).astype(np.uint8)
            mask = mask * cls
            mat += mask

        return mat

    def class2Occupancy(self, mask : np.ndarray) -> np.ndarray:
        #free_grid = np.where(np.isin(mask,self.occupancy_["free_int"]))
        #grid = np.where(np.isin(free_grid,self.occupancy_["occupied_int"]), 0)
        grid = np.isin(mask, self.occupancy_["free_int"])

        return grid

    def makeOccupancyViz(self, grid : np.ndarray) -> np.ndarray:
        grid = np.where(grid == 1, 255, grid)
        grid = np.where(grid == 2, 128, grid)

        return grid
