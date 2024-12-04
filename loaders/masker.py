"""
    CIS 6800 Final Project 

    Object to mask the unseen
    occupants
"""
import yaml
import numpy as np

class OccupancyMasker:

    def __init__(self):
        with open('/home/jason/loaders/occupancy.yaml', 'r') as f:
            self.occupancy_map_ = yaml.safe_load(f)
        self.occupied_int_ = self.occupancy_map_["occupied_int"]

    def mask(self, mask : np.ndarray) -> np.ndarray:
        # mask behind objects
        result = np.zeros_like(mask, dtype=np.uint8)
        for col in range(mask.shape[1]):
            one_indices = np.where(mask[:, col] == 1)[0]

            if len(one_indices) > 0:
                h = np.arange(0,one_indices[-1])
                result[h, col] = 2
                result[one_indices[-1], col] = 1
        
        return result
    
    def training_mask(self, img : np.ndarray) -> np.ndarray:
        mask = np.zeros_like(img, dtype='bool')
        indices = np.argwhere(img == 2)
        mask[indices[:,0], indices[:,1]] = 1 

        return mask
