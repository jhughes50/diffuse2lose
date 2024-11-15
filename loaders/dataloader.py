"""
    CIS 6800 Final Project

    Data loader for processed occupancy grids
"""
import os
import glob
#import torch

#from torch.data.utils import DataLoader, Dataset

class OccupancyDataset:#(Dataset):

    def __init__(self, path : str) -> None:
        # get all the files
        self.files_ = list()
        self.path_ = path
        self.folders_ = os.listdir(path)

        for folder in os.listdir(path):
            cpath = os.path.join(path,folder)
            self.files_ = self.files_ + [os.path.join(cpath,item) for item in os.listdir(cpath) if "labelTrainIds" in item]

    def __len__(self) -> int:
        return len(self.files_)

    def __get_item__(self, iter : int) -> torch.Tensor:
        # load an image 
        raw_img = cv2.imread(self.files_[iter])
        # mask the image

        # tensor transforms
    
        # return the image

        return

if __name__ == "__main__":
    OccupancyDataset("../data")
