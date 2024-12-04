"""
    CIS 6800 Final Project

    Data loader for processed occupancy grids
"""
import os
import glob
import torch
import cv2
import numpy as np
from typing import Tuple
from torch.utils.data import DataLoader, Dataset

from .preprocess import OccupancyPreProcessor
from .masker import OccupancyMasker

class OccupancyDataset(Dataset):

    def __init__(self, path : str) -> None:
        # get all the files
        self.files_ = list()
        self.path_ = path
        self.folders_ = os.listdir(path)

        for folder in os.listdir(path):
            cpath = os.path.join(path,folder)
            self.files_ = self.files_ + [os.path.join(cpath,item) for item in os.listdir(cpath) if "labelTrainIds_viz" in item]

        self.preprocessor_ = OccupancyPreProcessor()
        self.masker_ = OccupancyMasker()

    @property
    def preprocessor(self) -> OccupancyPreProcessor:
        return self.preprocessor_

    def __len__(self) -> int:
        return len(self.files_)

    def __getitem__(self, iter : int) -> Tuple[torch.Tensor, torch.Tensor]:
        # load an image 
        raw_img = cv2.imread(self.files_[iter])
        raw_img = cv2.resize(raw_img, (64,64), interpolation=cv2.INTER_NEAREST)
        class_img = self.preprocessor_.color2Class(raw_img)
        grid = self.preprocessor_.class2Occupancy(class_img)

        # mask the image
        input_img = self.masker_.mask(grid)
        mask = self.masker_.training_mask(input_img)
        # tensor transforms
        label = torch.tensor(grid).unsqueeze(0)
        mask = torch.tensor(mask).to(torch.uint8).unsqueeze(0)
        exmpl = torch.tensor(input_img).unsqueeze(0)
        three_channel_exmpl = exmpl.repeat(3,1,1)
        three_channel_label = label.repeat(3,1,1)
        #return raw_img, class_img
        return (three_channel_exmpl, mask, three_channel_label)

if __name__ == "__main__":
    OccupancyDataset("../data")
