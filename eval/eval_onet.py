"""
    Jason Hughes
    December 2024

    Generate images from using onet occupancy diffusion
"""
import sys
import torch
import cv2
from torch import Tensor

sys.path.append("/home/jason/")
from loaders.dataloader import OccupancyDataset
from models.onet import OccupancyNet

from torch.utils.data import random_split

def post_process_image(gen_img : Tensor, label : Tensor, mask : Tensor) -> Tensor:
    masked_out_label = label * (-1*(mask-1))
    masked_out_gen   = gen_img * mask
    return masked_out_label + masked_out_gen

def calculate_iou(gen : Tensor, label : Tensor) -> float:
    intersection = torch.sum(gen & label)
    union = (torch.sum(gen) + torch.sum(label)) - intersection

    iou = intersection / union
    if union == 0:
        return -1.0
    else:
        return iou.item()

if __name__ == "__main__":
    torch.manual_seed(42)
    dataset = OccupancyDataset("/home/jason/data")
    onet = OccupancyNet()
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size

    # Split the dataset
    _, test_dataset = random_split(dataset, [train_size, test_size])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    onet.load_state_dict(torch.load("/home/jason/logs/onet/onet_epoch_40.pth"))
    onet.to(device)

    count = 0
    ious = list()
    for example, mask, label in test_dataset:
        output = onet(example.float().unsqueeze(0).to(device))

        final_img = post_process_image(output.classes.cpu(), label, mask)
        final_img = final_img.squeeze(0).permute(1,2,0)
        #final_img = final_img.detach().cpu().numpy() * 255
        
        label = label.permute(1,2,0)
        #label = label.numpy() * 255
        iou = calculate_iou(final_img, label)
        if iou != -1.0:
            ious.append(iou)
        
        #cv2.imwrite("/home/jason/logs/onet/gen_img_%i.png" %count, final_img)
        #cv2.imwrite("/home/jason/logs/onet/lbl_img_%i.png" %count, label)

        count += 1

    mean_iou = sum(ious) / len(ious)
    print("Mean IOU: ", mean_iou)
    print("Max IOU: ", max(ious))
    print("Min IOU: ", min(ious))
