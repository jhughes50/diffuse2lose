"""
    Jason Hughes and Tenzi Zhouga
    November 2024

    Training script to fine tune  stable diffusion inpainting model

"""
import torch

from loaders.dataloader import OccupancyDataset
from utils.logger import Logger
from diffusers import StableDiffusionInpaintPipeline
from diffusers import AutoencoderKL, UNet2DConditionModel

from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torchvision import transforms
from PIL import Image

import torch.nn.functional as F

def post_process(img : Image) -> torch.Tensor:
    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    img_tensor = transform(img)

    return img_tensor.requires_grad_(True)

if __name__ == "__main__":

    dataset = OccupancyDataset("./data")
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    logger = Logger()

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    #dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: ", device)

    model_id = "stabilityai/stable-diffusion-2-inpainting"
    pipeline = StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipeline.to(device)
    #unet = pipeline.unet
    #vae = pipeline.vae

    optimizer = AdamW(pipeline.unet.parameters(), lr=5e-5)

    num_epochs = 100

    for epoch in range(num_epochs):
        pipeline.unet.train()
        epoch_train_loss = []
        for example, mask, label in train_dataset:
            pred = pipeline(prompt="", image=example.to(device), mask_image=mask.to(device))
            pred = post_process(pred["images"][0])
            
            loss = F.mse_loss(pred, label.float())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_train_loss.append(loss.item())
        logger.log_list(epoch_train_loss, epoch, "train")

        pipeline.unet.eval()
        epoch_eval_loss = []
        for example, mask, label in val_dataset:
            pred = pipeline(prompt="", image=example.to(device), mask_image=mask.to(device))
            pred = post_process(pred["images"][0])
            
            loss = F.mse_loss(pred, label.float())
            epoch_eval_loss.append(loss)
        logger.log_list(epoch_eval_loss, epoch, "eval")

        avg_loss = sum(epoch_train_loss) / len(epoch_train_loss)
        logger.log_float(avg_loss)
        
        if epoch % 10 == 0:
            file_name = "./logs/unet_epoch_%i" %epoch
            pipeline.unet.save_pretrained(file_name)

        print(f"Epoch {epoch + 1}: Loss = {loss.item()}")
