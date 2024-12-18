"""
    Jason Hughes and Tenzi Zhuoga
    December 2024

    Script to validate our own unet model
"""
import cv2
import torch

from diffusers import UNet2DModel
from diffusers import DDPMScheduler

from models.vae import PatchVAE
from models.unet import UNet
from loaders.dataloader import OccupancyDataset
from torch.utils.data import random_split

import torch.nn.functional as F

if __name__ == "__main__":
    
    dataset = OccupancyDataset("./data")

    vae = PatchVAE(8, 1, 64, embed_dim=64, latent_dim=128)
    vae.load_state_dict(torch.load("./logs/vae_epoch_19.pth"))

    unet = UNet(128)
    unet.model.from_pretrained("./data/models/unet_epoch_10")
    
    size = int(0.2 * len(dataset))
    temp = len(dataset) - size
    test_set, _ = random_split(dataset, [size, temp])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    noise_scheduler = DDPMScheduler(num_train_timesteps=1500)

    vae.to(device)

    for example, mask, label in test_set:
        # generate some images
        orig_masked_img = (label * mask).squeeze().numpy()
        label=label.unsqueeze(0).float()
        p = vae.patch_embed(label.to(device))
        mu, logvar = vae.encode(p.unsqueeze(1).permute(0,3,1,2))
        latent_x = vae.reparameterize(mu, logvar)

        mask = F.interpolate(mask.unsqueeze(0), size=(64,128), mode="nearest")
        mask = mask.permute(0,3,1,2).to(device)
        masked_latents = (latent_x * mask).to(device)
       
        latent_input = torch.cat([latent_x,masked_latents],dim=1)

        noisy_pred = unet(latent_input.permute(0,2,1,3)).sample
        noisy_pred = noisy_pred.reshape(1,2,128,64)
        noisy_pred = torch.sum(noisy_pred, dim=1)
        
        decoded = vae.decode(noisy_pred.reshape(1,128,1,64))
        recon_img = vae.patch_embed.reconstruct(decoded.permute(0,2,1), img_size=64).reshape(64,64)
        img = recon_img.detach().cpu().numpy()
        
        # TODO add in what we know, i.e. the unmasked parts of the image
        #masked_ex = (label.cpu() * mask.cpu()).numpy()
        final_img = img #i+ orig_masked_img
        print(img.shape)
        print(orig_masked_img.shape)
        print(final_img.shape)
        #final_img = torch.where(example==2, 0, example).squeeze().numpy()
        cv2.imwrite("./logs/finalimg.png",final_img)
