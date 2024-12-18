"""
    Jason Hughes and Tenzi Zhouga
    December 2024

    Script to train PatchVAE
"""
import torch
import torch.nn.functional as F

from models.vae import PatchVAE, VAELoss
from models.unet import UNet, UNetLoss
from loaders.dataloader import OccupancyDataset
from utils.logger import Logger

from torch.utils.data import DataLoader
from torch.optim import Adam

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = OccupancyDataset("./data")
    dataloader = DataLoader(dataset, batch_size=4)

    vae = PatchVAE(8, 1, 64, embed_dim=64, latent_dim=128)
    unet = UNet(128)
    vae_loss = VAELoss()

    epochs = 20

    logger = Logger()

    optimizer = Adam(unet.model.parameters(), lr=1e-4)

    vae.to(device)

    kl_weight = torch.linspace(0.0005, 1.0, steps = epochs*len(dataset))
    # TODO Load vae state dict here
    vae.load_state_dict(torch.load("./logs/vae/vae_epoch_19.pth"))
    iter = 0
    for epoch in range(epochs):
        vae.eval()
        unet.model.train()
        train_loss = list()
        for examples, masks, labels in dataloader:
            labels = labels.float()
            examples = examples.float()
            p = vae.patch_embed(examples.to(device))
            mu, logvar = vae.encode(p.unsqueeze(1).permute(0,3,1,2))
            latent_mask = vae.reparameterize(mu, logvar)
            
            p = vae.patch_embed(labels.to(device))
            mu, logvar = vae.encode(p.unsqueeze(1).permute(0,3,1,2))
            latent_x = vae.reparameterize(mu, logvar)

            masks = F.interpolate(masks,size=(64,128), mode="nearest")
            masks = masks.permute(0,3,1,2).to(device)

            latent_input = torch.cat([latent_x, masks, latent_mask], dim=1)
            output = unet(latent_input.permute(0,2,1,3).to(device))

            loss = output.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            iter += 1
            train_loss.append(loss.item())
        
        logger.log_list(train_loss, epoch, "train_unet")
        file_name = "./data/models/unet_epoch_%i" %epoch
        if (epoch+1) % 10 == 0:
            unet.model.save_pretrained(file_name)
        print("Epoch %i, Current Loss %f" %(epoch+1, loss.item()))
    print("Training Complete")
