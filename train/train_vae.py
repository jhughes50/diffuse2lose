"""
    Jason Hughes and Tenzi Zhouga
    December 2024

    Script to train PatchVAE
"""
import torch

from models.vae import PatchVAE, VAELoss
from loaders.dataloader import OccupancyDataset
from utils.logger import Logger

from torch.utils.data import DataLoader
from torch.optim import Adam

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = OccupancyDataset("./data")
    dataloader = DataLoader(dataset, batch_size=32)

    vae = PatchVAE(8, 1, 64, embed_dim=64, latent_dim=128)
    vae_loss = VAELoss()

    epochs = 20

    logger = Logger()

    optimizer = Adam(vae.parameters(), lr=1e-4)

    vae.to(device)

    kl_weight = torch.linspace(0.0005, 1.0, steps = epochs*len(dataset))

    iter = 0
    for epoch in range(epochs):
        vae.train()
        train_loss = list()
        for examples, masks, labels in dataloader:
            print(labels.shape)
            labels = labels.float()
            img, mu, logvar = vae(labels.to(device))
            print(img.shape)
            reconstruction_loss, kl_loss = vae_loss.loss(img, labels.to(device), mu, logvar)
            loss = reconstruction_loss + kl_weight[iter] * kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter += 1
            train_loss.append(loss.item())
        logger.log_list(train_loss, epoch, "train_vae")

        print("Epoch %i, Current Loss %f" %(epoch, loss.item()))
    print("Training Complete")
