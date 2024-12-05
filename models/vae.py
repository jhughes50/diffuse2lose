"""
    Jason Hughes and Tenzi Zhouga
    December 2024

    VAE object for pretraining
"""

import torch
import torch.nn.functional as F
from torch import nn
from .embed import PatchEmbed
from einops import rearrange, repeat


class VAELoss:

    def __init__(self) -> None:
        pass

    def loss(self, recon_image, original_image, mu, logvar):
        recon_loss = F.mse_loss(recon_image, original_image, reduction="sum")
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
   
        return recon_loss, kl_loss


class PatchVAE(nn.Module):
    def __init__(self, patch_size, img_channels, img_size, embed_dim=1024, latent_dim=512, stride=8):
        super(PatchVAE, self).__init__()

        self.patch_size = patch_size
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Patch embedding layer (Patchify the image)
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, stride=stride, channels=img_channels, embed_dim=embed_dim, bias=True)
        self.num_patches = self.patch_embed.num_patches

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(embed_dim, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.conv_mu = nn.Conv2d(128, latent_dim, kernel_size=3, stride=1, padding=1)
        self.conv_logvar = nn.Conv2d(128, latent_dim, kernel_size=3, stride=1, padding=1)

        # Decoder
        self.decoder_input = nn.Conv2d(latent_dim, 128, kernel_size=3, stride=1, padding=1)
        self.decoder = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, embed_dim, kernel_size=3, stride=1, padding=1),
        )

    def encode(self, patches):
        x = self.encoder(patches)
        mu = self.conv_mu(x)
        logvar = self.conv_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        patch_recon = self.decoder(self.decoder_input(z))
        return rearrange(patch_recon, 'b c 1 p -> b p c')  # Back to (B, num_patches, embed_dim)

    def forward(self, x):
        p = self.patch_embed(x)
        mu, logvar = self.encode(p.unsqueeze(1).permute(0,3,1,2))
        z = self.reparameterize(mu, logvar)
        r = self.decode(z)
        recon_image = self.patch_embed.reconstruct(r, img_size=self.img_size)

        return recon_image, mu, logvar

    def sample(self, num_samples):
        with torch.no_grad():
            z_batch = torch.randn(num_samples, self.latent_dim, 1, self.num_patches)
            recon = self.decode(z_batch.to(self.device))
            sample_images = self.patch_embed.reconstruct(recon, img_size=self.img_size)
        return sample_images

if __name__ == "__main__":
    vae = PatchVAE(8, 1, 64, embed_dim=64, latent_dim=128)
    x = torch.rand(64,64).unsqueeze(0).unsqueeze(0)

    img, mu, logvar = vae(x)

