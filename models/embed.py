"""
    Jason Hughes and Tenzi Zhouga
    Decemeber 2024

    Patch Embedding model
"""

import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=8, stride=8, channels=3, embed_dim=128, bias=True):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.embed_dim = embed_dim
        self.img_size = img_size

        #print("channels", channels, "img size ", img_size)

        self.proj = nn.Conv2d(channels, embed_dim, kernel_size=patch_size, stride=stride, bias=bias)

        self.deconv = nn.ConvTranspose2d(embed_dim, channels, kernel_size=patch_size, stride=stride, bias=bias)

        H_out = (img_size - self.patch_size) // self.stride + 1
        W_out = (img_size - self.patch_size) // self.stride + 1
        self.num_patches = H_out * W_out

    def forward(self, x):
        batch_size = x.size(0)
        patches = self.proj(x)
        patches = patches.reshape(batch_size, self.embed_dim, self.num_patches).permute(0,2,1)

        return patches

    def reconstruct(self, patches, img_size):
        batch_size = patches.size(0)
        x = patches.permute(0,2,1).reshape(batch_size, self.embed_dim, int(torch.sqrt(torch.tensor(self.num_patches))),int(torch.sqrt(torch.tensor(self.num_patches))))
        reconstructed_image = self.deconv(x)

        return reconstructed_image

if __name__ == "__main__":
    x = torch.rand(64,64).unsqueeze(0).unsqueeze(0)
    print(x.shape)
    embedder = PatchEmbed(img_size=64, channels=1)

    p = embedder(x)
