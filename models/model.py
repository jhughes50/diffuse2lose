"""
    Jason Hughes and Tenzi Zhouga
    December 2024

    Model class
"""

from diffusers import AutoencoderKL, UNet2DModel

class DiffusionModel:

    def __init__(self):
        self.vae_ = AutoencoderKL(in_channels=1, out_channels=1)
        self.unet_ = UNet2DModel(in_channels=1, out_channels=1)

    def parameters(self):
        return list(self.vae_.parameters()) + list(self.unet_.parameters())

    def __call__(self, inp : torch.Tensor, mask : torch.Tensor):
        return self.forward(inp, mask)

    def generate_masked_image(inp : torch.Tensor, mask : torch.Tensor) -> torch.Tensor:
        return inp * mask

    def forward(self, inp : torch.Tensor, mask : torch.Tensor):
        mask_inp = self.generate_masked_image(inp, mask)

        latents = self.vae_.encode(inp).latent_dist.sample()
        latents = latents * self.vae_.config.scaling_factor

        masked_latents = self.vae_.encode()
