"""
    Jason Hughes and Tenzi Zhouga
    November 2024

    Training script to fine tune  stable diffusion inpainting model

"""

from loaders import OccupancyGridDataset

from diffusers import StableDiffusionPipeline

from torch.utils.data import DataLoader
from torch.optim import AdamW

if __name__ == "__main__":

    dataset = OccupancyGridDataset("./data")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model_id = "runwayml/stable-diffusion-inpainting"
    pipeline = StableDiffusionPipeline.from_pretrained(model_id)

    unet = pipeline.unet
    optimizer = AdamW(unet.parameters(), lr=5e-5)

    num_epochs = 10

    for epoch in range(num_epochs):
        unet.train()
        for batch in dataloader:
            input_img = batch["input"]
            label_img = batch["label"]

            loss = unet(input_img, label_img).loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f"Epoch {epoch + 1}: Loss = {loss.item()}")
