"""
    Jason Hughes
    December 2024
    
    Train CNN model script
"""
import sys
import torch 

sys.path.append("/home/jason/")

from loaders.dataloader import OccupancyDataset
from utils.logger import Logger
from models.onet import OccupancyNet, OccupancyNetLoss

from torch.utils.data import DataLoader, random_split
from torch.optim import Adam


if __name__ == "__main__":
    
    dataset = OccupancyDataset("/home/jason/data")
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[ONET] Training on device: ", device)

    onet = OccupancyNet()
    onet.to(device)
    criterion = OccupancyNetLoss()

    optimizer = Adam(onet.parameters(), lr=1e-4)
    epochs = 40

    train_dataloader = DataLoader(train_dataset, batch_size=16)
    val_dataloader = DataLoader(val_dataset, batch_size=16)

    logger = Logger("/home/jason/logs/onet/")

    for epoch in range(epochs):
        onet.train()
        train_epoch_loss = list()
        for example, mask, label, in train_dataloader:
            example = example.float()
            label = label.float()
            output = onet(example.to(device))

            loss = criterion.loss(output.probabilities.cpu(), label)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_epoch_loss.append(loss.item())
        logger.log_list(train_epoch_loss, epoch, "train_onet")

        onet.eval()
        val_epoch_loss = list()
        with torch.no_grad():
            for example, mask, label in val_dataloader:
                example = example.float()
                label = label.float()
                output = onet(example.to(device))

                loss = criterion.loss(output.probabilities.cpu(), label)
                val_epoch_loss.append(loss.item())
        logger.log_list(val_epoch_loss, epoch, "val_onet")
        
        if (epoch+1) % 10 == 0:
            print("saving")
            file_name = "/home/jason/logs/onet/onet_epoch_%i.pth" %(epoch+1)
            torch.save(onet.state_dict(), file_name)
            

