"""
    Jason Hughes
    December 2024 

    A CNN for occupancy grid completion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class OccupancyNetLoss:

    def __init__(self) -> None:
        self.bce_ = nn.BCELoss()

    def __call__(self, probs : torch.Tensor, label : torch.Tensor) -> torch.Tensor:
        return self.bce_(probs, label)
    
    def loss(self, probs : torch.Tensor, label : torch.Tensor) -> torch.Tensor:
        """ compute the Binary Cross-Entropy Loss"""
        return self(probs, label)

class ONetOutput:

    def __init__(self, logits : torch.Tensor, probs : torch.Tensor) -> None:
        self.logits_ = logits
        self.probs_ = probs

    @property
    def logits(self) -> torch.Tensor:
        return self.logits_

    @property
    def probabilities(self) -> torch.Tensor:
        return self.probs_

    @property
    def classes(self) -> torch.Tensor:
        return torch.where(self.probs_>0.5, 1, 0)

class OccupancyNet(nn.Module):

    def __init__(self, input_dim : int = 64) -> None:
        super(OccupancyNet, self).__init__()

        self.relu_ = nn.ReLU()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        self.tconv4 = nn.ConvTranspose2d(64, 32, kernel_size=5, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        
        self.tconv5 = nn.ConvTranspose2d(32, 16, kernel_size=5, padding=1)
        self.bn5 = nn.BatchNorm2d(16)

        self.tconv6 = nn.ConvTranspose2d(16, 4, kernel_size=5, padding=1)
        self.bn6 = nn.BatchNorm2d(4)

        self.tconv7 = nn.ConvTranspose2d(4, 1, kernel_size=5, stride = 1, padding=2)

        self.sigmoid_ = nn.Sigmoid()


    def encode(self, x : torch.Tensor) -> torch.Tensor:
        # encoder
        x = self.relu_((self.bn1(self.conv1(x))))
        x = self.relu_((self.bn2(self.conv2(x))))
        x = self.relu_((self.bn3(self.conv3(x))))

        return x


    def decode(self, x : torch.Tensor) -> torch.Tensor:
        # decoder
        x = self.relu_(self.bn4(self.tconv4(x)))
        x = self.relu_(self.bn5(self.tconv5(x)))
        x = self.relu_(self.bn6(self.tconv6(x)))

        x = self.tconv7(x)

        return x


    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """ forward pass through network """
        x = self.encode(x)
        logits = self.decode(x)
        probs = self.sigmoid_(logits)
         
        return ONetOutput(logits, probs)

if __name__ == "__main__":
    """ Test script for debugging network architecture """
    rand1 = torch.bernoulli(torch.full((64,64), 0.5)).unsqueeze(0).unsqueeze(0)
    rand2 = torch.bernoulli(torch.full((64,64), 0.5)).unsqueeze(0)
    #rand = torch.cat([rand1, rand2], dim=0)

    onet = OccupancyNet()
    criterion = OccupancyNetLoss()
    
    output = onet(rand1)
    print(output.logits.shape)
    print(output.probabilities.shape)
    loss = criterion.loss(output.probabilities, rand2)
