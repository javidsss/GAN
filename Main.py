import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

## Hyperparameters
lr = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"

class Discriminator(nn.Module):
    def __init__(self, Image_Dim):
        super().__init__()

        self.disc = nn.Sequential(
            nn.Linear(Image_Dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()
            )

    def forward(self, x):
        return self.disc(x)

class Generator(nn.Module):
    def __init__(self, Noise_Dim, Image_Dim):
        super().__init__()

        self.gen = nn.Sequential(nn.Linear(Noise_Dim, 256),
                                 nn.LeakyReLU(0.1),
                                 nn.Linear(256, Image_Dim),
                                 nn.Tanh(),
        )
    def forward(self, x):
        return self.gen(x)

