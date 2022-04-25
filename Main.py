import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from DataLoader import FFHQ_Dataset
from torch.utils.data import DataLoader

TrainDataLoc = '/Users/javidabderezaei/Downloads/TransferToServer/Explicit-GAN-Project/FFHQ/Images_Combined'

## Hyperparameters
lr = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
Noise_Dim = 64
Image_Dim = 128
batch_size = 4
num_epoch = 10

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


disc = Discriminator(Image_Dim).to(device)
gen = Generator(Noise_Dim, Image_Dim).to(device)
transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307), (0.3081))]
)

dis_optim = torch.optim.Adam(disc.parameters(), lr=lr)
gen_optim = torch.optim.Adam(gen.parameters(), lr=lr)
criterion = nn.BCELoss()

DataLoading = FFHQ_Dataset(TrainDataLoc, transform=transforms)
IterationOfTheData = DataLoader(DataLoading, batch_size=batch_size, shuffle=True)

writer_fake = SummaryWriter(f"runs/GAN/fake")
Writer_real = SummaryWriter(f"runs/GAN/real")

for epochs in range(num_epoch):
    for Batch_Index, (Real_Image, _) in enumerate(IterationOfTheData):
        Real_Image = Real_Image.view(-1, Image_Dim).to(device)
        Noise_Rand = torch.rand(batch_size, Noise_Dim)

        gen_Fake_Image = gen(Noise_Rand)
        disc_Real_Image = disc(Real_Image)

        loss_disc_real = criterion(disc_Real_Image.view(-1), torch.ones(Image_Dim))

        disc_gen_Fake_Image = disc(gen_Fake_Image).detach()
        loss_disc_fake = criterion(disc_gen_Fake_Image.view(-1), torch.zeros(Image_Dim))
        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        disc.zero_grad()
        loss_disc.backward() #Or use retain_graph=True instead of detach in a few lines above here!
        dis_optim.step()

        loss_gen = criterion(disc_gen_Fake_Image, torch.ones(Image_Dim))
        gen.zero_grad()
        loss_gen.backward()
        gen_optim.step()

if Batch_Index == 0:
    print(f"eEpoch: [{epochs}/{num_epoch}] \ "
          f"Loss Discriminator: {loss_disc: 0.4f} & Loss Generator: {loss_gen: 0.4f}"
    )

    with torch.no_grad():
        fake_Image = gen(Noise_Rand)

