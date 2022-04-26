import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from DataLoader import FFHQ_Dataset
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import matplotlib.pyplot as plt


TrainDataLoc = '/Users/javidabderezaei/Downloads/TransferToServer/Explicit-GAN-Project/FFHQ/Images_Combined'

## Hyperparameters
lr = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
Noise_Dim = 64
Image_Width = 28
Image_Height = 28
Num_ColorChannels = 1
Image_Dim_Total = Image_Width * Image_Height * Num_ColorChannels
batch_size = 32
num_epochs = 100

class Discriminator(nn.Module):
    def __init__(self, Image_Dim_Total):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(Image_Dim_Total, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1),
            nn.Sigmoid()
            )

    def forward(self, x):
        return self.disc(x)

class Generator(nn.Module):
    def __init__(self, Noise_Dim, Image_Dim_Total):
        super().__init__()

        self.gen = nn.Sequential(nn.Linear(Noise_Dim, 256),
                                 nn.LeakyReLU(0.01),
                                 nn.Linear(256, Image_Dim_Total),
                                 nn.Tanh(),
        )
    def forward(self, x):
        return self.gen(x)


disc = Discriminator(Image_Dim_Total).to(device)
gen = Generator(Noise_Dim, Image_Dim_Total).to(device)
transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Resize([Image_Width, Image_Height]), transforms.Normalize((0.5,), (0.5,))]
)

fixed_noise_For_Tensorboard = torch.randn((batch_size, Noise_Dim)).to(device)

disc_optim = torch.optim.Adam(disc.parameters(), lr=lr)
gen_optim = torch.optim.Adam(gen.parameters(), lr=lr)
criterion = nn.BCELoss()

# DataLoading = FFHQ_Dataset(TrainDataLoc, transform=transforms)
# IterationOfTheData = DataLoader(DataLoading, batch_size=batch_size, shuffle=True)

DataLoading = datasets.MNIST(root="/Users/javidabderezaei/Downloads/TransferToServer/Explicit-GAN-Project/MNIST_Data", transform=transforms, download=True)
IterationOfTheData = DataLoader(DataLoading, batch_size=batch_size, shuffle=True)

writer_fake = SummaryWriter(f"/Users/javidabderezaei/Downloads/TransferToServer/Explicit-GAN-Project/MNIST_Data/Tensorboard/runs/Fake")
writer_real = SummaryWriter(f"/Users/javidabderezaei/Downloads/TransferToServer/Explicit-GAN-Project/MNIST_Data/Tensorboard/runs/Real")
step = 0

for epoch in range(num_epochs):
    for Batch_Index, (Real_Image,_) in enumerate(IterationOfTheData):
        Real_Image = Real_Image.view(-1, Image_Dim_Total).to(device)
        Noise_Rand = torch.randn(batch_size, Noise_Dim).to(device)

        gen_Fake_Image = gen(Noise_Rand)
        disc_Real_Image = disc(Real_Image)

        loss_disc_real = criterion(disc_Real_Image.view(-1), torch.ones_like(disc_Real_Image).view(-1))

        disc_gen_Fake_Image = disc(gen_Fake_Image)
        loss_disc_fake = criterion(disc_gen_Fake_Image.view(-1), torch.zeros_like(disc_gen_Fake_Image).view(-1))
        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        disc.zero_grad()
        loss_disc.backward(retain_graph=True) #Retain graph is set to True, since we are disc(gen_Fake_Image) and performing another backprop below that involves disc(gen_Fake_Image)
        disc_optim.step()

        disc_gen_Fake_Image = disc(gen_Fake_Image)
        loss_gen = criterion(disc_gen_Fake_Image.view(-1), torch.ones_like(disc_gen_Fake_Image).view(-1))
        gen.zero_grad()
        loss_gen.backward()
        gen_optim.step()

        if Batch_Index % 500 == 0:
            print(f"Batch: [{Batch_Index}/{len(IterationOfTheData)}] \ "
                  f"eEpoch: [{epoch}/{num_epochs}] \ "
                  f"Loss Discriminator: {loss_disc: 0.4f} & Loss Generator: {loss_gen: 0.4f}"
            )

            with torch.no_grad():
                fake_Image = gen(fixed_noise_For_Tensorboard).reshape(-1, 1, Image_Height, Image_Width)
                real_image_resized = Real_Image.reshape(-1, 1, Image_Height, Image_Width)

                img_grid_fake = torchvision.utils.make_grid(fake_Image, normalize=True)
                img_grid_real = torchvision.utils.make_grid(real_image_resized, normalize=True)

                writer_fake.add_image(
                    "Fake Images", img_grid_fake, global_step=step
                )

                writer_real.add_image(
                    "Real Images", img_grid_real, global_step=step
                )

                step += 1