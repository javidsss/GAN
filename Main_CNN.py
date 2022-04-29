import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from DataLoader import FFHQ_Dataset
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import sys

Model_Save = True
Model_Load = False

if Model_Load is True:
    ModelVersion_disc = "/Model_Save_DCGAN/Discriminator_Epoch0_BatchIdx60.pth"
    ModelVersion_gen = "/Model_Save_DCGAN/Generator_Epoch0_BatchIdx60.pth"

FFHQdataset = False
MNISTdataset = False
Celebdataset = True


if FFHQdataset == True:
    TrainDataLoc = '/Users/javidabderezaei/Downloads/TransferToServer/Explicit-GAN-Project/FFHQ/Images_Combined'
if MNISTdataset == True:
    TrainDataLoc = "/Users/javidabderezaei/Downloads/TransferToServer/Explicit-GAN-Project/MNIST_Data"
if Celebdataset == True:
    TrainDataLoc = '/Users/javidabderezaei/Downloads/TransferToServer/Explicit-GAN-Project/Celeb'

## Hyperparameters
lr = 2e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
Noise_Dim = 100
Image_Width = 64
Image_Height = 64
Num_ColorChannels = 3
batch_size = 128
num_epochs = 100
feature_d = 64
feature_g = 64
Num_Imgs_On_Tensorboard = 32



def ModelSave_func(model, optimization, loss, batch_num, epoch_num, path):
    checkpoint = {'epoch': epoch_num, 'State_dict': model.state_dict(), 'Optimizer': optimization.state_dict(), 'loss': loss, 'batch_num': batch_num}
    torch.save(checkpoint, path)

class Discriminator(nn.Module):
    def __init__(self, in_channels, feature_d):
        super(Discriminator, self).__init__() #Differentce between super(Discriminator) and super() should be checked!

        self.disc = nn.Sequential(
            nn.Conv2d(in_channels, feature_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            self.ConvBlock(feature_d, feature_d*2, kernel_size=4, stride=2, padding=1),
            self.ConvBlock(feature_d*2, feature_d*4, kernel_size=4, stride=2, padding=1),
            self.ConvBlock(feature_d*4, feature_d*8, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(feature_d*8, 1, kernel_size=4, stride=2, padding=0),
            nn.Sigmoid()
        )

    def ConvBlock(self, in_channels, out_channel, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channel, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2)
        )
    def forward(self, x):
        return self.disc(x)

class Generator(nn.Module):
    def __init__(self, Noise_Dim, in_channels_img, feature_g):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            self.ConvTransposeBlock(Noise_Dim, feature_g*16, kernel_size=4, stride=1, padding=0),
            self.ConvTransposeBlock(feature_g*16, feature_g*8, kernel_size=4, stride=2, padding=1),
            self.ConvTransposeBlock(feature_g*8, feature_g*4, kernel_size=4, stride=2, padding=1),
            self.ConvTransposeBlock(feature_g*4, feature_g*2, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(feature_g*2, in_channels_img, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def ConvTransposeBlock(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self, x):
        return self.gen(x)

def Initialize_Weight(Model):
    for m in Model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


disc = Discriminator(Num_ColorChannels, feature_d).to(device)
gen = Generator(Noise_Dim, Num_ColorChannels, feature_g).to(device)

Initialize_Weight(disc)
Initialize_Weight(gen)

disc_Optim = torch.optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))
gen_Optim = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
criterion = nn.BCELoss()

if Model_Load is True:
    ModelLocation_disc = TrainDataLoc + ModelVersion_disc
    Loaded_Checkpoint_disc = torch.load(ModelLocation_disc)
    disc.load_state_dict(Loaded_Checkpoint_disc['State_dict'])
    disc_Optim.load_state_dict(Loaded_Checkpoint_disc['Optimizer'])
    epoch_saved = Loaded_Checkpoint_disc['epoch']
    batch_saved = Loaded_Checkpoint_disc['batch_num']
    loss_disc = Loaded_Checkpoint_disc['loss']

    ModelLocation_gen = TrainDataLoc + ModelVersion_gen
    Loaded_Checkpoint_gen = torch.load(ModelLocation_gen)
    gen.load_state_dict(Loaded_Checkpoint_gen['State_dict'])
    gen_Optim.load_state_dict(Loaded_Checkpoint_gen['Optimizer'])
    loss_gen = Loaded_Checkpoint_gen['loss']

# def test(N, in_channel, Height, Width, Noise_dim):
#     InputDisc = torch.randn(N, in_channel, Height, Width)
#
#     disc = Discriminator(in_channel, feature_d=8)
#     Initialize_Weight(disc)
#     # assert disc(Input).shape == (N, 1, 1, 1)
#     TestDisc = disc(InputDisc)
#     print(F"Discriminator test: {TestDisc.shape}")
#
#     img_channel = in_channel
#     InputGen_Noise = torch.randn(N, Noise_dim, 1, 1)
#     gen = Generator(Noise_dim, img_channel, 8)
#     Initialize_Weight(gen)
#     # assert gen(InputGen_Noise).shape == (N, in_channel, 1, 1)
#     TestGen = gen(InputGen_Noise)
#     print(f"Generator test: {TestGen.shape}")

# test(8, 3, 64, 64, 100)

# sys.exit("Testing the Disc and Genrator networks")


transforms = transforms.Compose(
    [
    transforms.ToTensor(),
    transforms.Resize([Image_Width, Image_Height]),
    transforms.Normalize(
        [0.5 for _ in range(Num_ColorChannels)], [0.5 for _ in range(Num_ColorChannels)]
    )
    ]
)

if FFHQdataset == True:
    DataLoading = FFHQ_Dataset(TrainDataLoc, transform=transforms)
    IterationOfTheData = DataLoader(DataLoading, batch_size=batch_size, shuffle=True)

if MNISTdataset == True:
    DataLoading = datasets.MNIST(root=TrainDataLoc, transform=transforms, download=True)
    IterationOfTheData = DataLoader(DataLoading, batch_size=batch_size, shuffle=True)

if Celebdataset == True:
    TrainDataLocFinal = TrainDataLoc+"/Celeb_Dataset"
    DataLoading = datasets.ImageFolder(root=TrainDataLocFinal, transform=transforms)
    IterationOfTheData = DataLoader(DataLoading, batch_size=batch_size, shuffle=True)

fixed_noise_For_Tensorboard = torch.randn(Num_Imgs_On_Tensorboard, Noise_Dim, 1, 1).to(device)
writer_fake = SummaryWriter(f"{TrainDataLoc}/Tensorboard/runs/Fake")
writer_real = SummaryWriter(f"{TrainDataLoc}/Tensorboard/runs/Real")
step = 0

for epoch in range(num_epochs):
    if Model_Load is True:
        epoch = epoch_saved + epoch
        if batch_saved == len(IterationOfTheData):
            epoch = epoch + 1

    for Batch_Index, (Real_Image, _) in enumerate(IterationOfTheData):
        if Model_Load is True:
            Batch_Index = batch_saved + Batch_Index + 1

        Real_Image = Real_Image.to(device)
        Noise_input = torch.randn(batch_size, Noise_Dim, 1, 1).to(device)
        Gen_Noise = gen(Noise_input)

        # Disc Loss
        Disc_ReImg = disc(Real_Image).reshape(-1)
        Disc_Gen_Noise = disc(Gen_Noise.detach()).reshape(-1)  # kind of similar to squeeze

        loss_disc_real = criterion(Disc_ReImg, torch.ones_like(Disc_ReImg))
        loss_disc_fake = criterion(Disc_Gen_Noise, torch.zeros_like(Disc_Gen_Noise))
        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        disc.zero_grad()
        loss_disc.backward() #Retain graph could be set to True instead of detach a few lines above here, since we are disc(gen_Fake_Image) and performing another backprop below that involves disc(gen_Fake_Image)
        disc_Optim.step()

        # Gen Loss
        Disc_Gen_Noise = disc(Gen_Noise).reshape(-1)
        loss_gen = criterion(Disc_Gen_Noise, torch.ones_like(Disc_Gen_Noise))
        gen.zero_grad()
        loss_gen.backward()
        gen_Optim.step()

        if Batch_Index % 200 == 0 and Batch_Index != 0:
            if Model_Save == True:
                ModelSave_func(disc, disc_Optim, loss_disc, batch_num=Batch_Index, epoch_num=epoch, path=f'{TrainDataLoc}/Model_Save/Discriminator_Epoch{epoch}_BatchIdx{Batch_Index}.pth')
                ModelSave_func(gen, gen_Optim, loss_gen, batch_num=Batch_Index, epoch_num=epoch, path=f'{TrainDataLoc}/Model_Save/Generator_Epoch{epoch}_BatchIdx{Batch_Index}.pth')

        if Batch_Index % 10 == 0:
            print(f"Batch: [{Batch_Index}/{len(IterationOfTheData)}] \ "
                  f"eEpoch: [{epoch}/{num_epochs}] \ "
                  f"Loss Discriminator: {loss_disc: 0.4f} & Loss Generator: {loss_gen: 0.4f}"
            )

            with torch.no_grad():
                fake_Image = gen(fixed_noise_For_Tensorboard)

                img_grid_fake = torchvision.utils.make_grid(fake_Image, normalize=True)
                img_grid_real = torchvision.utils.make_grid(Real_Image[:Num_Imgs_On_Tensorboard], normalize=True)

                writer_fake.add_image(
                    "Fake Images", img_grid_fake, global_step=step
                )

                writer_real.add_image(
                    "Real Images", img_grid_real, global_step=step
                )

                step += 1
