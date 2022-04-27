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

TrainDataLoc = '/Users/javidabderezaei/Downloads/TransferToServer/Explicit-GAN-Project/FFHQ/Images_Combined'

## Hyperparameters
# lr = 3e-4
# device = "cuda" if torch.cuda.is_available() else "cpu"
# Noise_Dim = 64
# Image_Width = 28
# Image_Height = 28
# Num_ColorChannels = 1
# Image_Dim_Total = Image_Width * Image_Height * Num_ColorChannels
# batch_size = 32
# num_epochs = 100

# class ConvBloc(nn.Module):
#     def __init__(self, in_channels):
#         super().__init__()
#
#         self.ConvBloc_Layer = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels, kernel_size=3),
#             nn.BatchNorm2d(in_channels),
#             nn.LeakyReLU(0.01),
#             nn.Conv2d(in_channels, in_channels, kernel_size=3),
#             nn.BatchNorm2d(in_channels),
#             nn.Conv2d(in_channels, in_channels, kernel_size=3),
#             nn.BatchNorm2d(in_channels),
#         )
#
#     def forward(self, x):
#         return self.ConvBloc_Layer(x)

## Testing the first block!
# ConvBlocBloc = ConvBloc(1)
# RandomData = torch.randn(28, 28).unsqueeze(0).unsqueeze(0)
# y = ConvBlocBloc(RandomData)
# print(y.shape)
# sys.exit("Error message")

# class Convblock_Downsample(nn.Module):
#     def __init__(self, in_channels):
#         super().__init__()
#         self.Convblock_Downsample_Layer = nn.Sequential(
#             nn.Conv2d(in_channels, 2*in_channels, kernel_size=4, stride=1, padding=1),
#             nn.BatchNorm2d(2*in_channels, 2*in_channels),
#             nn.LeakyReLU(0.01),
#             nn.Conv2d(2*in_channels, 4*in_channels, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(4*in_channels, 4*in_channels),
#         )
#     def forward(self, x):
#         return self.Convblock_Downsample_Layer(x)

## Testing the block!
# ConvblockDown = Convblock_Downsample(1)
# Testdata = torch.randn(1, 1, 28, 28)
# y = ConvblockDown(Testdata)
# print(y.shape)
# sys.exit('Just testing')


# class Convblock_Upsample(nn.Module):
#     def __init__(self, in_channels):
#         super().__init__()
#
#         self.Convblock_Upsample_Layers = nn.Sequential(
#             nn.Conv2d(in_channels, int(in_channels/2), kernel_size=3),
#             nn.BatchNorm2d(int(in_channels/2), int(in_channels/2)),
#             nn.LeakyReLU(0.01),
#             nn.Conv2d(int(in_channels/2), int(in_channels/4), kernel_size=3),
#             nn.BatchNorm2d(int(in_channels/4), int(in_channels/4)),
#         )
#
#     def forward(self, x):
#         return self.Convblock_Upsample_Layers(x)

## Testing the block!
# ConvBlocDown = Convblock_Upsample(1)
# RandomData = torch.randn(28, 28).unsqueeze(0).unsqueeze(0)
# y = ConvBlocDown(RandomData)
# print(y.shape)
# sys.exit("Error message")

class Discriminator(nn.Module):
    def __init__(self, in_channels, feature_d):
        super().__init__() #Differentce between super(Discriminator) and super() should be checked!

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
            nn.Conv2d(in_channels, out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channel, out_channel),
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
                nn.BatchNorm2d(out_channels, out_channels),
                nn.ReLU()
            )
    def forward(self, x):
        return self.gen(x)


def Initialize_Weight(Model):
    for m in Model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
    return Model

def test(N, in_channel, Height, Width, Noise_dim):
    InputDisc = torch.randn(N, in_channel, Height, Width)

    disc = Discriminator(in_channel, feature_d=8)
    Initialize_Weight(disc)
    # assert disc(Input).shape == (N, 1, 1, 1)
    TestDisc = disc(InputDisc)
    print(F"Discriminator test: {TestDisc.shape}")

    img_channel = in_channel
    InputGen_Noise = torch.randn(N, Noise_dim, 1, 1)
    gen = Generator(Noise_dim, img_channel, 8)
    Initialize_Weight(gen)
    # assert gen(InputGen_Noise).shape == (N, in_channel, 1, 1)
    TestGen = gen(InputGen_Noise)
    print(f"Generator test: {TestGen.shape}")

test(8, 3, 64, 64, 100)

sys.exit("Testing the Disc and Genrator networks")

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




# def calc_conv_dim():
#     in_sz = int(input('input size: '))
#     p = int(input('padding: '))#0
#     d = int(input('dialation: '))#1
#     k = int(input('kernel: '))#3
#     s = int(input('stride: '))#1
#     tr = input('transpose?(Y/N): ')
#     if tr in ['y','Y','Yes','yes','YES']:
#         op = int(input('output padding: '))
#         out_sz = int((in_sz-1)*s - (2*p) + (d*(k-1)) + op + 1)
#     elif tr in ['n','N','no','No','NO']:
#         out_sz = int(((in_sz + (2*p) - (d*(k-1))-1)/s) + 1)
#     else:
#         print('NO')
#         return 0
#     print('\n---OUTPUT---\noutput size =', out_sz)
#     print('Δ =', abs(in_sz-out_sz))
#     return 1
#
# if __name__ == "__main__":
#     flag = True
#     while(flag):
#         calc_conv_dim()
#         temp = input('\nExit? ')
#         if temp in ['y','Y','Yes','yes','YES']:
#             flag = False
#         elif temp not in ['n','N','no','No','NO']:
#             print('NO')
#             flag = False