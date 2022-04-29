import torch
import torch.nn as nn

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
