import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
# from DataLoader import FFHQ_Dataset
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from Model import Critic, Generator, Initialize_Weight
import matplotlib.pyplot as plt
import sys

Model_Save = True
Model_Load = False

if Model_Load is True:
    ModelVersion_crit = "/Model_Save_WGAN/Critic_Epochxx_BatchIdxxx.pth"
    ModelVersion_gen = "/Model_Save_WGAN/Generator_Epochxx_BatchIdxxx.pth"

FFHQdataset = False
MNISTdataset = False
Celebdataset = True


if FFHQdataset == True:
    TrainDataLoc = '/Users/javidabderezaei/Downloads/TransferToServer/Explicit-GAN-Project/FFHQ/Images_Combined'
if MNISTdataset == True:
    TrainDataLoc = "/Users/javidabderezaei/Downloads/TransferToServer/Explicit-GAN-Project/MNIST_Data"
if Celebdataset == True:
    TrainDataLoc = "C:/TransferToServerJavid/Celeb"

## Hyperparameters
lr = 5e-5
device = "cuda" if torch.cuda.is_available() else "cpu"
Noise_Dim = 128
Image_Width = 64
Image_Height = 64
Num_ColorChannels = 3
batch_size = 64
num_epochs = 100
feature_d = 64
feature_g = 64
Num_Imgs_On_Tensorboard = 32
Critic_Iteration = 5
Weight_Clip = 0.01

def ModelSave_func(model, optimization, loss, batch_num, epoch_num, path):
    checkpoint = {'epoch': epoch_num, 'State_dict': model.state_dict(), 'Optimizer': optimization.state_dict(), 'loss': loss, 'batch_num': batch_num}
    torch.save(checkpoint, path)

crit = Critic(Num_ColorChannels, feature_d).to(device)
gen = Generator(Noise_Dim, Num_ColorChannels, feature_g).to(device)

Initialize_Weight(crit)
Initialize_Weight(gen)

crit_Optim = torch.optim.RMSprop(crit.parameters(), lr=lr)
gen_Optim = torch.optim.RMSprop(gen.parameters(), lr=lr)


if Model_Load is True:
    ModelLocation_crit = TrainDataLoc + ModelVersion_crit
    Loaded_Checkpoint_crit = torch.load(ModelLocation_crit)
    crit.load_state_dict(Loaded_Checkpoint_crit['State_dict'])
    crit_Optim.load_state_dict(Loaded_Checkpoint_crit['Optimizer'])
    epoch_saved = Loaded_Checkpoint_crit['epoch']
    batch_saved = Loaded_Checkpoint_crit['batch_num']
    loss_crit = Loaded_Checkpoint_crit['loss']

    ModelLocation_gen = TrainDataLoc + ModelVersion_gen
    Loaded_Checkpoint_gen = torch.load(ModelLocation_gen)
    gen.load_state_dict(Loaded_Checkpoint_gen['State_dict'])
    gen_Optim.load_state_dict(Loaded_Checkpoint_gen['Optimizer'])
    loss_gen = Loaded_Checkpoint_gen['loss']


transforms = transforms.Compose(
    [
    transforms.ToTensor(),
    transforms.Resize([Image_Width, Image_Height]),
    transforms.Normalize(
        [0.5 for ii in range(Num_ColorChannels)], [0.5 for ii in range(Num_ColorChannels)]
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
    TrainDataLocFinal = TrainDataLoc+"\Celeb_Dataset"
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
        for i in range(Critic_Iteration):
            Noise_input = torch.randn(batch_size, Noise_Dim, 1, 1).to(device)

            ## Discriminator Loss
            Gen_Img = gen(Noise_input)
            crit_real = crit(Real_Image).reshape(-1)
            crit_fake = crit(Gen_Img).reshape(-1)

            loss_crit = -(torch.mean(crit_real) - torch.mean(crit_fake)) #Minimizing the negative format of this equation do that technically we arw maximizing it according to the paper!
            crit.zero_grad()
            loss_crit.backward(retain_graph=True)
            crit_Optim.step()

            for p in crit.parameters():
                p.data.clamp_(-Weight_Clip, +Weight_Clip)

            ## Generator Loss: max E[Critic(gen_fake)] <-> min -E[Critic(gen_fake)]
        loss_gen = -torch.mean(crit(Gen_Img).reshape(-1))
        gen.zero_grad()
        loss_gen.backward()
        gen_Optim.step()
        
        
   


        if Batch_Index % 600 == 0 and Batch_Index != 0:
            if Model_Save == True:
                ModelSave_func(crit, crit_Optim, loss_crit, batch_num=Batch_Index, epoch_num=epoch, path=f'{TrainDataLoc}\Model_Save_WGAN\critic_Epoch{epoch}_BatchIdx{Batch_Index}.pth')
                ModelSave_func(gen, gen_Optim, loss_gen, batch_num=Batch_Index, epoch_num=epoch, path=f'{TrainDataLoc}\Model_Save_WGAN\Generator_Epoch{epoch}_BatchIdx{Batch_Index}.pth')

        if Batch_Index % 10 == 0:
            print(f"Batch: [{Batch_Index}/{len(IterationOfTheData)}] \ "
                  f"Epoch: [{epoch}/{num_epochs}] \ "
                  f"Loss Critic: {loss_crit: 0.4f} & Loss Generator: {loss_gen: 0.4f}"
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
