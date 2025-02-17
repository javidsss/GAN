import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision
# import torchvision.transforms
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import math
# from PIL import Image
import torchvision.transforms as transforms
# import cv2
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
import nibabel as nib
# import nibabel.freesurfer.mghformat as mgh

class CerebellumData(Dataset):
    def __init__(self, train_loc, scale_factor=None, transform=None):
        super(CerebellumData, self).__init__()

        self.train_loc = train_loc
        self.train_foldernames = os.listdir(self.train_loc)
        self.train_foldernames_NoExtra = []
        for FolderNames in self.train_foldernames:
            if FolderNames.startswith('Subj'):
                self.train_foldernames_NoExtra.append(FolderNames)

        self.train_mask_foldernames = os.listdir(self.train_loc)
        self.train_mask_foldernames_NoExtra = []
        for FolderNames in self.train_mask_foldernames:
            if FolderNames.startswith('Subj'):
                self.train_mask_foldernames_NoExtra.append(FolderNames)

        self.scale_factor = scale_factor
        self.transform = transform

    def __getitem__(self, index):


        FileNamesInFolder = os.listdir(os.path.join(self.train_loc, self.train_foldernames_NoExtra[index]))

        for FileNames in FileNamesInFolder:
            if FileNames.startswith('T1_LPI') and not FileNames.endswith('seg1.nii') and not FileNames.endswith('seg2.nii') and FileNames.endswith('.nii'):
                LastName_T1 = FileNames
            if FileNames.startswith('aseg_LPI'):
                LastName_aseg = FileNames

        train_loc_final = os.path.join(self.train_loc, self.train_foldernames_NoExtra[index], LastName_T1)
        train_data_numpy3D = np.array(nib.load(train_loc_final).get_fdata())
        SliceToSave = math.ceil(train_data_numpy3D.shape[0] / 2)
        train_data_numpy = np.squeeze(train_data_numpy3D[SliceToSave, :, : ])
        [h, w] = train_data_numpy.shape
        if self.scale_factor is not None:
            train_data_final = np.zeros([int(math.ceil(h / self.scale_factor)), int(math.ceil(w / self.scale_factor))], dtype='float32')
            train_data_final[:, :] = cv2.resize(train_data_numpy[:, :], dsize=(int(math.ceil(w / self.scale_factor)), int(h / self.scale_factor)))  # open cvs size input automatically transposes the sizess!!!
        else:
            train_data_final = train_data_numpy

        train_mask_loc_final = os.path.join(self.train_loc, self.train_mask_foldernames_NoExtra[index], LastName_aseg)
        train_mask_numpy3D = np.array(nib.load(train_mask_loc_final).get_fdata())
        train_mask_numpy = np.squeeze(train_mask_numpy3D[SliceToSave, :, :])
        if self.scale_factor is not None:
            train_mask_final = np.zeros([int(math.ceil(h / self.scale_factor)), int(math.ceil(w / self.scale_factor))], dtype='float32')
            train_mask_final[:, :] = cv2.resize(train_mask_numpy[:, :], dsize=(int(math.ceil(w / self.scale_factor)), int(h / self.scale_factor))) #open cvs size input automatically transposes the sizess!!!
        else:
            train_mask_final = train_mask_numpy


        if self.transform is not None:
            train_mask_finalIndex = np.isin(train_mask_final, [6, 7, 8, 45, 46, 47])
            train_mask_final_Ones = train_mask_finalIndex.astype(int)
            MaskedImage = train_mask_final_Ones * train_data_final
            MaskedImage = self.transform(MaskedImage)
            nib.save(MaskedImage.to(torch.float32), os.path.join(train_loc_final, 'MidSlice_Cerebellum_LPI.nii'))

        else:

            train_mask_finalIndex = np.isin(train_mask_final, [6, 7, 8, 45, 46, 47])
            train_mask_final_Ones = train_mask_finalIndex.astype(int)
            MaskedImage = train_mask_final_Ones * train_data_final
            SavePath = os.path.join(self.train_loc, self.train_foldernames_NoExtra[index], 'MidSlice_Cerebellum_LPI')
            print(SavePath)
            nib.save(nib.Nifti1Image(MaskedImage, affine=np.eye(4)), SavePath)

        print(train_loc_final)
        # return train_data_final, train_mask_final_Ones, MaskedImage
        return 0, 0, MaskedImage

    def __len__(self):
        return len(self.train_mask_foldernames_NoExtra)


if __name__== "__Main__":
    # WorkstationAddress = "Z:\Chiari_Morphology\AutomaticSegmentationData\Combined\Chiari\Data"
    MacAddress = "/Volumes/Kurtlab/Chiari_Morphology/AutomaticSegmentationData/Combined/Chiari/Data"


    if 'MacAddress' in locals():
        TrainDataLoc = MacAddress
    if "WorkstationAddress" in locals():
        TrainDataLoc = WorkstationAddress

    Noise_Dim = 128
    Image_Width = 64
    Image_Height = 64
    Num_ColorChannels = 1
    batch_size = 1
    num_epochs = 100
    feature_d = 64
    feature_g = 64
    Num_Imgs_On_Tensorboard = 32
    Critic_Iteration = 5
    Lambda_GradientPenalty = 10

    transforms = transforms.Compose(
        [
        transforms.ToTensor(),
        transforms.Resize([Image_Width, Image_Height]),
        transforms.Normalize(
            [0.5 for ii in range(Num_ColorChannels)], [0.5 for ii in range(Num_ColorChannels)]
        )
        ]
    )


    DataLoadPractice = CerebellumData(TrainDataLoc, transform=None)
    IterationOfTheData = DataLoader(DataLoadPractice, batch_size=batch_size, shuffle=False)

    for i, (train_data_final, train_mask_final_Ones, MaskedImage) in enumerate(IterationOfTheData):

        plt.imshow(MaskedImage[0,:,:])
        plt.close()
        print(f'Save_Index_{i}')
