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

# class CerebellumData(Dataset):
#     def __init__(self, train_loc, scale_factor=None, transform=None):
#         super(CerebellumData, self).__init__()
#
#         self.train_loc = train_loc
#         self.train_foldernames = os.listdir(self.train_loc)
#         self.train_foldernames_NoExtra = []
#         for FolderNames in self.train_foldernames:
#             if FolderNames.startswith('Subj'):
#                 self.train_foldernames_NoExtra.append(FolderNames)
#
#         self.train_mask_foldernames = os.listdir(self.train_loc)
#         self.train_mask_foldernames_NoExtra = []
#         for FolderNames in self.train_mask_foldernames:
#             if FolderNames.startswith('Subj'):
#                 self.train_mask_foldernames_NoExtra.append(FolderNames)
#
#         self.scale_factor = scale_factor
#         self.transform = transform
#
#     def __getitem__(self, index):
#         train_loc_final = os.path.join(self.train_loc, self.train_foldernames_NoExtra[index], 'T1_LPI.nii')
#         train_data_numpy3D = np.array(nib.load(train_loc_final).get_fdata())
#         SliceToSave = math.ceil(train_data_numpy3D.shape[0] / 2)
#         train_data_numpy = np.squeeze(train_data_numpy3D[SliceToSave, :, : ])
#         [h, w] = train_data_numpy.shape
#         if self.scale_factor is not None:
#             train_data_final = np.zeros([int(math.ceil(h / self.scale_factor)), int(math.ceil(w / self.scale_factor))], dtype='float32')
#             train_data_final[:, :] = cv2.resize(train_data_numpy[:, :], dsize=(int(math.ceil(w / self.scale_factor)), int(h / self.scale_factor)))  # open cvs size input automatically transposes the sizess!!!
#         else:
#             train_data_final = train_data_numpy
#
#
#         train_mask_loc_final = os.path.join(self.train_loc, self.train_mask_foldernames_NoExtra[index], 'aseg_LPI.nii')
#         train_mask_numpy3D = np.array(nib.load(train_mask_loc_final).get_fdata())
#         train_mask_numpy = np.squeeze(train_mask_numpy3D[SliceToSave, :, :])
#         if self.scale_factor is not None:
#             train_mask_final = np.zeros([int(math.ceil(h / self.scale_factor)), int(math.ceil(w / self.scale_factor))], dtype='float32')
#             train_mask_final[:, :] = cv2.resize(train_mask_numpy[:, :], dsize=(int(math.ceil(w / self.scale_factor)), int(h / self.scale_factor))) #open cvs size input automatically transposes the sizess!!!
#         else:
#             train_mask_final = train_mask_numpy
#
#
#         if self.transform is not None:
#             train_mask_finalIndex = np.isin(train_mask_final, [6, 7, 8, 45, 46, 47])
#             train_mask_final_Ones = train_mask_finalIndex.astype(int)
#             MaskedImage = train_mask_final_Ones*train_data_final
#             MaskedImage = self.transform(MaskedImage)
#             # nib.save(maskedimage.to(torch.float32), os.path.join(train_loc_final, '2DTonsil.nii'))
#
#         else:
#
#             train_mask_finalIndex = np.isin(train_mask_final, [6, 7, 8, 45, 46, 47])
#             train_mask_final_Ones = train_mask_finalIndex.astype(int)
#             MaskedImage = train_mask_final_Ones*train_data_final
#             Rotated = np.rot90(train_data_final, 3, [0, 1])
#             plt.imshow(train_data_final)  # T1-LPI
#             plt.figure()
#             plt.imshow(Rotated)  # Rotated T1-LPI
#             plt.figure()
#             plt.imshow(train_mask_final)  # T1
#             plt.figure()
#             plt.imshow(Rotated-train_mask_final)
#
#             SavePath = os.path.join(self.train_loc, self.train_foldernames_NoExtra[index], 'MidSlice_Tonsil')
#             print(SavePath)
#             # nib.save(nib.Nifti1Image(maskedimage, affine=np.eye(4)), SavePath)
#
#         return 0, 0
#
#     def __len__(self):
#         return len(self.train_mask_foldernames_NoExtra)


class CerebellumData_LPI_To_Normal(Dataset):
    def __init__(self, train_loc, scale_factor=None, transform=None):
        super(CerebellumData_LPI_To_Normal, self).__init__()

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
        train_loc_final = os.path.join(self.train_loc, self.train_foldernames_NoExtra[index], 'T1_LPI_Test.nii')
        train_data_numpy3D = np.array(nib.load(train_loc_final).get_fdata())
        SliceToSave = math.ceil(train_data_numpy3D.shape[0] / 2)
        # train_data_numpy3D_Rotated = np.rot90(train_data_numpy3D, 1, [0, 1])
        # train_data_numpy3D_Rotated_Permuted = np.transpose(train_data_numpy3D_Rotated, [0, 2, 1])
        # train_data_numpy3D_Rotated_Permuted_Fliped = np.flipud(train_data_numpy3D_Rotated_Permuted)

        # train_data_numpy_AllSlices = np.zeros(np.shape(train_data_numpy3D))

        train_data_numpy3D_Permuted = np.transpose(train_data_numpy3D, [0, 2, 1])

        # for Slicess in range(train_data_numpy_AllSlices.shape[1]):
        #     EachSlice = np.squeeze(train_data_numpy3D_Permuted[:, Slicess, :])
        #     train_data_numpy_AllSlices[:, Slicess, :] = np.flipud(np.rot90(EachSlice, 1, [0, 1]))


        train_data_numpy = np.squeeze(train_data_numpy3D_Permuted[:, :, SliceToSave+20])


        [h, w] = train_data_numpy.shape
        if self.scale_factor is not None:
            train_data_final = np.zeros([int(math.ceil(h / self.scale_factor)), int(math.ceil(w / self.scale_factor))],
                                        dtype='float32')
            train_data_final[:, :] = cv2.resize(train_data_numpy[:, :], dsize=(int(math.ceil(w / self.scale_factor)),
                                                                               int(h / self.scale_factor)))  # open cvs size input automatically transposes the sizess!!!
        else:
            train_data_final = train_data_numpy

        train_mask_loc_final = os.path.join(self.train_loc, self.train_mask_foldernames_NoExtra[index], 'T1.nii')
        train_mask_numpy3D = np.array(nib.load(train_mask_loc_final).get_fdata())


        train_mask_numpy = np.fliplr(np.squeeze(train_mask_numpy3D[:, :, SliceToSave+20]))


        if self.scale_factor is not None:
            train_mask_final = np.zeros([int(math.ceil(h / self.scale_factor)), int(math.ceil(w / self.scale_factor))],
                                        dtype='float32')
            train_mask_final[:, :] = cv2.resize(train_mask_numpy[:, :], dsize=(int(math.ceil(w / self.scale_factor)),
                                                                               int(h / self.scale_factor)))  # open cvs size input automatically transposes the sizess!!!
        else:
            train_mask_final = train_mask_numpy

        if self.transform is not None:
            train_mask_finalIndex = np.isin(train_mask_final, [6, 7, 8, 45, 46, 47])
            train_mask_final_Ones = train_mask_finalIndex.astype(int)
            MaskedImage = train_mask_final_Ones * train_data_final
            MaskedImage = self.transform(MaskedImage)
            # nib.save(maskedimage.to(torch.float32), os.path.join(train_loc_final, '2DTonsil.nii'))

        else:

            train_mask_finalIndex = np.isin(train_mask_final, [6, 7, 8, 45, 46, 47])
            train_mask_final_Ones = train_mask_finalIndex.astype(int)
            MaskedImage = train_mask_final_Ones * train_data_final
            # Rotated = np.rot90(train_data_final, 3, [0, 1])

            plt.imshow(train_data_final)  # T1-LPI
            plt.figure()
            plt.imshow(train_mask_final)  # T1
            plt.figure()
            plt.imshow(train_data_final - train_mask_final)

            SavePath = os.path.join(self.train_loc, self.train_foldernames_NoExtra[index], 'T1_Converted_From_LPI')
            print(SavePath)
            # nib.save(nib.Nifti1Image(maskedimage, affine=np.eye(4)), SavePath)

        return 0, 0

    def __len__(self):
        return len(self.train_mask_foldernames_NoExtra)



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
batch_size = 8
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


# DataLoadPractice = CerebellumData(TrainDataLoc, transform=None)
DataLoadPractice = CerebellumData_LPI_To_Normal(TrainDataLoc, transform=None)
IterationOfTheData = DataLoader(DataLoadPractice, batch_size=batch_size, shuffle=False)

for i, image in enumerate(IterationOfTheData):
    plt.imshow(image[0,0,:,:,])
