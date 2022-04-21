import torch
import numpy as np
import os
import matplotlib.pyplot as plt
# plt.switch_backend('TKAgg')
from torch.utils.data import Dataset, DataLoader
import math
# import cv2
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
from PIL import Image
import csv
# import json

# Metadata_Loc = '/Users/javidabderezaei/Downloads/TransferToServer/Explicit-GAN-Project/FFHQ-Images/ffhq-dataset-v2.json'
# f = open(Metadata_Loc, "r")
# data = json.loads(f.read())


class FFHQ_Dataset(Dataset):
    def __init__(self, train_loc, transform=None):
        super(FFHQ_Dataset, self).__init__()

        self.transform = transform

        self.train_loc = train_loc
        self.train_foldernames = os.listdir(self.train_loc)
        self.train_foldernames_NoExtra = []
        for FolderNames in self.train_foldernames:
            if FolderNames.endswith('000'):
                self.train_foldernames_NoExtra.append(FolderNames)

    def ImagePreprocessing(self, train_loc_final):
        # train_data_numpy = np.array(nib.load(train_loc_final).get_fdata())
        train_data = np.array(Image.open(train_loc_final))
        [h, w, z] = train_data.shape

        if self.transform is not None:
            augmentations = self.transform(image=train_data)  # image and mask are dict names, I can use whatever name I want. Then I have to call them as I named them!
            image = np.squeeze(augmentations["image"])  # For some reason, image is permuted extra!!
        else:
            image = train_data
        image = torch.from_numpy(image).unsqueeze(0)
        return image

    def __getitem__(self, index):

        for ImageName in os.listdir(os.path.join(self.train_loc, self.train_foldernames_NoExtra[index])):
            if ImageName.endswith('.png'):
                    Image_Loc = os.path.join(self.train_loc, self.train_foldernames_NoExtra[index], ImageName)
                    Image_Final = self.ImagePreprocessing(Image_Loc)

        return Image_Final

    def __len__(self):
        return len(self.train_foldernames_NoExtra)

TrainDataLoc = '/Users/javidabderezaei/Downloads/TransferToServer/Explicit-GAN-Project/FFHQ-Images'
batch_size = 1

# IMAGE_HEIGHT = 512
# IMAGE_WIDTH = 512
# train_transform = A.Compose(
#     [
#         A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
#         A.Rotate(limit=35, p=1.0),
#         A.HorizontalFlip(p=0.5),
#         A.VerticalFlip(p=0.1),
#         A.Normalize(
#             mean=[0.0],
#             std=[1.0],
#             max_pixel_value=255.0,
#         ),
#         # ToTensorV2(),
#     ],
# )


DataLoadPractice = FFHQ_Dataset(TrainDataLoc, transform=None)
IterationOfTheData = DataLoader(DataLoadPractice, batch_size=batch_size, shuffle=False)
#
for i, image in enumerate(IterationOfTheData):
    for i in range(image.shape[0]):
        x=2
