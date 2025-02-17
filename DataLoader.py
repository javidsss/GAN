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
import torchvision.transforms as transforms
# import json

# Metadata_Loc = '/Users/javidabderezaei/Downloads/TransferToServer/Explicit-GAN-Project/FFHQ-Images/ffhq-dataset-v2.json'
# f = open(Metadata_Loc, "r")
# data = json.loads(f.read())


class FFHQ_Dataset(Dataset):
    def __init__(self, train_loc, transform=None):
        super(FFHQ_Dataset, self).__init__()

        self.transform = transform
        self.train_loc = train_loc
        self.ImageNames = os.listdir(self.train_loc)

    def ImagePreprocessing(self, train_loc_final):
        # train_data_numpy = np.array(nib.load(train_loc_final).get_fdata())
        train_data = np.array(Image.open(train_loc_final))
        [h, w, z] = train_data.shape

        if self.transform is not None:
            image = self.transform(train_data)  # image and mask are dict names, I can use whatever name I want. Then I have to call them as I named them!
        else:
            image = train_data

        return image.unsqueeze(0)

    def __getitem__(self, index):
        if self.ImageNames[index].endswith('.png'):
            Image_Loc = os.path.join(self.train_loc, self.ImageNames[index])
            Image_Final = self.ImagePreprocessing(Image_Loc)
        else:
            Image_Final = []

        return Image_Final

    def __len__(self):
        return len(self.ImageNames)

if __name__ == "__main__":
    TrainDataLoc = '/Users/javidabderezaei/Downloads/TransferToServer/Explicit-GAN-Project/FFHQ/Images_Combined'
    batch_size = 1

    IMAGE_HEIGHT = 128
    IMAGE_WIDTH = 128
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
    transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize([IMAGE_HEIGHT, IMAGE_WIDTH]), transforms.Normalize((0.1307), (0.3081))]
    )
    DataLoadPractice = FFHQ_Dataset(TrainDataLoc, transform=transforms)
    IterationOfTheData = DataLoader(DataLoadPractice, batch_size=batch_size, shuffle=False)

    for i, image in enumerate(IterationOfTheData):
        for i in range(image.shape[0]):
            x=2
