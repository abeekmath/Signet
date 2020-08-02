import os 
import numpy as np
import pandas as pd 
from PIL import Image, ImageFile

import torch
from torch.utils.data import Dataset 
import albumentations

import config

ImageFile.LOAD_TRUNCATED_IMAGES = True

class IcdarDataset(Dataset):
    def __init__(self, data_root, csv_file, resize=None):
        self.data_root = data_root
        self.dataframe = pd.read_csv(csv_file)
        self.resize = resize 

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        self.aug = albumentations.Compose([
            albumentations.InvertImg(always_apply=True),
            albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)
        ])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img0_path = os.path.join(self.data_root, 
                                 self.dataframe.iloc[idx, 0])
        img1_path = os.path.join(self.data_root, 
                                 self.dataframe.iloc[idx, 1])
        label = self.dataframe.iloc[idx, 2]

        img0 = Image.open(img0_path)
        img1 = Image.open(img1_path)
        
        if self.resize is not None: 
            image0 = img0.resize(
                (self.resize[1], self.resize[0]), resample=Image.BILINEAR
            )
            image1 = img1.resize(
                (self.resize[1], self.resize[0]), resample=Image.BILINEAR
            )


            image0, image1 = np.array(image0), np.array(image1)
            augmented_0, augmented_1 = self.aug(image=image0), self.aug(image=image1)
            image0, image1 = augmented_0["image"], augmented_1["image"]

            image0 = np.transpose(image0, (2, 0, 1)).astype(np.float32)
            image0 = np.transpose(image1, (2, 0, 1)).astype(np.float32)

            return {
                "images0": torch.tensor(image0, dtype=torch.float),
                "images1": torch.tensor(image1, dtype=torch.float),
                "label": torch.tensor(label, dtype=torch.long),
            }
        
if __name__ == "__main__":
    sign_dataset =  IcdarDataset(data_root=config.DATA_ROOT, 
                                csv_file=config.TRAIN_FILE, 
                                resize=(config.IMAGE_WIDTH, config.IMAGE_HEIGHT))

    for i in range(len(sign_dataset)):
        sample = sign_dataset[i]

        print(i, sample["images0"].shape, sample["images0"].shape, sample["label"].item())

        if i == 20:
            break





