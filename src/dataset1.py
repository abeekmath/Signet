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

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img0_path = os.path.join(self.data_root, 
                                 self.dataframe.iloc[idx, 0])
        img1_path = os.path.join(self.data_root, 
                                 self.dataframe.iloc[idx, 1])
        label = self.dataframe.iloc[idx, 2]

        img0 = Image.open(img0_path).convert("L")
        img1 = Image.open(img1_path).convert("L")

        if self.resize is not None: 
            image0 = img0.resize(
                (self.resize[1], self.resize[0]), resample=Image.BILINEAR
            )
            image1 = img1.resize(
                (self.resize[1], self.resize[0]), resample=Image.BILINEAR
            )

            if transform is not None:
                image0 = self.transform(image0)
                image1 = self.transform(image1)

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

        print(i, sample["images0"].shape, sample["images1"].shape, sample["label"].item())

        if i == 1:
            break





