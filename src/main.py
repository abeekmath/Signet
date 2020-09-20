import os 
import torch 
import numpy as np 

import config 
from model import SiameseNet 
from engine import train_fn
from dataset import IcdarDataset 
from loss import Contrastiveloss
from tqdm import tqdm 

import torch.nn as nn 
from torch.utils.data import DataLoader 

def run_training():
    train_dataset = IcdarDataset(
        data_root=config.DATA_ROOT,
        csv_file=config.TRAIN_FILE,
        resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size = config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS, 
        shuffle=True, 
        pin_memory=True
    )
    test_dataset = IcdarDataset(
        data_root=config.DATA_ROOT, 
        csv_file = config.TEST_FILE, 
        resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        num_workers=config.NUM_WORKERS,
        shuffle=False,
        pin_memory=True,
    )

    model = SiameseNet()
    model.to(config.DEVICE)

    loss_fn =  Contrastiveloss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.8, patience=5, verbose=True
    )
    for epoch in range(config.EPOCHS):
        train_loss = train_fn(model, train_loader, optimizer, loss_fn)
        print(
            "Epoch={}, Train Loss={}".format(epoch, train_loss)
        )


if __name__ == "__main__":
    run_training()