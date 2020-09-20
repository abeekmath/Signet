import torch
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import DataLoader
from model import SiameseNet
from dataset import IcdarDataset
from loss import Contrastiveloss
import copy

import numpy as np
import config

def train(dataloader, model, loss_fn, optimizer, epochs, batch_size):
    model.train() 
    min_loss = 1000
    loss_epoch_arr = []

    n_iters  = np.ceil(len(dataloader) / batch_size)

    for epoch in range(epochs):
        for batch_id, data in enumerate(dataloader, 0):

            inp1, inp2, label = data["images0"], data["images1"], data["label"]
            inp1, inp2, label = inp1.to(device), inp2.to(device), label.to(device)

            optimizer.zero_grad()
            op1, op2 = model(inp1, inp2)
            loss = loss_fn(op1, op2, label)
            loss.backward()
            optimizer.step()

            if min_loss > loss.item():
                min_loss = loss.item()
                best_model = copy.deepcopy(model.state_dict())
                print('Min loss %0.2f' % min_loss)

            if batch_id % 100 == 0:
                print('Iteration: %d/%d, Loss: %0.2f' % (batch_id, n_iters, loss.item()))

        
            del inp1, inp2, label, op1, op2
            torch.cuda.empty_cache()

        loss_epoch_arr.append(loss.item())

    plt.plot(loss_epoch_arr)
    plt.show()

    return best_model


def compute_accuracy_roc(predictions, labels):
    dmax = np.max(predictions)
    dmin = np.min(predictions)
    nsame = np.sum(labels == 1)
    ndiff = np.sum(labels == 0)
    step = 0.001
    max_acc = 0

    d_optimal = 0
    for d in np.arange(dmin, dmax + step, step):
        idx1 = predictions.ravel() <= d
        idx2 = predictions.ravel() > d

        tpr = float(np.sum(labels[idx1] == 1)) / nsame
        tnr = float(np.sum(labels[idx2] == 0)) / ndiff

        acc = 0.5 * (tpr + tnr)

        if acc > max_acc:
            max_acc = acc
            d_optimal = d

    return max_acc, d_optimal



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    sign_dataset =  IcdarDataset(data_root=config.DATA_ROOT, 
                                csv_file=config.TRAIN_FILE, 
                                resize=(config.IMAGE_WIDTH, config.IMAGE_HEIGHT))
    train_dataloader = DataLoader(sign_dataset, 8, True)

    net = SiameseNet().to(device)
    optimizer = optim.RMSprop(net.parameters(), lr=1e-4, weight_decay=5e-4, 
                              momentum=0.9, eps=1e-8)
    loss_fn = Contrastiveloss()
    batch_size = 8
    train(train_dataloader, net, loss_fn, optimizer, 1, batch_size)
        


