import torch 
import torch.nn as nn 
from torch.nn import functional as F 


class Contrastiveloss(nn.Module):
    """
    Contrastive loss function 
    """

    def __init__(self, margin=2.0):
        super(Contrastiveloss, self).__init__()
        self.margin = margin 

    def forward(self, op1, op2, label):
        euclidean_distance = F.pairwise_distance(op1, op2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive


if __name__ == "__main__":
    Contrastiveloss()
