import torch 
import torch.nn as nn 
from torch.nn import functional as F 


class SiameseNet(nn.Module):
    def __init__(self):
        super(SiameseNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 96, 11, stride=1), 
            nn.ReLU(inplace=True), 
            nn.LocalResponseNorm(5, k=2),
            nn.MaxPool2d(3, 2),

            nn.Conv2d(96, 256, 5, stride=1, padding=2),
            nn.ReLU(inplace = True),
            nn.LocalResponseNorm(5, k=2),
            nn.MaxPool2d(3, 2),
            nn.Dropout(p=0.3),

            nn.Conv2d(256, 384, 3, stride=1, padding=1), 
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            nn.Dropout(p=0.3),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(30976, 1024),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 128)
        )
    
    def forward_once(self, x):
        output = self.conv_layers(x)
        output = output.view(output.size()[0], -1)
        output = self.fc_layers(output)
        return output

    def forward(self, img1, img2):
        output1 = self.forward_once(img1)
        output2 = self.forward_once(img2)
        return output1, output2
    
if __name__ == "__main__":
    x = torch.randn([4, 1, 105, 105])
    net = SiameseNet()
    out = net(x)

