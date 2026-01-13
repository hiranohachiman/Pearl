import torch
import torch.nn as nn


class HadamardNet(nn.Module):
    def __init__(self):
        super(HadamardNet, self).__init__()

        self.network = nn.Sequential(
            nn.Conv2d(2, 128, kernel_size=1), 
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(p=0.0),
            nn.Conv2d(128, 256, kernel_size=1),  
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(p=0.0),
            nn.Conv2d(256, 512, kernel_size=1),  
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(p=0.0),
            nn.Conv2d(512, 1024, kernel_size=1), 
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.0),
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(p=0.0),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(p=0.0),
            nn.Conv2d(256, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(p=0.0),
            nn.Conv2d(128, 1, kernel_size=1),  
        )

    def forward(self, x1, x2):
        x1 = x1.unsqueeze(1).unsqueeze(-1)
        x2 = x2.unsqueeze(1).unsqueeze(-1)

        combined = torch.cat((x1, x2), dim=1)

        output = self.network(combined)

        return output.squeeze(1).squeeze(-1)

