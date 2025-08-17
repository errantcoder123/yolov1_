import torch
import torch.nn as nn

"""
YOLOv1 model architecture
Kernel size, number of filters, stride, padding
"""
architecture_config = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]

class CNNblock(nn.Module):
    def __init__(self, inc, outc, **kwargs):
        super(CNNblock, self).__init__()
        self.conv = nn.Conv2d(inc, outc, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(outc)
        self.leakyrelu = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))
    
class Yolov1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(Yolov1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self.create_conv_layers(self.architecture)
        self.fully_connected = self.create_fully_connected(**kwargs)
    
    def forward(self, x):
        x = self.darknet(x)
        return self.fully_connected(torch.flatten(x, start_dim=1))
    
    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels
        
        for x in architecture:
            if type(x) == tuple:
                layers += [
                    CNNblock(
                        in_channels, outc=x[1], kernel_size=x[0], stride=x[2], padding=x[3],
                    )
                ]
                in_channels = x[1] 
                
            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]
                
            elif type(x) == list:
                conv1 = x[0]
                conv2 = x[1]
                num_repeats = x[2]
                
                for _ in range(num_repeats):
                    layers += [
                        CNNblock(
                            in_channels, conv1[1], kernel_size=conv1[0], stride=conv1[2], padding=conv1[3],
                        )
                    ]
                    layers += [
                        CNNblock(
                            conv1[1], conv2[1], kernel_size=conv2[0], stride=conv2[2], padding=conv2[3],
                        )
                    ]
                    in_channels = conv2[1] 
                
        return nn.Sequential(*layers)
    
    def create_fully_connected(self, S, B, C):
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * 7 * 7, 496), # it was 4096 in original paper
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(496, S * S * (C + B * 5)), # it was 4096 in original paper
        ) # to be reshaped as (S, S, C + B * 5 = 30)
    
