import torch
import torch.nn as nn
# from torch import sigmoid

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        x = self.conv(x)
        return x

class Unet(nn.Module):
    def __init__(self, n_channels, n_filters, n_classes):
        super().__init__()
        
        self.conv1 = ConvBlock(n_channels, n_filters)
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2))
        
        self.conv2 = ConvBlock(n_filters, n_filters*2)
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2))
        
        self.conv3 = ConvBlock(n_filters*2, n_filters*4)
        self.pool3 = nn.MaxPool2d(kernel_size=(2,2))
        
        self.conv4 = ConvBlock(n_filters*4, n_filters*8)
        self.pool4 = nn.MaxPool2d(kernel_size=(2,2))
        
        self.bridge5 = ConvBlock(n_filters*8, n_filters*16)
        
        self.up6 = nn.ConvTranspose2d(n_filters*16, n_filters*8, kernel_size=(2, 2), stride=(2, 2))
        self.conv6 = ConvBlock(n_filters*16, n_filters*8)
        
        self.up7 = nn.ConvTranspose2d(n_filters*8, n_filters*4, kernel_size=(2, 2), stride=(2, 2))
        self.conv7 = ConvBlock(n_filters*8, n_filters*4)
        
        self.up8 = nn.ConvTranspose2d(n_filters*4, n_filters*2, kernel_size=(2, 2), stride=(2, 2))
        self.conv8 = ConvBlock(n_filters*4, n_filters*2)
        
        self.up9 = nn.ConvTranspose2d(n_filters*2, n_filters, kernel_size=(2, 2), stride=(2, 2))
        self.conv9 = ConvBlock(n_filters*2, n_filters)
        
        self.outputs = nn.Conv2d(n_filters, n_classes, kernel_size=(1,1))
    
    def forward(self, x):        
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)
        
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)
        
        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)
        
        bridge5 = self.bridge5(pool4)
        
        up6 = self.up6(bridge5)
        cat6 = torch.cat([up6, conv4], dim=1)
        conv6 = self.conv6(cat6)
        
        up7 = self.up7(conv6)
        cat7 = torch.cat([up7, conv3], dim=1)
        conv7 = self.conv7(cat7)
        
        up8 = self.up8(conv7)
        cat8 = torch.cat([up8, conv2], dim=1)
        conv8 = self.conv8(cat8)
        
        up9 = self.up9(conv8)
        cat9 = torch.cat([up9, conv1], dim=1)
        conv9 = self.conv9(cat9)
        
        x = self.outputs(conv9)

        sigmoid = nn.Sigmoid()
        
        return sigmoid(x)
