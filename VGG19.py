import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
# the VGG19 architecture

class VGG19(nn.Module):
    def __init__(self,num_classes,in_channel):
        super(VGG19,self).__init__()
        self.CNN_Layers = nn.Sequential(
            nn.Conv2d(in_channel,64,kernel_size=3,padding=1),
            nn.Relu(inplace=False),
            nn.MaxPool1d(kernel_size=2,stride=2),

            nn.Conv2d(64,128,kernel_size=3,padding=1),
            nn.Relu(inplace=False),
            nn.MaxPool1d(kernel_size=2,stride=2),

            nn.Conv2d(128,256,kernel_size=3,padding=1),
            nn.Relu(inplace=False),
            nn.MaxPool1d(kernel_size=2,stride=2),

            nn.Conv2d(256,512,kernel_size=3,padding=1),
            nn.Relu(inplace=False),
            nn.MaxPool1d(kernel_size=2,stride=2),

            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.Relu(inplace=False),
            nn.MaxPool1d(kernel_size=2,stride=2),
        )
        self.FC_Layers = nn.Sequential(

        )
    
    def forward(self,x):
        
        return x
    
'''
class VGG19(nn.Module):
    def __init__(self, s_classes):
        super(VGG19, self).__init__()
        # conv layers: (in_channel size, out_channels size, kernel_size, stride, padding)
        self.conv1_1 = nn.Conv2d(3,64,kernel_size=3,padding=1)
        self.conv1_2 = nn.Conv2d(64,64,kernel_size=3,padding=1)

        self.conv2_1 = nn.Conv2d(64,128,kernel_size=3,padding=1)
        self.conv2_2 = nn.Conv2d(128,128,kernel_size=3,padding=1)

        self.conv3_1 = nn.Conv2d(128,256,kernel_size=3,padding=1)
        self.conv3_2 = nn.Conv2d(256,256,kernel_size=3,padding=1)
        self.conv3_3 = nn.Conv2d(256,256,kernel_size=3,padding=1)

        self.conv4_1 = nn.Conv2d(256,512,kernel_size=3,padding=1)
        self.conv4_2 = nn.Conv2d(512,512,kernel_size=3,padding=1)
        self.conv4_3 = nn.Conv2d(512,512,kernel_size=3,padding=1)
        self.conv4_4 = nn.Conv2d(512,512,kernel_size=3,padding=1)
        
        # max pooling (kernel_size, stride)
        self.pool = nn.MaxPool1d(2,2)
         # fully conected layers:
    def forward(self, x, training=True):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool(x)
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool(x)
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.pool(x)
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = F.relu(self.conv4_4(x))
        x = self.pool(x)
        return x
'''
def get_model(num_classes, in_channel):
    model = VGG19(num_classes, in_channel)
    return model