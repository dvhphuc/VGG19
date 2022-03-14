import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
# the VGG19 architecture

class VGG19(nn.Module):
    def __init__(self,num_classes,in_channel):
        super(VGG19,self).__init__()
        self.Conv_layers = nn.Sequential(
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
            nn.Linear(25088,4096),
            nn.Relu(),
            nn.Dropout2d(0.5),
            nn.Linear(4096,4096),
            nn.Relu(),
            nn.Dropout2d(0.5),          
            nn.Liner(4096,out_features = self.num_classes)  
        )
    
    def forward(self,x):
        x = self.Conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.FC_Layers(x)
        return x

'''
class CNN(nn.Module):
  def __init__(self, in_channel, num_classes):
    super(CNN, self).__init__()
    self.CNN_Layers = Sequential(
        Conv2d(in_channel, 8, stride=1, kernel_size=3),
        ReLU(inplace=False),
        MaxPool2d(kernel_size=2, stride=2),
        Conv2d(8, 32, stride=1, kernel_size=3),
        ReLU(inplace=False),
        MaxPool2d(kernel_size=3, stride=2),
        Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        ReLU(inplace=False),
        Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        ReLU(inplace=False),
        MaxPool2d(kernel_size=2, stride=2),
        Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
        ReLU(inplace=False),
        Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
        ReLU(inplace=False),
        MaxPool2d(kernel_size=2, stride=2)
    )

    self.FC_Layers = Sequential(
        Dropout(inplace=False),
        Linear(8192, 4096, bias=True),
        ReLU(inplace=True),
        Dropout(inplace=False),
        Linear(4096, 1024, bias=True),
        ReLU(inplace=True),
        Dropout(inplace=False),
        Linear(1024, num_classes, bias=True)        
    )

  def forward(self, x):
    x = self.CNN_Layers(x)
    x = x.view(x.size(0), -1)
    x = self.FC_Layers(x)
    return x
'''

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