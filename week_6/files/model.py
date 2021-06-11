import torch
import torch.nn as nn
import torch.nn.functional as F

Nos_grps = 2
dropout_value = 0.1

def get_normalization_layer(out_channels, normalization_type):
    if normalization_type == "batch":
        nl = nn.BatchNorm2d(out_channels)
    elif normalization_type == "group":
        group = Nos_grps
        nl = nn.GroupNorm(group, out_channels)
    elif normalization_type == "layer":
        nl = nn.GroupNorm(1, out_channels)

    return nn.Sequential(nl)


class Net(nn.Module):
    def __init__(self, normalization_layer_type, *args, **kwargs):
        super(Net, self).__init__()
        global dropout_value
        self.normalization_layer_type = normalization_layer_type

        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            get_normalization_layer(10, self.normalization_layer_type),
            nn.Dropout(dropout_value)
        ) 


        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            get_normalization_layer(16, self.normalization_layer_type),
            nn.Dropout(dropout_value)
        )

        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),            
        ) 

        self.pool1 = nn.Sequential(
            nn.MaxPool2d(2, 2),
        )

        

        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            get_normalization_layer(16, self.normalization_layer_type),
            nn.Dropout(dropout_value)
        ) 

        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            get_normalization_layer(16, self.normalization_layer_type),
            nn.Dropout(dropout_value)
        ) 

        
        
        


        self.pool2 = nn.Sequential(
            nn.MaxPool2d(2, 2)
        )

        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),            
        ) 

        

        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            get_normalization_layer(16, self.normalization_layer_type),
            nn.Dropout(dropout_value)
        )
        self.pool3 = nn.Sequential(
            nn.MaxPool2d(2, 2)
        ) 

        self.convblock8= nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            get_normalization_layer(16, self.normalization_layer_type),
            nn.Dropout(dropout_value)
        ) 

        
        self.gap= nn.Sequential(
            nn.AvgPool2d(kernel_size=3), 
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), bias=False) 
        )

         # output_size = 1

    def forward(self, x):
        x = self.convblock1(x)    #input-(28,28) Output-(28,28)   RF=3   input channels=1 output channels=10
        x = self.convblock2(x)    #input-(28,28) Output-(28,28)   RF=5   input channels=10 output channels=16
        x = self.convblock3(x)    #input-(28,28) Output-(28,28)   RF=5   input channels=16 output channels=10
        x = self.pool1(x)         #input-(28,28) Output-(14,14)   RF=6   input channels=10 output channels=10
        x = self.convblock4(x)    #input-(14,14) Output-(14,14)   RF=10   input channels=10 output channels=16
        x = self.convblock5(x)    #input-(14,14) Output-(14,14)   RF=14   input channels=16 output channels=16
        x = self.pool2(x)         #input-(14,14) Output-(7,7)     RF=16   input channels=16 output channels=16
        x = self.convblock6(x)    #input-(7,7) Output-(7,7)       RF=16   input channels=16 output channels=10
        x = self.convblock7(x)    #input-(7,7) Output-(7,7)       RF=24   input channels=10 output channels=16
        x = self.pool3(x)         #input-(7,7) Output-(3,3)       RF=28   input channels=16 output channels=16
        x = self.convblock8(x)    #input-(3,3) Output-(3,3)       RF=44   input channels=16 output channels=16


        x = self.gap(x)


        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
