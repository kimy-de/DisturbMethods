import torch.nn as nn
import torch.nn.functional as F

'''
Architecture of modified LeNet from the paper for CIFAR10

A 32 × 32 × 3 image is passed through three units consisting
of convolution, ReLU and max-pooling operations.
[C5(S1P2)@32-MP3(S2)]-[C5(S1P2)@32-MP3(S2)]-
[C5(S1P2)@64-MP3(S2)]-FC64-D0.5-FC10.
'''
class LeNetC(nn.Module):
    def __init__(self, use_dropout):
        super(LeNetC, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2),
            
            nn.Conv2d(32, 32, 5, 1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2),

            nn.Conv2d(32, 64, 5, 1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2)
        )
        if use_dropout==True:
            self.classifier = nn.Sequential(
                nn.Linear(576, 64),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(64, 10)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(576, 64),
                nn.ReLU(True),
                nn.Linear(64, 10)
            )
            
    def get_features(self, x):
        return self.features(x)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
#%%    
class LeNetC100(nn.Module):
    def __init__(self, use_dropout):
        super(LeNetC100, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2),
            
            nn.Conv2d(32, 32, 5, 1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2),

            nn.Conv2d(32, 64, 5, 1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2)
        )
        if use_dropout==True:
            self.classifier = nn.Sequential(
                nn.Linear(576, 512),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(512, 100)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(576, 512),
                nn.ReLU(True),
                nn.Linear(512, 100)
            )
    
    def get_features(self, x):
        return self.features(x)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
#%%
'''
Architecture of modified LeNet from the paper for MNIST
[C5(S1P0)@32-MP2(S2)]-[C5(S1P0)@64-
MP2(S2)]-FC512-D0.5-FC10
'''
class LeNetM(nn.Module):
    def __init__(self, use_dropout):
        super(LeNetM, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 5, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 5, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        if use_dropout==True:
            self.classifier = nn.Sequential(
                nn.Linear(1024, 512),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(512, 10)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(1024, 512),
                nn.ReLU(True),
                nn.Linear(512, 10)
            )
            
    def get_features(self, x):
        return self.features(x)
  
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
#%%
''' Modified LeNet for MNIST with data augmentation'''
class LeNetMAug(nn.Module):
    def __init__(self, use_dropout):
        super(LeNetMAug, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 5, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 5, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        if use_dropout==True:
            self.classifier = nn.Sequential(
                nn.Linear(576, 512),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(512, 10)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(576, 512),
                nn.ReLU(True),
                nn.Linear(512, 10)
            )
    def get_features(self, x):
        return self.features(x)
  
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
        
'''
Architecture of modified LeNet from the paper for CIFAR 10 borrowed for INTEL
[C5(S1P2)@32-MP3(S2)]-[C5(S1P2)@32-MP3(S2)]-
[C5(S1P2)@64-MP3(S2)]-FC64-D0.5-FC10.
'''
class LeNetI(nn.Module):
    def __init__(self, use_dropout):
        super(LeNetI, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2),
            
            nn.Conv2d(32, 32, 5, 1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2),

            nn.Conv2d(32, 64, 5, 1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2)
        )
        if use_dropout==True:
            self.classifier = nn.Sequential(
                nn.Linear(64*17*17, 64),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(64, 6)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(64*17*17, 64),
                nn.ReLU(True),
                nn.Linear(64, 6)
            )
    def get_features(self, x):
        return self.features(x)
  
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

'''
Architecture of modified LeNet from the paper for CIFAR 10 borrowed for MCINDOOR2000
[C5(S1P2)@32-MP3(S2)]-[C5(S1P2)@32-MP3(S2)]-
[C5(S1P2)@64-MP3(S2)]-FC64-D0.5-FC10.
'''
class LeNetMC2(nn.Module):
    def __init__(self, use_dropout):
        super(LeNetMC2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2),
            
            nn.Conv2d(32, 32, 5, 1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2),

            nn.Conv2d(32, 64, 5, 1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2)
        )
        if use_dropout==True:
            self.classifier = nn.Sequential(
                nn.Linear(64*27*27, 64),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(64, 6)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(64*27*27, 64),
                nn.ReLU(True),
                nn.Linear(64, 6)
            )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

'''
Architecture of modified LeNet from the paper for CIFAR 10 borrowed for ART dataset
[C5(S1P2)@32-MP3(S2)]-[C5(S1P2)@32-MP3(S2)]-
[C5(S1P2)@64-MP3(S2)]-FC64-D0.5-FC10.
'''
class LeNetA(nn.Module):
    def __init__(self, use_dropout):
        super(LeNetA, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2),
            
            nn.Conv2d(32, 32, 5, 1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2),

            nn.Conv2d(32, 64, 5, 1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2)
        )
        if use_dropout==True:
            self.classifier = nn.Sequential(
                nn.Linear(64*27*27, 64),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(64, 5)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(64*27*27, 64),
                nn.ReLU(True),
                nn.Linear(64, 5)
            )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x