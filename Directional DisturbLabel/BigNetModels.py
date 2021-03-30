import torch.nn as nn
import torch.nn.functional as F
import math

'''
Architecture of BigNet from the paper for CIFAR10

[C5(S1P2)@128-MP3(S2)]-[C3(S1P1)@128-
D0.7-C3(S1P1)@256-MP3(S2)]-D0.6- [C3(S1P1)@512]-
D0.5-[C3(S1P1)@1024-MPS(S1)]-D0.4-FC10.

* need to recheck CrossmapLRN2d()
'''

class BigNet(nn.Module): #a model from pytorch tutorial
    def __init__(self, use_dropout):
        super(BigNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, 5, 1, padding=2)
        self.pool1 = nn.MaxPool2d(3, 2)
        self.LRN1 = nn.LocalResponseNorm(size=3, alpha=0.0001, beta=0.75, k= math.log(128)) #alpha and beta as in the paper of Imagenet
        self.conv2 = nn.Conv2d(128, 128, 3,1, padding=1)
        self.use_dropout = use_dropout
        self.dropout1 = nn.Dropout(0.7)
        self.conv3 = nn.Conv2d(128, 256, 3,1, padding=1)
        self.LRN2 = nn.LocalResponseNorm(size=3, alpha=0.0001, beta=0.75, k= math.log(256))
        self.dropout2 = nn.Dropout(0.6)
        self.conv4 = nn.Conv2d(256, 512, 3,1, padding=1)
        self.dropout3 = nn.Dropout(0.5)
        self.conv5 = nn.Conv2d(512, 1024, 3,1, padding=1)
        self.pool2 = nn.MaxPool2d(3, 2)
        self.LRN3 = nn.LocalResponseNorm(size=3, alpha=0.0001, beta=0.75, k= math.log(1024))
        self.dropout4 = nn.Dropout(0.4)
        self.fc1 = nn.Linear(1024*3*3, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.LRN1(x)
        x = F.relu(self.conv2(x))
        if self.use_dropout==True:
            x = self.dropout1(x)
        x = self.pool1(F.relu(self.conv3(x)))
        x = self.LRN2(x)
        if self.use_dropout==True:
            x = self.dropout2(x)
        x = F.relu(self.conv4(x))
        if self.use_dropout==True:
            x = self.dropout3(x)
        x = self.pool2(F.relu(self.conv5(x)))
        x = self.LRN3(x)
        if self.use_dropout==True:
            x = self.dropout4(x)
        x = x.view(x.size(0), 1024*3*3)
        x = F.relu(self.fc1(x))
        
        return x