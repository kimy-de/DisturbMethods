""" Images folder shared here: 
https://drive.google.com/drive/folders/1o8q8uvFARF_u4hMy-HeadIVrdSHsY2co?usp=sharing
or original source here : https://www.kaggle.com/puneet6060/intel-image-classification

"""

import torch
from torchvision import datasets, transforms

class dataINTEL():
    def __init__(self, augmentation, batch_size, test_batch_size, num_workers):
        super(dataINTEL, self).__init__()
        self.batch_size = batch_size
        self.test_batch_size=test_batch_size
        self.num_workers=num_workers
        
        if augmentation == True:
            self.transformation=transforms.Compose([
                               transforms.Resize((150, 150)),
                               transforms.RandomCrop((150,150),pad_if_needed=True),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                               ])
        else:
            self.transformation=transforms.Compose([
                               transforms.Resize((150, 150)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                               ])
    

    def loadData(self):
        self.train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder('/content/drive/My Drive/Colab Notebooks/disturblabel-master 2/archive/seg_train/seg_train',
                           transform=self.transformation),
            batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True)
            
        self.test_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder('/content/drive/My Drive/Colab Notebooks/disturblabel-master 2/archive/seg_train/seg_train', 
            transform=transforms.Compose([
                transforms.Resize((150, 150)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])),
            batch_size=self.test_batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True)
    
        return self.train_loader, self.test_loader