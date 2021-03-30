"""

 Dataset: https://drive.google.com/drive/folders/1zuuaw-n62at5Iy9kdc46bN8gslrrf57m?usp=sharing 
 
 
 """



import warnings
warnings.simplefilter("ignore", UserWarning)

import torch
from torchvision import datasets, transforms

#    print(len(datasets['train']))
#    print(len(datasets['test']))
#    print(datasets['train'].dataset)

class dataART():
    def __init__(self, augmentation, batch_size, test_batch_size, num_workers):
        super(dataART, self).__init__()
        
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers

        if augmentation == True:
            print('augmenting')
            self.transformation=transforms.Compose([
                               transforms.Resize((227, 227)),
                               transforms.RandomCrop((227,227),pad_if_needed=True),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
                               ])
        else:
            self.transformation=transforms.Compose([
                               transforms.Resize((227, 227)),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
                               ])

    

    def loadData(self):
        self.train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder('/content/drive/My Drive/Colab Notebooks/disturblabel-master 2/art_dataset/training_set',
            transform=self.transformation),
            batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True)
            
        self.test_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder('/content/drive/My Drive/Colab Notebooks/disturblabel-master 2/art_dataset/validation_set', 
            transform=transforms.Compose([
                transforms.Resize((227, 227)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
                ])),
            batch_size=self.test_batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True)
    
        return self.train_loader, self.test_loader