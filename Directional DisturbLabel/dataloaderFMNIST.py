import torch
from torchvision import datasets, transforms

class dataFMNIST():
    def __init__(self, augmentation, batch_size, test_batch_size, num_workers):
        super(dataFMNIST, self).__init__()
        self.batch_size = batch_size
        self.test_batch_size=test_batch_size
        self.num_workers=num_workers
        
        if augmentation == True:
            self.transformation=transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.5),
#                               transforms.CenterCrop((24)),
#                               transforms.RandomRotation(degrees=10),
#                               transforms.RandomCrop(28, pad_if_needed=True),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,),(0.5,),)
#                               transforms.Normalize((0.1307,), (0.3081,),)
                               ])
        else:
            self.transformation=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,),(0.5,),)
                               ])
    
    def loadData(self):
        self.train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('data', train=True, download=True,
                           transform=self.transformation),
            batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True)
            
#        self.test_loader = torch.utils.data.DataLoader(
#            datasets.MNIST('data', train=False, transform=transforms.Compose([
#                transforms.ToTensor(),
#                transforms.Normalize((0.1307,), (0.3081,))
#                ])),
#            batch_size=self.test_batch_size, shuffle=True,
#            num_workers=self.num_workers, pin_memory=True)
        self.test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('data', train=False, transform=self.transformation),
            batch_size=self.test_batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True)
    
        return self.train_loader, self.test_loader
    
    
    

    
        
