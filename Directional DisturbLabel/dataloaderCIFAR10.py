import torch
from torchvision import datasets, transforms


class dataCIFAR10():
    def __init__(self, augmentation, batch_size, test_batch_size, num_workers):
        super(dataCIFAR10, self).__init__()
        self.batch_size = batch_size
        self.test_batch_size=test_batch_size
        self.num_workers=num_workers
        
        if augmentation == True:
            #@TODO (?): add LocalContrastNormalization? https://github.com/pytorch/pytorch/issues/7773
            self.transformation=transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
#                               transforms.RandomCrop((32,32),pad_if_needed=True),
                               transforms.RandomCrop(32, padding=4),
                               transforms.ToTensor(),
                               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
#                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                               ])
        else:
            self.transformation=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
#                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                               ])
    
    def loadData(self):
        self.train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('data', train=True, download=True,
                           transform=self.transformation),
            batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True)

        self.test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])),
            batch_size=self.test_batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True)
    
        return self.train_loader, self.test_loader
    
        

        

    
    
    