import torch
from torchvision import datasets, transforms
from urllib.request import Request, urlopen
from six.moves import urllib

opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

req = Request('http://www.cmegroup.com/trading/products/#sortField=oi&sortAsc=false&venues=3&page=1&cleared=1&group=1', headers={'User-Agent': 'Mozilla/5.0'})
webpage = urlopen(req).read()

class dataMNIST():
    def __init__(self, augmentation, batch_size, test_batch_size, num_workers):
        super(dataMNIST, self).__init__()
        self.batch_size = batch_size
        self.test_batch_size=test_batch_size
        self.num_workers=num_workers
        
        if augmentation == True:
            self.transformation=transforms.Compose([
                               transforms.RandomCrop((24,24),pad_if_needed=True),
#                    transforms.RandomHorizontalFlip(p=0.5),
#                               transforms.CenterCrop((24)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                               ])
            self.transformation_test=transforms.Compose([
                                transforms.CenterCrop((24,24)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                               ])
        else:
            self.transformation=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                               ])
            self.transformation_test=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                               ])
    
    def loadData(self):
        self.train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=True, download=True,
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
            datasets.MNIST('data', train=False, transform=self.transformation_test),
            batch_size=self.test_batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True)
    
        return self.train_loader, self.test_loader
    
    
    

    
        
