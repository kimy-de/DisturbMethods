''' images: https://drive.google.com/drive/folders/1ykI4pF_Ok3VY1aLj0Ozk9OT2j1decqcC?usp=sharing '''


import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from torchvision.transforms import Compose, ToTensor, Resize
from torch.utils.data import DataLoader


#    print(len(datasets['train']))
#    print(len(datasets['test']))
#    print(datasets['train'].dataset)

class dataMC2000():
    def __init__(self, augmentation, batch_size, test_batch_size, num_workers):
        super(dataMC2000, self).__init__()
        
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.augmentation= augmentation

    

    def loadData(self):
        
        dataset = ImageFolder('/content/drive/My Drive/Colab Notebooks/disturblabel-master 2/MCIndoor2000')
        train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=0.5)
        train_data = Subset(dataset, train_idx)
        if self.augmentation == True:
            print('augmenting')
            train_data.dataset.transform=transforms.Compose([
                               transforms.Resize((224, 224)),
                               transforms.RandomCrop((224,224),pad_if_needed=True),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
                               ])
        else:
            train_data.dataset.transform=transforms.Compose([
                               transforms.Resize((224, 224)),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
                               ])
                  
        
        test_data = Subset(dataset, test_idx)
        test_data.dataset.transform=transforms.Compose([
                               transforms.Resize((224, 224)),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
                               ])
                               
        self.train_loader = torch.utils.data.DataLoader(
            train_data,batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True)
          
        self.test_loader = torch.utils.data.DataLoader(
            test_data,batch_size=self.test_batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True)
            
        print(f'Train: {len(train_idx)}, Class: {train_data.dataset.classes}')  
        print(f'Train: {len(test_idx)}, Class: {test_data.dataset.classes}')
        
        return self.train_loader, self.test_loader