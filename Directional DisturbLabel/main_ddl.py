import os
import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from LeNetModels import LeNetC, LeNetM, LeNetMAug, LeNetI, LeNetC100, LeNetMC2, LeNetA
from BigNetModels import BigNet
from dataloaderCIFAR10 import dataCIFAR10
from dataloaderMNIST import dataMNIST
from dataloaderFMNIST import dataFMNIST
from dataloaderCIFAR100 import dataCIFAR100
from dataloaderINTEL import dataINTEL
from dataloaderMCIndoor2000 import dataMC2000
from dataloaderART import dataART
from disturbance import DisturbLabel
import torchvision
from models import *


from tensorboardX import SummaryWriter
import numpy as np

def main():
    # parameters
    parser = argparse.ArgumentParser(description='PyTorch Directional DisturbLabel')
    parser.add_argument('--mode', help='Select regularization mode', type=str, default='ddl')
    parser.add_argument('--alpha', help='Select hyperparameter value for DL/DDL ', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--test-batch-size', type=int, default=32)
    parser.add_argument('--epochs', help='Select number of epochs', type=int, default=100)
    parser.add_argument('--lr', help='Select learning rate', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--device', help='Select device (CPU/GPU)', type=str, choices=['cpu','gpu'], default='gpu')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--dataaug', help='Select if data augmentation is applied', type=bool, choices=['True','False'], default=False)
    parser.add_argument('--model', help='Select model', type=str, choices=['lenet','resnet18'], default='lenet')
    parser.add_argument('--dataset', help='Select dataset', type=str,choices=['MNIST','FMNIST', 'CIFAR10', 'CIFAR100', 'INTEL', 'ART'], default='MNIST')
    parser.add_argument('--logf',  help='Select folder for output logs', type=str, default='def')
    parser.add_argument('--init', type=str, default='def') #xavier_uniform, xavier_normal
    parser.add_argument('--optim', type=str, help='Select optimizer', choices=['SGD','Adam'], default='SGD') #SGD, Adam
    parser.add_argument('--resnetimp', type=str, default='pytorch') #pytorch, custom

    args = parser.parse_args()


    global writer
    mode=args.mode
    LOGS_PATH='logs'+'/'+str(args.dataset)+'/'+str(args.model)

    if args.logf=='def':
        if mode=='dl' or mode=='dldr'or mode=='ddl' or mode == 'ddldr':
            FOLDER_NAME=mode+'_alpha_'+str(args.alpha)   
        else:
            FOLDER_NAME=mode
    
    else:
        FOLDER_NAME=args.logf
    

    writer = SummaryWriter(os.path.join(LOGS_PATH, FOLDER_NAME, 'tb'))
    # GPU/CPU
    device = torch.device('cuda' if args.device == 'gpu' else 'cpu')
#%%
    # Reading data

    if args.dataset == 'CIFAR10':
        dataset = dataCIFAR10(args.dataaug, args.batch_size, args.test_batch_size, args.num_workers)
        classes = 10
    elif args.dataset == 'MNIST':
        dataset = dataMNIST(args.dataaug, args.batch_size, args.test_batch_size, args.num_workers)
        classes = 10
    elif args.dataset == 'FMNIST':
        dataset = dataFMNIST(args.dataaug, args.batch_size, args.test_batch_size, args.num_workers)
        classes = 10
    elif args.dataset == 'CIFAR100':
        dataset = dataCIFAR100(args.dataaug, args.batch_size, args.test_batch_size, args.num_workers)
        classes = 100
    elif args.dataset == 'INTEL':
        dataset = dataINTEL(args.dataaug, args.batch_size, args.test_batch_size, args.num_workers)
        classes = 6
    elif args.dataset=='MC2000': #only orginal images
        dataset=dataMC2000(args.dataaug, args.batch_size, args.test_batch_size, args.num_workers)
        classes=3
    elif args.dataset=='ART': #only orginal images
        dataset=dataART(args.dataaug, args.batch_size, args.test_batch_size, args.num_workers)
        classes=5

    train_loader, test_loader=dataset.loadData()
    

#%%
    # Model
    if mode in ['dropout', 'dldr', 'ddldr']:
        use_dropout=True
    else:
        use_dropout=False
        
    milestones=[40, 60, 80] 
    
    if args.model=='lenet':
        milestones=[40, 60, 80]
        if args.dataset =='CIFAR10':
            model = LeNetC(use_dropout).to(device)
            
        elif args.dataset =='CIFAR100':
            model = LeNetC100(use_dropout).to(device)
            
        elif args.dataset =='INTEL':
            model = LeNetI(use_dropout).to(device)
        
        elif args.dataset =='FMNIST':
            model = LeNetM(use_dropout).to(device)
            
           
        elif args.dataset =='MNIST':# or args.dataset =='FMNIST':
            if args.dataaug==True:
                model =  LeNetMAug(use_dropout).to(device)
            else:
                model = LeNetM(use_dropout).to(device)
        
        elif args.dataset =='MC2000':
            model = LeNetMC2(use_dropout).to(device)
            
        elif args.dataset =='ART':
            model = LeNetA(use_dropout).to(device)

    if args.model=='bignet' and args.dataset =='CIFAR10':
        milestones=[200, 300, 400]
        model=BigNet(use_dropout).to(device)
        print('BigNet')

    if args.model =='resnet18' and args.resnetimp == 'pytorch' and args.dataset in ['CIFAR10', 'CIFAR100','INTEL','MNIST','FMNIST']:
        milestones=[40, 60, 80]
        model = torchvision.models.resnet18(pretrained=False)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        if args.dataset =='MNIST' or args.dataset =='FMNIST':
            #(conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
            model.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, classes)

        if use_dropout == True:
            model.fc = nn.Sequential(nn.Dropout2d(0.5), nn.Linear(num_ftrs, classes))
            
    if args.model =='resnet18' and args.resnetimp == 'custom' and args.dataset in ['CIFAR10', 'CIFAR100','INTEL', 'ART']:
        milestones=[40, 60, 80]
        model = ResNet18(num_classes=classes, use_dropout = use_dropout, channels=3)
        
    if args.model =='resnet18' and args.resnetimp == 'custom' and args.dataset in ['MNIST','FMNIST']:
        milestones=[40, 60, 80]
        model = ResNet18(num_classes=classes, use_dropout = use_dropout, channels=1)
        


    if args.model == 'resnet50' and args.dataset in ['CIFAR10', 'CIFAR100', 'INTEL', 'MNIST', 'FMNIST']:
        milestones = [40, 60, 80]
        model = torchvision.models.resnet50(pretrained=False)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        if args.dataset == 'MNIST' or args.dataset == 'FMNIST':
            # (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
            model.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, classes)

        if use_dropout == True:
            model.fc = nn.Sequential(nn.Dropout2d(0.5), nn.Linear(num_ftrs, classes))

#%% Optimizer + scheduler
    if args.optim=='SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    elif args.optim=='Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    
        

#%% Weight initialization
    def weights_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            if args.init=='xavier_uniform':
                torch.nn.init.xavier_uniform_(m.weight)
            elif args.init=='xavier_normal':
                torch.nn.init.xavier_normal_(m.weight)


    if args.init !='def':
        model.apply(weights_init)
        
#%%
    criterion = nn.CrossEntropyLoss().to(device)
    disturb = None


    if mode in ['dl','dldr','ddl','ddldr']:
        disturb = DisturbLabel(alpha=args.alpha, C=classes)
        
        
#%%
    # Train and Test
    start=time.time()
    output_dump=np.zeros((args.epochs, 5))

    
    for epoch in range(1, args.epochs + 1):
        epoch_start=time.time()
        train_error, train_loss, num_disturbed=train(args, model, device, train_loader, optimizer, criterion, epoch, disturb)
        test_error, test_loss=test(args, model, device, test_loader, criterion, epoch)
        if args.optim=='SGD':
            scheduler.step()
        output_dump[epoch-1,0]=train_error
        output_dump[epoch-1,1]=test_error
        output_dump[epoch-1,2]=train_loss
        output_dump[epoch-1,3]=test_loss
        output_dump[epoch-1,4]=num_disturbed
        
        writer.add_scalars('{0}/loss'.format(args.mode), {'train':train_loss,
                                    'test':test_loss}, epoch)
        writer.add_scalars('{0}/error'.format(args.mode), {'train':train_error,
                                    'test':test_error}, epoch)
        print('Elapsed: {0:.2f} seconds for epoch, {1:.2f} minutes in total\n'.format(
                                            time.time()-epoch_start, (time.time()-start)/60))
    
    print("Smallest Train Error={0:.2f}% is achieved at epoch {1}".format(np.min(output_dump[:,0], axis=0), np.argmin(output_dump[:,0], axis=0)))
    print("Smallest Test Error={0:.2f}% is achieved at epoch {1}".format(np.min(output_dump[:,1], axis=0), np.argmin(output_dump[:,1], axis=0)))
    np.save((LOGS_PATH+'/'+FOLDER_NAME+'_ErrAndLoss'), output_dump)
    print('Log folder: {0}'.format(LOGS_PATH))
    print('Error and Loss logs are saved to {0} file'.format(FOLDER_NAME+'_ErrAndLoss.npy'))
    
    MODELPATH = LOGS_PATH+'/'+FOLDER_NAME+'_model.pth'
    torch.save(model.state_dict(), MODELPATH)
    print('Model saved to {0} file'.format(FOLDER_NAME+'_model.pth'))
    
#%%
def train(args, model, device, train_loader, optimizer, criterion, epoch, disturb):
    model.train()
    correct = 0
    running_loss = 0.0
    disturbed_count = 0
  
    for batch_idx, (data, target) in enumerate(train_loader, 0):
        data, target = data.to(device), target.to(device)
        
        actual_target = target.clone()
            
        if args.mode == 'dl' or args.mode == 'dldr':
            target = disturb(target).to(device)

        model.to(device)            
        optimizer.zero_grad()
        output = model(data)
        
        if args.mode == 'ddl' or args.mode == 'ddldr':
            out = F.softmax(output, dim=1)
            norm = torch.norm(out, dim=1)
            out = out / norm[:, None]
            idx = []
            for i in range(len(out)):
                if out[i,target[i]] > .5:
                    idx.append(i)
            """
            if batch_idx % 500 ==0:
                print(len(idx))
            """
            if len(idx) > 0:
                target[idx] = disturb(target[idx]).to(device)                          
      
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() 
        

        # calculate error rate
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(actual_target.view_as(pred)).sum().item()
        

    train_error = 100 - (100. * correct / len(train_loader.dataset))
    train_loss=running_loss/(batch_idx+1)
    share_disturbed= disturbed_count*100 / len(train_loader.dataset)
    print('Epoch [{0}] Train Loss: {1:.4f} | Error: {2:.2f}% '.format(epoch, train_loss, train_error))

    return train_error, train_loss, share_disturbed


def test(args, model, device, test_loader, criterion, epoch):
    model.eval() #eval mode switches off applying dropout
    correct = 0
    running_test_loss = 0.0
    
    with torch.no_grad(): #no_grad prevents gradients calculation
          
        for batch_idx, (data, target) in enumerate(test_loader, 0):
            data, target = data.to(device), target.to(device)

            model.to(device) 
            output = model(data)
            loss = criterion(output, target)


            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            running_test_loss+=loss.item()

    test_error = 100 - (100. * correct / len(test_loader.dataset))
    test_loss=running_test_loss/(batch_idx+1)
    print('Epoch [{0}] Test Loss: {1:.4f} | Error: {2:.2f}%'.format(epoch, test_loss, test_error))
        
    return test_error, test_loss


if __name__ == '__main__':
    main()

