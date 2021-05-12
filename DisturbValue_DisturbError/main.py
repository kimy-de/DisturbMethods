# Input
import argparse

# Data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import datasets
import gridsearch

# ANN
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

# Loss
from sklearn.metrics import mean_squared_error

# Plot
import matplotlib.pyplot as plt

class TensorData(Dataset):

    def __init__(self, x_data, y_data):
        self.x_data = torch.FloatTensor(x_data)
        self.y_data = torch.FloatTensor(y_data)
        self.len = self.y_data.shape[0]

    def __getitem__(self, index):

        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

def noise_generator(x, alpha):
    noise = torch.normal(0, 1e-2, size=(len(x), 1))
    noise[torch.randint(0, len(x), (int(len(x)*(1-alpha)),))] = 0

    return noise

def disturberror(outputs, values, interval):
    epsilon = 1e-8
    e = values - outputs
    for i in range(len(e)):
        if (e[i] < epsilon) & (e[i] >= 0):
            values[i] = values[i] + e[i] / interval
        elif (e[i] > -epsilon) & (e[i] < 0):
            values[i] = values[i] - e[i] / interval

    return values

def minmax(x):
    for i in range(x.shape[1]):
        x[:,i] = (x[:,i] - np.min(x[:,i]))/(np.max(x[:,i])-np.min(x[:,i]))

    return x

class Baseline(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.fc1 = nn.Linear(n_features, n_features, bias=True)
        self.fc2 = nn.Linear(n_features, 5, bias=True)
        self.fc3 = nn.Linear(5, 1, bias=True)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))

        return x

class Dropout_model(nn.Module):
    def __init__(self, n_features, drop_rate):
        super().__init__()
        self.fc1 = nn.Linear(n_features, n_features, bias=True)
        self.fc2 = nn.Linear(n_features, 5, bias=True)
        self.fc3 = nn.Linear(5, 1, bias=True)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(F.relu(self.fc2(x)))
        x = torch.sigmoid(self.fc3(x))

        return x

if __name__ == "__main__":
    # !python main.py --dropout 'y' --dv 'n' --de 'n' --l2 'n' --dataset 'housing'--epoch 100
    parser = argparse.ArgumentParser(description='reg')
    parser.add_argument('--dropout', default='n', type=str, help='dropout')
    parser.add_argument('--dataset', default='boston', type=str, help='dataset')
    parser.add_argument('--dv', default='n', type=str, help='data type')
    parser.add_argument('--de', default='n', type=str, help='data type')
    parser.add_argument('--l2', default='n', type=str, help='data type')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--epoch', default=100, type=int, help='training epoch')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--dv_annealing', default='n', type=str, help='cos annealing')
    parser.add_argument('--T', default=80, type=int, help='cos frequency')
    
    args = parser.parse_args()
    print(args)

    # Dataset
    X, Y = datasets.dataloader(args.dataset)
    print("Completed reading the dataset.")
    
    # Scaling
    X = minmax(X)
    Y = minmax(Y)
    n_features = X.shape[1]

    print("Completed scaling the data features.")

    # Grid Search - dropout, dv, de, l2
    grid_on = [args.dropout, args.dv, args.de, args.l2]
    grid_search1, grid_search2, switch  = gridsearch.grid(grid_on, args.dv_annealing)

    l2 = 0
    # Training
    test_RMSE_set = []
    test_std_set = []
    for g1 in grid_search1:
        for g2 in grid_search2:
            if switch == 0:
                drop_rate = g2
            elif switch == 1:  
                alpha = g2
            elif switch == 2:  
                interval = g2 
            elif switch == 3:  
                l2 = g2     
            elif switch == 4:  
                drop_rate = g1
                alpha = g2
            elif switch == 5:  
                drop_rate = g1
                interval = g2    
            elif switch == 6:  
                drop_rate = g1
                l2 = g2  
            elif switch == 7:  
                alpha = g1
                interval = g2 
            elif switch == 8:  
                alpha = g1
                l2 = g2  
            elif switch == 9:  
                interval = g1
                l2 = g2 
            elif switch == 10:
                print("No regularization")
                
            trials = 1
            train_rmse = []
            val_rmse = []
            test_rmse = []

            while trials <= 20:

                # Data Set
                X_train, X_tmp, Y_train, Y_tmp = train_test_split(X, Y, test_size=0.3)
                X_val, X_test, Y_val, Y_test = train_test_split(X_tmp, Y_tmp, test_size=0.5)
    
                # Mini Batch
                trainsets = TensorData(X_train, Y_train)
                trainloader = torch.utils.data.DataLoader(trainsets, batch_size=args.batch_size, shuffle=True)

                valsets = TensorData(X_val, Y_val)
                valloader = torch.utils.data.DataLoader(valsets, batch_size=args.batch_size, shuffle=False)
    
                testsets = TensorData(X_test, Y_test)
                testloader = torch.utils.data.DataLoader(testsets, batch_size=args.batch_size, shuffle=False)
    
                # Model
                if args.dropout == 'n':
                    model = Baseline(n_features)
                else:
                    model = Dropout_model(n_features, drop_rate)
    
                criterion = nn.MSELoss()
                optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=l2)
               
                n = len(trainloader)
                check = 0
                st_rmse = 10
                for epoch in range(args.epoch):
                    model.train()
                    
                    # testing fixed vaules
                    alpha_min = 0.01
                    alpha_max = 0.10
                    T_i = args.T   
                    for i, data in enumerate(trainloader, 0):
    
                        if args.dv_annealing == 'y':
                            T = i % T_i
                            alpha =alpha_min + .5*(alpha_max - alpha_min)*(1+np.cos(np.pi*T/T_i)) # Top-down
                            #alpha =alpha_max - .5*(alpha_max - alpha_min)*(1+np.cos(np.pi*T/T_i)) # Bottom-up
                            
                        inputs, values = data
    
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        
                        if args.dv == 'y':
                            values = values + noise_generator(values, alpha)
    
                        if args.de == 'y':
                            values = disturberror(outputs, values, interval)
    
                        loss = criterion(outputs, values)
                        loss.backward()
                        optimizer.step()

    
                    predictions = torch.tensor([], dtype=torch.float)

                    with torch.no_grad():
                        model.eval()
                        train_predictions = torch.tensor([], dtype=torch.float)
                        train_actual = torch.tensor([], dtype=torch.float)
                        for data in trainloader:
                            inputs, values = data
                            outputs = model(inputs)

                            train_predictions = torch.cat((train_predictions, outputs), 0)
                            train_actual = torch.cat((train_actual, values), 0)

                    train_pred = train_predictions.detach().numpy()
                    train_actual_values = train_actual.detach().numpy()

                    div_condition = np.sqrt(mean_squared_error(train_pred, train_actual_values))

                    with torch.no_grad():
                        model.eval()
                        predictions = torch.tensor([], dtype=torch.float)

                        for data in valloader:
                            inputs, values = data
                            outputs = model(inputs)

                            predictions = torch.cat((predictions, outputs), 0)

                    predictions = predictions.numpy()
                    val_l = np.sqrt(mean_squared_error(predictions, Y_val))

                    if val_l <= st_rmse:

                        st_rmse = val_l
                        train_tmp = div_condition

                        torch.save(model.state_dict(), './val_best.pth')
                        #print(train_tmp,st_rmse)

                #print(trials, train_tmp, div_condition)
                if div_condition < 0.1:

                    train_rmse.append(train_tmp)
                    val_rmse.append(st_rmse)
                    trials += 1

                    predictions = torch.tensor([], dtype=torch.float)
                    model.load_state_dict(torch.load('./val_best.pth'))
                    with torch.no_grad():
                        model.eval()

                        for data in testloader:
                            inputs, values = data
                            outputs = model(inputs)

                            predictions = torch.cat((predictions, outputs), 0)

                    predictions = predictions.numpy()
                    test_rmse.append(np.sqrt(mean_squared_error(predictions, Y_test)))

            print("#####################################################")
            if len(grid_search1) > 1:
                print('Grid value1: ',g1)
                print('Grid value2: ',g2)
            else:
                print('Grid value: ',g2)
            print("Test RMSE: ", np.mean(np.array(test_rmse)).round(5), "Test std: ", np.std(np.array(test_rmse)).round(5))
            print("Val RMSE: ", np.mean(np.array(val_rmse)).round(5), "Val std: ",
                  np.std(np.array(val_rmse)).round(5))
            print("Train RMSE: ", np.mean(np.array(train_rmse)).round(5), "Train std: ", np.std(np.array(train_rmse)).round(5))
            print("#####################################################")
            test_RMSE_set.append(np.mean(np.array(test_rmse)).round(5))
            test_std_set.append(np.std(np.array(test_rmse)).round(5))
            

