# Regularization: Extending DistrubLabel Method

## 1. Classification Task
PyTorch implementation of [DisturbLabel: Regularizing CNN on the Loss Layer](https://arxiv.org/abs/1605.00055) [CVPR 2016] extended with Directional DisturbLabel method.

This classification code is built on top of  https://github.com/amirhfarzaneh/disturblabel-pytorch/blob/master/README.md project
and utilizes implementation from ResNet 18 from https://github.com/huyvnphan/PyTorch_CIFAR10

### Directional DisturbLabel 
```
  if args.mode == 'ddl' or args.mode == 'ddldr':
      out = F.softmax(output, dim=1)
      norm = torch.norm(out, dim=1)
      out = out / norm[:, None]
      idx = []
      for i in range(len(out)):
          if out[i,target[i]] > .5:
              idx.append(i)
              
      if len(idx) > 0:
          target[idx] = disturb(target[idx]).to(device) 
```

### Usage

`python main_ddl.py --mode=dl --alpha=20`


### Most important arguments

`--dataset` - which data to use 

Possible values:


| value | dataset |
| ------ | ------ |
|MNIST    | [MNIST](http://yann.lecun.com/exdb/mnist/)     |
|FMNIST   |[Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist)           |
|CIFAR10      |[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)     |
|CIFAR100  |[CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)                        |
|ART     |[Art Images: Drawing/Painting/Sculptures/Engravings](https://www.kaggle.com/thedownhill/art-images-drawings-painting-sculpture-engraving)         |
|INTEL    |[Intel Image Classification](https://www.kaggle.com/puneet6060/intel-image-classification)         |

Default: MNIST

`-- mode` - regularization method applied

Possible values:

| value | method |
| ------ | ------ |
|noreg    |Without any regularization    |
|dl       |Vanilla DistrubLabel          |
|ddl      |Directional DisturbLabel      |
|dropout  |Dropout                       |
|dldr     |DistrubLabel+Dropout          |
|ddldl    |Directional DL+Dropout        |

Default: ddl

`--alpha` - alpha for vanilla Distrub label and Directional DisturbLabel 

Possible values: int from 0 to 100. 
Default: 20

`--epochs` - number of training epochs

Default: 100

## 2. Regression Task
### DisturbValue
```
def noise_generator(x, alpha):
    noise = torch.normal(0, 1e-8, size=(len(x), 1))
    noise[torch.randint(0, len(x), (int(len(x)*(1-alpha)),))] = 0

    return noise
```

### DisturbError 
```
def disturberror(outputs, values):
    epsilon = 1e-8
    e = values - outputs
    for i in range(len(e)):
        if (e[i] < epsilon) & (e[i] >= 0):
            values[i] = values[i] + e[i] / 4
        elif (e[i] > -epsilon) & (e[i] < 0):
            values[i] = values[i] - e[i] / 4

    return values
```

### Datasets
1) Boston: 506 instances, 13 features
2) Bike Sharing: 731 instances, 13 features
3) Air Quality(AQ): 9357 instances, 10 features
4) make_regression(MR): 5000 instances, 30 features (random sample for regression)
5) Housing Price - Kaggle(HP): 1460 instances, 81 features
6) Student Performance (SP): 649 instances, 13 features (20 - categorical were dropped)
7) Superconductivity Dataset (SD): 21263 instances, 81 features
8) Communities & Crime (CC): 1994 instances, 100 features
9) Energy Prediction (EP): 19735 instancies, 27 features

### Experiment Setting
#### Model: MLP which has 3 hidden layers
#### Result: Averaged over 20 runs
#### Hyperparameters: Using grid search options

1) Boston: epoch 300, lr 0.001
2) Bike Sharing: epoch 100, lr 0.0001
3) Air Quality: epoch 50, lr 0.001
4) make_regression: epoch 50, lr 0.001
5) Housing Price - Kaggle: epoch 100, lr 0.0001
6) Student Performance (SP): epoch 100
7) Superconductivity Dataset (SD): epoch 50
8) Communities & Crime (CC): epoch 150
9) Energy Prediction (EP): epoch 50

- Implementation examples - main.py
```
python main_new.py --de y --dataset "bike" --dv_annealing y --epoch 100 --T 80
python main_new.py --de y --dv y --dataset "bike" -epoch 100
python main_new.py --de y --l2 y --dataset "air" -epoch 100
python main_new.py --dv y --dv_annealing y --dataset "air" -epoch 100 #for annealing setting dv should be "y"

--dataset: 'bike', 'air', 'boston', 'housing', 'make_sklearn', 'superconduct', 'energy', 'crime', 'students'
--dropout, --dv(disturbvalue), --de(disturberror), --l2, --dv_annealing: (string) y / n
--lr: (float)
--batch_size, --epoch, --T(cos annealing T): (int)
-- default dv_annealing: alpha_min = 0.05, alpha_max = 0.12, T_i = 80
```









