from math import log2
import torch

class AverageMeter(object):
    """ Computes and stores the average and current value """
    def __init__(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class KL():
    def __init__(self, p,q):
        self.p=p
        self.q=q
        
    def calcKL(self):
        try:
            divergence=sum(self.p[i] * log2(self.p[i]/self.q[i]) for i in range(len(self.p)))
        except ValueError:
            #hack for computational stability in case some elements in the input are zeros
            maxvalue, maxind= torch.max(self.p, 0)
            #add a small value to each element of the vector so that none of them is 0
            self.p=torch.add(self.p, 0.00001) 
            #deduct added value from the largest element so that the sum remains = 1
            self.p[maxind]-=0.00001*10
            divergence=sum(self.p[i] * log2(self.p[i]/self.q[i]) for i in range(len(self.p)))
            print("Math domain error exception caught")
            pass
        
        return divergence

class Threshold():
    def __init__(self, valuesList, order, percentage):
        self.values=valuesList
        if order=='Top':
            self.order=True
        elif order=='Bottom':
            self.order=False
        self.percentage=percentage
        
    def calculate(self):
        values=self.values.copy()
        values.sort(reverse=self.order)
        threshold=values[round(self.percentage/100*len(values))]
        
        return threshold
        
