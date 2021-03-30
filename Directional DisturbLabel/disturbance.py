import torch
import torch.nn.functional as F
   

class DisturbLabel(torch.nn.Module):
    def __init__(self, alpha, C):
        super(DisturbLabel, self).__init__()
        self.alpha = alpha
        self.C = C
        # Multinoulli distribution
        self.p_c = (1 - ((C - 1)/C) * (alpha/100))
        self.p_i = (1 / C) * (alpha / 100)

    def forward(self, y):
        
        if y.shape[0]==0:
            y_disturbed=y
        else:           
            # convert classes to index
            y_tensor = y
            y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    
            # create disturbed labels
            depth = self.C
            y_one_hot = torch.ones(y_tensor.size()[0], depth) * self.p_i
            y_one_hot.scatter_(1, y_tensor, self.p_c)
            y_one_hot = y_one_hot.view(*(tuple(y.shape) + (-1,)))
    
            # sample from Multinoulli distribution
            distribution = torch.distributions.OneHotCategorical(y_one_hot)
            y_disturbed = distribution.sample()
            y_disturbed = y_disturbed.max(dim=1)[1]  # back to categorical

        return y_disturbed
    