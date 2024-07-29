import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict




class HiddenBlock(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0):
        super().__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        torch.nn.init.kaiming_uniform_(self.fc.weight, mode='fan_in', nonlinearity='relu')
        torch.nn.init.zeros_(self.fc.bias)
        self.drop = nn.Dropout(p=dropout)
        self.bn = nn.BatchNorm1d(hidden_size)
    def forward(self, x):
        x = F.relu(self.fc(x))
        x = self.drop(x)
        x = self.bn(x)
        return x
        
        
class TabNNmodel(nn.Module):
    def __init__(self, n_ord, dim_sizes, n_nom=0, n_numerical=0, n_label=0, dropouts=None, class_ratios=None, is_integer_encoder=False):
        super(TabNNmodel, self).__init__()
        self.n_layer = len(dim_sizes)
        self.n_ord = n_ord
        self.n_num = n_numerical
        self.n_nom = n_nom
        self.is_integer_encoder = is_integer_encoder
        dim_sizes = [n_ord + n_nom + n_numerical]+dim_sizes
        if dropouts is None:
            dropouts = [0]*(self.n_layer+1)
        elif isinstance(dropouts, float):
            dropouts = [dropouts]*(self.n_layer+1)
        self.dropouts = dropouts
        self.drop = nn.Dropout(p=dropouts[0])
        
        if is_integer_encoder:
            self.bn_ie = nn.BatchNorm1d(n_ord)
        if self.n_num>0:
            self.bn = nn.BatchNorm1d(n_numerical)
            
        self.hidden_blocks = nn.Sequential(OrderedDict([(f'hidden{i+1}', HiddenBlock(dim_sizes[i], dim_sizes[i+1], self.dropouts[i+1])) for i in range(len(dim_sizes)-1)]))
        
        if n_label > 0:
            if n_label <= 2: 
                self.fc_last = nn.Linear(dim_sizes[-1], 1) 
                if not class_ratios is None:
                    assert len(class_ratios) == 2, "class ratio should has length of 2"
                    positive_ratio = class_ratios[-1]
                    bias = -torch.log(torch.tensor(1 / positive_ratio - 1))
                    self.fc_last.bias.data.fill_(bias)
                else : torch.nn.init.zeros_(self.fc_last.bias)
            else :
                self.fc_last = nn.Linear(dim_sizes[-1], n_label)
                if not class_ratios is None:
                    assert len(class_ratios) == n_label, "class ratio should has length of n_label"
                    biases = [-torch.log(torch.tensor(1 / ratio - 1)) for ratio in class_ratios]
                    self.fc_last.bias.data = torch.tensor(biases)
                else : torch.nn.init.zeros_(self.fc_last.bias)
            torch.nn.init.xavier_uniform_(self.fc_last.weight) 
        else: 
            self.fc_last = nn.Linear(dim_sizes[-1], 1)
            torch.nn.init.uniform_(self.fc_last.weight) 
        self.n_label = n_label
           
    def forward(self, x_ord, x_nom, x_num=None):
        if self.is_integer_encoder:
            x_ord = self.bn_ie(x_ord.double())
        x = torch.cat((x_ord, x_nom), 1) if x_nom is not None else x_ord
        x = self.drop(x.double())
        if self.n_num>0:
            x_num = self.bn(x_num.double())
            x = torch.cat((x, x_num), 1)
        x = self.hidden_blocks(x)
        x = self.fc_last(x)
        if self.n_label > 0:
            if self.n_label <= 2: 
                x = F.sigmoid(x)
            else : x = F.softmax(x, dim=-1)
        return x