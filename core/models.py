import torch
import torch.nn as nn
from collections import OrderedDict

class Net(nn.Module):
    def __init__(self, layers, act):
        super(Net, self).__init__()

        self.layers = layers
        
        layers_dict = OrderedDict()
        for l in range(len(self.layers)-2):
            in_dim = layers[l]
            out_dim = layers[l+1]
            
            layers_dict['linear'+str(l+1)] = nn.Linear(in_dim, out_dim)
            
            if act=='relu':
                layers_dict['act'+str(l+1)] = nn.ReLU()
            elif act=='tanh':
                layers_dict['act'+str(l+1)] = nn.Tanh()
            else:
                raise Exception('Error: unknow activation function.')
            #
            
        #
        
        in_dim = self.layers[-2]
        out_dim = self.layers[-1]
        
        layers_dict['linear'+str(len(self.layers)-1)] = nn.Linear(in_dim, out_dim)
        
        self.net = nn.Sequential(layers_dict)
        
    def forward(self, X):
        return self.net(X)

def get_fcnn_regressor(in_dim, out_dim, hidden_depth=2, hidden_width=32, act='tanh'):
    
    layers = [in_dim] + [hidden_width]*hidden_depth + [out_dim]
    
    return Net(layers, act)


