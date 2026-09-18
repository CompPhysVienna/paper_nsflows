from torch import nn

class base_transform(nn.Module):
    
    def __init__(self, device):
        super(base_transform, self).__init__()
        
        self.device = device
    

    def F_data2network(self):
        raise NotImplementedError
    
    
    def F_network2data(self):        
        raise NotImplementedError