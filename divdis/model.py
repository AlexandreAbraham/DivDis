import torch
from torch import nn

              
class DivDis(nn.Module):
    
    def __init__(self, backbone, n_output, n_heads, n_classes, lambda_mi=1., lambda_reg=1.):
        super().__init__()
        self.backbone = backbone
        self.heads = [nn.Sequential(nn.Linear(n_output, n_classes), nn.Softmax(dim=1)) for _ in range(n_heads)]
        self.active_head = None
        
    def set_active_head(self, active_head):
        if not 0 <= active_head < len(self.heads):
            raise ValueError('Invalid_head')
        self.active_head = active_head
    
    def forward(self, x):
        t = self.backbone(x)
        if self.training:
            # Training mode, predict with all heads
            preds = [head(t) for head in self.heads]
            return torch.stack(preds, dim=1)
        else:
            return self.heads[self.active_head](t)
        
    def parameters(self):
        for parameters in self.backbone.parameters():
            yield parameters
        for head in self.heads:
            for parameters in head.parameters():
                yield parameters