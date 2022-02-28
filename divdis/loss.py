import torch
from torch import nn
from einops import rearrange


def mutual_info_loss(probs):
    """ Input: predicted probabilites on target batch. 
    
    Note: This function is a copy/paste from the paper appendix
    """
    B, H, D = probs.shape # B=batch_size, H=heads, D=pred_dim
    marginal_p = probs.mean(dim=0) # H, D
    marginal_p = torch.einsum("hd,ge->hgde", marginal_p, marginal_p) # H, H, D, D
    marginal_p = rearrange(marginal_p, "h g d e -> (h g) (d e)") # H^2, D^2

    joint_p = torch.einsum("bhd,bge->bhgde", probs, probs).mean(dim=0) # H, H, D, D
    joint_p = rearrange(joint_p, "h g d e -> (h g) (d e)") # H^2, D^2

    kl_divs = joint_p * (joint_p.log() - marginal_p.log())
    kl_grid = rearrange(kl_divs.sum(dim=-1), "(h g) -> h g", h=H) # H, H
    pairwise_mis = torch.triu(kl_grid, diagonal=1) # Get only off-diagonal KL divergences

    return pairwise_mis.mean()

           
class DivDisLoss(nn.Module):
    
    def __init__(self, class_probas, lambda_mi=1., lambda_reg=1.):
        super().__init__()
        # self.loss_xent = nn.CrossEntropyLoss()
        self.loss_xent = nn.NLLLoss()
        self.loss_kld = nn.KLDivLoss(reduction='batchmean')
        self.class_probas = class_probas
        self.lambda_mi = lambda_mi
        self.lambda_reg = lambda_reg
    
    def __call__(self, prediction, target, test_prediction, verbose=None):
        """ Input: predicted probabilites on target batch. """
        B, H, D = prediction.shape # B=batch_size, H=heads, D=pred_dim
        
        N = test_prediction.shape[0]
        
        l_xent = sum([self.loss_xent(prediction[:, h].log(), target) for h in range(H)])
        l_mi = self.lambda_mi * mutual_info_loss(test_prediction)
        if self.lambda_reg is not None:
            l_kld = self.lambda_reg * sum([self.loss_kld(nn.functional.one_hot(test_prediction[:, h].argmax(axis=1), num_classes=self.class_probas.shape[0]).float().mean(axis=0).log(), self.class_probas) for h in range(H)])
        else:
            l_kld = torch.tensor(0.)
            
        if verbose is not None and verbose >= 1:
            print("Loss Xent", l_xent.item())
            print("Loss MI", l_mi.item())
            print("Loss reg", l_kld.item())
        
        return l_xent + l_mi + l_kld