import torch

def soft_copy_param(ema_netG, netG, beta):
    netG_para = netG.state_dict()
    for name, param in ema_netG.named_parameters():
        param.data *= beta
        param.data += (1-beta) * netG_para[name].cpu().data

