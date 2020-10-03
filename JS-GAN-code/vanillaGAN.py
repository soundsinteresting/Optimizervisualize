import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, grad
import numpy as np
import os
import argparse
from torchvision.utils import save_image
import networks
import datasets
import utils
from torch.utils.data import Dataset, DataLoader, TensorDataset
import copy
import moving_average


def getSavePath():
    fixz = 'fixz' if args.fixz else 'randz'
    sn = 'withspcenorm' if args.norm == 'sn' else 'withbatchnorm'
    dir_name = './{}_struct{}_GfeatureNum{}_DfeatureNum{}_imgsize{}_loss{}_{}/'.format(
        args.dataset, args.structure, args.Gnum_features, args.Dnum_features, args.image_size, args.losstype, sn)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    dir_name = dir_name + 'Vanilla_{}_{}_dlr{}_glr{}_diter{}_giter{}_b1{}_b2{}_GfeatureNum{}_DfeatureNum{}_batchsize{}'.format(args.dataset.upper(),
                                            fixz, args.d_lr, args.g_lr, args.d_freq, args.g_freq, args.beta1,
                                            args.beta2, args.Gnum_features, args.Dnum_features, args.batch_size)

    dir_name += '_ematrick' if args.ema_trick else ''
    dir_name += args.specname
    dir_name += 'seed{}/'.format(args.seed)

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    save_path = dir_name + 'models/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    return dir_name, save_path

BLL = torch.nn.BCEWithLogitsLoss()

def get_gloss(type, dis_fake):
    if type == 'log':
        return torch.mean(F.softplus(-dis_fake))
    elif type == 'hinge':
        return -torch.mean(dis_fake)

def get_dloss(type, dis_fake, dis_real):
    if type == 'log':
        return torch.mean(F.softplus(-dis_real)) + torch.mean(F.softplus(dis_fake))
    elif type == 'hinge':
        return torch.mean(torch.relu(1. - dis_real)) + torch.mean(torch.relu(1. + dis_fake))

def train():
    dir_name, save_path = getSavePath()
    if args.norm == 'sn':
        netG, netD = networks.getGD_SN(args.structure, args.dataset, args.Gnum_features, args.Dnum_features)
    elif args.norm == 'bn':
        netG, netD = networks.getGD_batchnorm(args.structure, args.dataset, args.Gnum_features, args.Dnum_features,
                                              dim_z=args.input_dim)

    if args.ema_trick:
        ema_netG_9999 = copy.deepcopy(netG)

    netG.cuda()
    netD.cuda()
    g_optimizer = torch.optim.Adam(netG.parameters(), lr=args.g_lr, betas=(args.beta1, args.beta2))
    d_optimizer = torch.optim.Adam(netD.parameters(), lr=args.d_lr, betas=(args.beta1, args.beta2))

    g_losses, d_losses = [], []
    grad_normD, grad_normG = [], []

    loader = datasets.getDataLoader(args.dataset, args.image_size, batch_size=args.batch_size, shuffle=not args.fixz)
    data_iter = iter(loader)

    data_num = len(loader.dataset)
    zs = TensorDataset(torch.randn(data_num, args.input_dim))
    zloader = DataLoader(zs, batch_size=args.batch_size, shuffle=not args.fixz, num_workers=0, drop_last=True)
    z_iter = iter(zloader)

    for i in range(1, args.num_iters+1):
        if i >= args.lr_decay_start:
            utils.decay_lr(g_optimizer, args.num_iters, args.lr_decay_start, args.g_lr)
            utils.decay_lr(d_optimizer, args.num_iters, args.lr_decay_start, args.d_lr)

        if i == 1:
            torch.save(netG.state_dict(), save_path + 'G_epoch0.pth')
            torch.save(netD.state_dict(), save_path + 'D_epoch0.pth')

        # G-step
        for _ in range(args.g_freq):
            if args.fixz:
                try:
                    z = next(z_iter)[0].cuda()
                except:
                    z_iter = iter(zloader)
                    z = next(z_iter)[0].cuda()
            else:
                z = torch.randn(args.batch_size, args.input_dim, device=device)
            g_optimizer.zero_grad()
            x_hat = netG(z)
            y_hat = netD(x_hat)
            g_loss = get_gloss(args.losstype, y_hat)
            g_losses.append(g_loss.item())
            g_loss.backward()
            g_optimizer.step()
            grad_normG.append(utils.getGradNorm(netG))

            if args.ema_trick:
                moving_average.soft_copy_param(ema_netG_9999, netG, 0.9999)

        for _ in range(args.d_freq):
            try:
                x = next(data_iter)[0].cuda().float()
            except StopIteration:
                data_iter = iter(loader)
                x = next(data_iter)[0].cuda().float()
            if args.fixz:
                try:
                    z = next(z_iter)[0].cuda()
                except:
                    z_iter = iter(zloader)
                    z = next(z_iter)[0].cuda()
            else:
                z = torch.randn(args.batch_size, args.input_dim, device=device)

            d_optimizer.zero_grad()
            x_hat = netG(z).detach()
            y_hat = netD(x_hat)
            y = netD(x)
            d_loss = get_dloss(args.losstype, y_hat, y)
            d_losses.append(d_loss.item())

            d_loss.backward()
            d_optimizer.step()
            grad_normD.append(utils.getGradNorm(netD))

        if i % args.print_freq == 0:
            print('Iteration: {}; G-Loss: {}; D-Loss: {};'.format(i, g_loss, d_loss))

        if i == 1:
            save_image((x / 2. + 0.5)[:36], os.path.join(dir_name, 'real.png'))

        if i==1 or i % args.plot_freq == 0:
            plot_x = netG(torch.randn(36, args.input_dim, device=device)).data
            plot_x = plot_x / 2. + 0.5
            save_image(plot_x, os.path.join(dir_name, 'fake_images-{}.png'.format(i + 1)))
            utils.plot_losses(g_losses, d_losses, grad_normG, grad_normD, dir_name)
            utils.saveproj(y.cpu(), y_hat.cpu(), i, save_path)

        if i % args.save_freq == 0:
            torch.save(netG.state_dict(), save_path + 'G_epoch{}.pth'.format(i))
            torch.save(netD.state_dict(), save_path + 'D_epoch{}.pth'.format(i))
            if args.ema_trick:
                torch.save(ema_netG_9999.state_dict(), save_path + 'emaG0.9999_epoch{}.pth'.format(i))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar', choices=['mnist', 'cifar', 'stl', 'celeba', 'lsun'])
    parser.add_argument('--structure', type=str, default='dcgan', choices=['resnet', 'dcgan'])
    parser.add_argument('--losstype', type=str, default='log', choices=['log', 'hinge'])
    parser.add_argument('--norm', type=str, default='sn', choices=['sn', 'bn'])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--input_dim', type=int, default=128)
    parser.add_argument('--num_iters', type=int, default=100000)
    parser.add_argument('--fixz', action='store_true')
    parser.add_argument('--Gnum_features', type=int, default=64)
    parser.add_argument('--Dnum_features', type=int, default=64)

    parser.add_argument('--g_lr', type=float, default=2e-4)
    parser.add_argument('--d_lr', type=float, default=2e-4)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--g_freq', type=int, default=1)
    parser.add_argument('--d_freq', type=int, default=1)
    parser.add_argument('--lr_decay_start', type=int, default=50000)
    parser.add_argument('--perturb', type=float, default=0.1)

    parser.add_argument('--print_freq', type=int, default=20)
    parser.add_argument('--plot_freq', type=int, default=200)
    parser.add_argument('--save_freq', type=int, default=500)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--specname', type=str, default='')
    parser.add_argument('--ema_trick', action='store_true')

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda')

    train()

