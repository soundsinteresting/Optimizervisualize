import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import matplotlib.pyplot as plt
import copy
import modeloperations as mo
from network import Resnet, Twohiddenlayerfc, CNNMnist
import re
import os

def load_models(name, target_net=Twohiddenlayerfc()):
    net = copy.deepcopy(target_net)
    net.load_state_dict(torch.load(name))
    net.eval()
    return net

def loss_value(net, test_loader, device):
    with torch.no_grad():
        loss = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = net.to(device)(images)
                loss_i = nn.CrossEntropyLoss()(outputs, labels)
                loss += (loss_i.item())*len(images)
                total += len(images)
        return loss/total

def loss_surface_and_trajectory(name_list0, name_list1, sample_net, test_loader, grid_x=10, grid_y=10, device='cpu'):
    nop = len(name_list0)
    start_net = load_models(name_list0[0], sample_net)
    middle_net = load_models(name_list0[-1], sample_net)
    end_net = load_models(name_list1[-1], sample_net)

    sm = mo.scalar_mul([1,-1], [start_net, middle_net])
    se = mo.scalar_mul([1,-1], [start_net, end_net])
    me = mo.scalar_mul([1, -1], [middle_net, end_net])
    d_sm = torch.sqrt(mo.inner_p(sm, sm))
    d_se = torch.sqrt(mo.inner_p(se, se))
    d_me = torch.sqrt(mo.inner_p(me, me))

    #ld = torch.sqrt()
    # e1 and e2 are 2 unit vector in parameter space
    # ld is the half the thength of edge of the area to plot
    center = mo.scalar_mul([1/3,1/3,1/3],[start_net,middle_net,end_net])
    e_1 = mo.scalar_mul([1,-1],[center, start_net])
    ld = max(d_sm,d_se, d_me)
    e1norm = torch.sqrt(mo.inner_p(e_1,e_1))
    e_1 = mo.scalar_mul([1/e1norm],[e_1])

    e1dme = mo.inner_p(e_1, me)
    e_2 = mo.scalar_mul([1,-e1dme],[me, e_1])
    e_2 = mo.scalar_mul([torch.sqrt(1 / mo.inner_p(e_2, e_2))], [e_2])

    print('2D coordinate system established.')

    #calculate the loss surface, in the substace
    loss_sur = np.zeros((grid_x,grid_y))
    loss_grid_x = np.zeros(grid_x)
    loss_grid_y = np.zeros(grid_y)
    for i in range(grid_x):
        for j in range(grid_y):
            netij = mo.scalar_mul([1,2*(i-(grid_x-1)/2)/grid_x*ld, 2*(j-(grid_y-1)/2)/grid_y*ld],[center, e_1, e_2])
            loss_sur[i][j] = loss_value(netij, test_loader, device)
            loss_grid_x[i] = 2*(i-(grid_x-1)/2)/grid_x
            loss_grid_y[j] = 2*(j-(grid_y-1)/2)/grid_y
        print('[{}/{}] calculating loss surface'.format(i*grid_x+j+1,grid_x*grid_y))
    print('loss surface calculated')

    #calculate projection of the trajectory
    x = np.zeros((2,nop))
    y = np.zeros((2,nop))
    for i,name in enumerate(name_list0):
        neti = load_models(name, sample_net)
        dnet = mo.scalar_mul([1,-1],[neti, center])
        x[0][i] = mo.inner_p(dnet,e_1)/ld
        y[0][i] = mo.inner_p(dnet,e_2)/ld
    for i,name in enumerate(name_list1):
        neti = load_models(name, sample_net)
        dnet = mo.scalar_mul([1,-1],[neti, center])
        x[1][i] = mo.inner_p(dnet,e_1)/ld
        y[1][i] = mo.inner_p(dnet,e_2)/ld
    print('Path projections calculated.')
    return loss_sur, x, y, loss_grid_x, loss_grid_y

def load_and_create(source_folder_name, sample_net, test_loader, target_folder_name, device):
    all_files = os.listdir(source_folder_name) #all viable files
    #print(all_files)
    name_list_raw0 = []
    name_list0 = []
    name_list_raw1 = []
    name_list1 = []
    epoch_num0 = []
    epoch_num1 = []
    betatwo = []
    for name in all_files:
        pattern = re.compile(".*betatwo([0-9]+.[0-9]+)epoch([0-9]+).")
        s = pattern.search(name)
        if s:
            b2 = float(s.group(1))
            if len(betatwo) == 0:
                betatwo.append(b2)
            if len(betatwo) == 1 and betatwo[0] != b2:
                betatwo.append(b2)
            if betatwo[0] == b2:
                name_list_raw0.append(source_folder_name+name)
                epoch_num0.append(s.group(2))
            elif betatwo[1] == b2:
                name_list_raw1.append(source_folder_name + name)
                epoch_num1.append(s.group(2))
            #print(s.group(1))
    for i in range(len(epoch_num0)):
        name_list0.append(name_list_raw0[epoch_num0.index(str(i))])
        name_list1.append(name_list_raw1[epoch_num1.index(str(i))])
        #print(name_list_raw[epoch_num.index(str(i))])


    surface, x, y, gx, gy = loss_surface_and_trajectory(name_list0, name_list1, sample_net, test_loader, 20, 20, device)
    np.savetxt(target_folder_name+'surface.txt', surface)
    np.savetxt(target_folder_name + 'x.txt', x)
    np.savetxt(target_folder_name + 'y.txt', y)
    np.savetxt(target_folder_name + 'x_grid.txt', gx)
    np.savetxt(target_folder_name + 'y_grid.txt', gy)
    np.savetxt(target_folder_name + 'betatwo.txt', betatwo)
    '''
    doc = open(target_folder_name+'surface.txt', 'w')
    print(surface,file=doc)
    doc.close()
    doc = open(target_folder_name + 'x.txt', 'w')
    print(x, file=doc)
    doc.close()
    doc = open(target_folder_name + 'y.txt', 'w')
    print(y, file=doc)
    doc.close()
    '''
if __name__ == "__main__":
    #dirname = r"D:\shi\projects\adam\raw_data\2020-09-24-16\\"
    dirname = r"./raw_data/2020-10-02-08/"
    args = {'num_channels': 1, 'num_classes': 10}
    net = CNNMnist(args)
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])
    batchsize = 30000
    trainset = torchvision.datasets.MNIST(root='./mnist/data', train=True,
                                          download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize,
                                              shuffle=True, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB')
    load_and_create(dirname,net,trainloader,dirname,device)
