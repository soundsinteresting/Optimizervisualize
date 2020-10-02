import pickle
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from sklearn.decomposition import PCA
import copy

def mul(net1, net2):
    result = pickle.loads(pickle.dumps(net1))
    net1_dict = net1.state_dict()
    net2_dict = net2.state_dict()
    weight_keys = list(net1_dict.keys())
    result_dict = result.state_dict()
    for key in weight_keys:
        result_dict[key] = net1_dict[key] * net2_dict[key]
    result.load_state_dict(result_dict)
    return result


def scalar_mul(scalar_list, net_list, rg=False):
    result = pickle.loads(pickle.dumps(net_list[0]))
    worker_params = [list(x.parameters()) for x in net_list]
    for i, params in enumerate(result.parameters()):
        params.data = 0 * params.data
        for j in range(len(scalar_list)):
            params.data = params.data + worker_params[j][i] * scalar_list[j]
    return result



def inner_p(net1, net2, rg=False):
    result = torch.zeros(1)[0]
    if rg:
        result.requires_grad = True
    if next(net1.parameters()).is_cuda:
        #print('something is on cuda')
        result = result.cuda()
    net1_params = list(net1.parameters())
    net2_params = list(net2.parameters())

    for i in range(len(net1_params)):
        result = result + (net1_params[i]*net2_params[i]).sum()
    #print("inner product: {}".format(result))
    #print(result)
    return result


def net_flatten(net):
    newnet = pickle.loads(pickle.dumps(net))
    n_dict = newnet.state_dict()
    w_keys = list(n_dict.keys())
    res = torch.flatten(n_dict[w_keys[0]])
    for i in range(1,len(w_keys)):
        res = torch.cat((res,torch.flatten(n_dict[w_keys[0]])))
    return res


def add(net1, net2):
    result = pickle.loads(pickle.dumps(net1))
    net1_dict = net1.state_dict()
    net2_dict = net2.state_dict()
    weight_keys = list(net1_dict.keys())
    result_dict = result.state_dict()
    for key in weight_keys:
        result_dict[key] = net1_dict[key] + net2_dict[key]
    result.load_state_dict(result_dict)
    return result

def model_std(model_list):
    new_list = pickle.loads(pickle.dumps(model_list))
    wt = np.zeros(len(new_list))
    center_of_mass = scalar_mul(wt,new_list)
    var = 0
    for i in new_list:
        diff = scalar_mul([1,-1],[i,center_of_mass])
        var = var + inner_p(diff,diff)
    return torch.sqrt(var/(len(model_list)-1))

def model_star(model_list):
    res = []
    model_belong_dict = dict()
    lth = len(model_list)
    max_index = 0
    for i in range(lth):
        for j in range(0,i):
            diff = scalar_mul([1,-1],[model_list[i],model_list[j]])
            dis = torch.sqrt(inner_p(diff, diff))
            if dis<1e-1:
                if j in model_belong_dict.keys():
                    model_belong_dict[i] = copy.deepcopy(model_belong_dict[j])
                else:
                    model_belong_dict[i] = max_index
                    max_index += 1
            res.append(dis)
    return res, model_belong_dict

def weight_params_pca(model_list, rank, decentralize=True):
    A = net_flatten(model_list[0]).unsqueeze(0)
    for i in range(1,len(model_list)):
        A = torch.cat((A,net_flatten(model_list[i]).unsqueeze(0)),0)
    A=A.numpy()
    if decentralize:
        mean = A.mean(0)
        A = A - mean
    transformer = PCA(n_components=rank, random_state=0)
    transformer.fit(A)

    #U, S, V = torch.pca_lowrank(A,q=rank)
    return transformer.explained_variance_ratio_

def grad_over_model(numerator, denominator):
    result = pickle.loads(pickle.dumps(denominator))
    #denominator.zero_grad()
    numerator.backward()
    res_params = list(result.parameters())
    deno_params = list(denominator.parameters())
    for i in range(len(res_params)):
        res_params[i] = deno_params[i].grad
    result.zero_grad()
    denominator.zero_grad()
    return result


def rbk(net1, net2, r, rg=False):
    diff = scalar_mul([1,-1],[net1,net2])
    l2norm = inner_p(diff, diff, rg)
    return (-r*l2norm).exp()

def nabla_1rbk(net1, net2, r):
    exp = rbk(net1,net2,r)
    return scalar_mul([-2*exp*r, 2*exp*r], [net1, net2])

if __name__ == '__main__':
    from network import CNNMnist, Twohiddenlayerfc
    ml = []
    for i in range(10):
        ml.append(Twohiddenlayerfc())
    #ms = model_star(ml)
    import copy
    m11=copy.deepcopy(ml[0])
    res = m11(torch.randn(10))
    res = res.sum()
    res.backward()
    ms = weight_params_pca(ml,7)
    print('eigen values: {}'.format(ms))
'''
    n1 = Twohiddenlayerfc()
    n2 = Twohiddenlayerfc()
    print(inner_p(n1,n1))


    n3 = mul(n1,n2)
    input = torch.ones(10)
    print(n3(input))
    print('n3')
    #print(n3.state_dict())
    print(inner_p(n1, n1))
    print(inner_p(n2, n2))
    print(inner_p(n3,n3))
    #print(list(n3.parameters())[1])
    n4 = scalar_mul([10],[n3])
    #print(list(n4.parameters())[1])
    #
    #for p in n3.parameters():
    #    print(p)
    #    p.data = p.data*10
    res = inner_p(n4, n4)
    res.backward()
    #print(inner_p(n4, n4))
    p = n3.parameters()

'''
