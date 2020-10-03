import argparse
# import inception_score_tf
import fid
# import itertools
# import prd_score as prd
# import pickle
import networks
import datasets
import numpy as np
import torch

def getRealData(size):
    loader = datasets.getDataLoader(args.dataset, args.image_size, batch_size=size, train=False)
    data_iter = iter(loader)
    realdata = data_iter.next()[0]
    realdata = np.array(realdata)
    if args.dataset == 'mnist':
        realdata = realdata.repeat(3, axis=1)
    realdata = (realdata / 2 + 0.5) * 255
    realdata = realdata.astype(np.uint8)
    return realdata

def getFakedata(size, netG):
    data = []
    for _ in range(0, size, 100):
        z = (torch.rand(100, args.input_dim) * 2 - 1).cuda()
        with torch.no_grad():
            x = netG(z)
            if args.dataset == 'mnist':
                x = x.repeat(3, axis=1)
        data.append(x.clone().detach().cpu().numpy())
    data = np.concatenate(data)
    data = (data / 2 + 0.5) * 255
    data = data.astype(np.uint8)
    return data


def getFID(netG):
    realdata = getRealData(10000)
    fakedata = getFakedata(10000, netG)
    FID_val = fid.get_fid(realdata, fakedata)
    return FID_val


# def getIS(netG):
#     data = getFakedata(50000, netG)
#     mean, std = inception_score_tf.get_inception_score(data, splits=10)
#     return mean, std


# def getPRD(netG):
#     realdata = getRealData(10000)
#     fakedata = getFakedata(10000,netG)
#     ref_emb = fid_tf.get_inception_activations(realdata)
#     eval_emb = fid_tf.get_inception_activations(fakedata)
#     prd_res = prd.compute_prd_from_embedding(eval_data=eval_emb, ref_data=ref_emb)
#     return prd_res


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', type=str, default='FID', choices=['IS', 'FID', 'PRD'])
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar', 'stl', 'celeba', 'lsun'])
    parser.add_argument('--structure', type=str, default='dcgan', choices=['resnet', 'dcgan'])
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--input_dim', type=int, default=128)
    parser.add_argument('--num_features', type=int, default=64)
    parser.add_argument('--norm', type=str, default='sn', choices=['bn','sn'])
    parser.add_argument('--use_rezero', dest='rezero', action='store_true')
    parser.set_defaults(rezero=False)

    args = parser.parse_args()
    
    if args.norm == 'sn':
        netG, _ = networks.getGD_SN(args.structure, args.dataset, args.num_features, args.num_features, ignoreD=True)
    elif args.norm == 'bn':
        netG, _ = networks.getGD_batchnorm(args.structure, args.dataset, args.num_features, args.num_features, ignoreD=True)
    netG.load_state_dict(torch.load(args.model_path))
    netG.cuda()

    # if metric == 'IS':
    #     mean, std = getIS(netG)
    #     print('[*] Metric: IS, Mean: %s, Std: %s' % (mean, std))
    if args.metric == 'FID':
        FID_val = getFID(netG)
        print('[*] Metric: FID: %s ' % (FID_val))
    # elif metric == 'PRD':
    #     prd_res = getPRD(netG)
    #     print('[*] Metric: PRD: %s ' % (prd_res))