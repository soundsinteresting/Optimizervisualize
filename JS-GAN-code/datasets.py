import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms, datasets
import numpy as np


def sample_data(p, n=100):
    return np.random.multivariate_normal(p, np.array([[0.002, 0], [0, 0.002]]), n)


def getDataNp(n, gaussian_num):
    if gaussian_num == 5:
        d = np.linspace(0, 360, 6)[:-1]
        x = np.sin(d / 180. * np.pi)
        y = np.cos(d / 180. * np.pi)
        points = np.vstack((y, x)).T
        s0 = sample_data(points[0], n).astype(np.float)
        s1 = sample_data(points[1], n).astype(np.float)
        s2 = sample_data(points[2], n).astype(np.float)
        s3 = sample_data(points[3], n).astype(np.float)
        s4 = sample_data(points[4], n).astype(np.float)
        samples = np.vstack((s0, s1, s2, s3, s4))
    elif gaussian_num == 2:
        s0 = sample_data([1, 0], n).astype(np.float)
        s1 = sample_data([-1, 0], n).astype(np.float)
        samples = np.vstack((s0, s1))
    elif gaussian_num == 25:
        samples = np.empty((0, 2))
        for x in range(-2, 3, 1):
            for y in range(-2, 3, 1):
                samples = np.vstack((samples, sample_data([x, y], n).astype(np.float)))

    permutation = range(gaussian_num * n)
    np.random.shuffle(permutation)
    samples = samples[permutation]
    return samples


def toyDataLoder(n=100, batch_size=100, gaussian_num=5, shuffle=True, seed=0):
    np.random.seed(seed)
    samples = getDataNp(n, gaussian_num)
    data = TensorDataset(torch.from_numpy(samples))
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=True)
    return dataloader


def _noise_adder(img):
    return torch.empty_like(img, dtype=img.dtype).uniform_(0.0, 1/128.0) + img


def mnist_DataLoder(image_size, batch_size=128, shuffle=True, train=True):
    root = '~/DATA/mnist/'
    transform = transforms.Compose([
        transforms.Scale(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        _noise_adder,
    ])
    loader = torch.utils.data.DataLoader(
        datasets.MNIST(root=root, train=train, download=True, transform=transform),
        batch_size=batch_size, shuffle=shuffle, drop_last=True)
    return loader


def cifar_DataLoder(image_size, batch_size=128, shuffle=True, train=True):
    root = '~/DATA/cifar10/'
    transform = transforms.Compose([
        transforms.Scale(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        _noise_adder,
    ])
    loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=root, train=train, download=True, transform=transform),
        batch_size=batch_size, shuffle=shuffle, drop_last=True)
    return loader


def stl_DataLoder(image_size, batch_size=128, shuffle=True, train=True):
    root = '~/DATA/stl10/'
    transform = transforms.Compose([
        transforms.Scale(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        _noise_adder,
    ])
    if train:
        loader = torch.utils.data.DataLoader(
            datasets.STL10(root=root, split='unlabeled', download=True, transform=transform),
            batch_size=batch_size, shuffle=shuffle, drop_last=True)
    else:
        d1 = datasets.STL10(root=root, split='test', download=True, transform=transform)
        d2 = datasets.STL10(root=root, split='train', download=True, transform=transform)
        d = torch.utils.data.ConcatDataset([d1,d2])
        loader = torch.utils.data.DataLoader(d,batch_size=100, shuffle=shuffle, drop_last=True)
    return loader


def lsunDataLoder(image_size, batch_size=128, shuffle=True, train=True):
    dataset = datasets.LSUN(root='~/DATA', classes=['bedroom_train'],
                            transform=transforms.Compose([
                                transforms.Resize((image_size, image_size)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                _noise_adder,
                            ]))
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    return train_loader


def celebaDataLoader(image_size, batch_size=128, shuffle=True, train=True):
    np.random.seed(1)
    train_indices = torch.from_numpy(np.random.choice(200000, size=(4000,), replace=False))
    dataset = datasets.ImageFolder(root='~/DATA',
                   transform=transforms.Compose([
                       transforms.Resize((image_size, image_size)),
                       transforms.CenterCrop(image_size),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                       _noise_adder,
                   ]))
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_indices), drop_last=True)
    return train_loader


def getDataLoader(name, image_size, batch_size=100, shuffle=True, train=True):
    if name == 'mnist':
        return mnist_DataLoder(image_size, batch_size, shuffle, train)
    elif name == 'cifar':
        return cifar_DataLoder(image_size, batch_size, shuffle, train)
    elif name == 'stl':
        return stl_DataLoder(image_size, batch_size, shuffle, train)
    elif name == 'celeba':
        return celebaDataLoader(image_size, batch_size, shuffle, train)
    elif name == 'lsun':
        return lsunDataLoder(image_size, batch_size, shuffle, train)
