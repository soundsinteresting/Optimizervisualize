from torch.nn import init
import torch
import torch.nn as nn
from torch.nn import utils
import torch.nn.functional as F

from gen_resblocks import Block as GenBlock
from disc_resblocks import Block as DiscBlock
from disc_resblocks import OptimizedBlock


# =======  define SNGAN structure for MNIST and CIFAR ===========
class ResNetGenerator32(nn.Module):
    """Generator generates 32x32."""

    def __init__(self, num_features=256, dim_z=128, channel=3, bottom_width=4,
                 activation=F.relu, num_classes=0, distribution='normal'):
        super(ResNetGenerator32, self).__init__()
        self.num_features = num_features
        self.dim_z = dim_z
        self.bottom_width = bottom_width
        self.activation = activation
        self.num_classes = num_classes
        self.distribution = distribution

        self.l1 = nn.Linear(dim_z, num_features * bottom_width ** 2) 

        self.block2 = GenBlock(num_features, num_features,
                            activation=activation, upsample=True,
                            num_classes=num_classes) 
        self.block3 = GenBlock(num_features, num_features,
                            activation=activation, upsample=True,
                            num_classes=num_classes)  
        self.block4 = GenBlock(num_features, num_features,
                            activation=activation, upsample=True,
                            num_classes=num_classes) 
        self.b5 = nn.BatchNorm2d(num_features)
        self.conv5 = nn.Conv2d(num_features, channel, 1, 1)  

    def _initialize(self):
        init.xavier_uniform_(self.l1.weight.data)
        init.xavier_uniform_(self.conv5.weight.data)

    def forward(self, z, y=None, **kwargs):
        h = self.l1(z).view(z.size(0), -1, self.bottom_width, self.bottom_width)
        for i in range(2, 5):
            h = getattr(self, 'block{}'.format(i))(h, y, **kwargs)
        h = self.activation(self.b5(h))
        return torch.tanh(self.conv5(h))


class SNResNetProjectionDiscriminator32(nn.Module):
    def __init__(self, num_features=256, channel=3, num_classes=0, activation=F.relu):
        super(SNResNetProjectionDiscriminator32, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.activation = activation

        self.block1 = OptimizedBlock(channel, num_features)
        self.block2 = DiscBlock(num_features, num_features,
                            activation=activation, downsample=True)
        self.block3 = DiscBlock(num_features, num_features,
                            activation=activation, downsample=True)
        self.block4 = DiscBlock(num_features, num_features,
                            activation=activation, downsample=True)
        self.proj = utils.spectral_norm(nn.Linear(num_features, 1, bias=False))
        self._initialize()

    def _initialize(self):
        init.orthogonal_(self.proj.weight.data)
        optional_l_y = getattr(self, 'l_y', None)
        if optional_l_y is not None:
            init.orthogonal_(optional_l_y.weight.data)

    def forward(self, x, y=None):
        h = x  
        h = self.block1(h)  
        h = self.block2(h) 
        h = self.block3(h) 
        h = self.block4(h) 
        h = self.activation(h)
        # Global pooling
        h = torch.sum(h, dim=(2, 3))  
        # print h.shape
        output = self.proj(h) 
        if y is not None:
            output += torch.sum(self.l_y(y) * h, dim=1, keepdim=True)
        return output



# ========  define SNGAN structure for STL ========
class ResNetGenerator48(nn.Module):
    """Generator generates 48x48."""
    def __init__(self, num_features=64, dim_z=128, bottom_width=6,
                 activation=F.relu, num_classes=0, distribution='normal'):
        super(ResNetGenerator48, self).__init__()
        self.num_features = num_features
        self.dim_z = dim_z
        self.bottom_width = bottom_width
        self.activation = activation
        self.num_classes = num_classes
        self.distribution = distribution

        self.l1 = nn.Linear(dim_z, 8 * num_features * bottom_width ** 2) 
        self.block2 = GenBlock(num_features * 8, num_features * 4,
                            activation=activation, upsample=True,
                            num_classes=num_classes)  
        self.block3 = GenBlock(num_features * 4, num_features * 2,
                            activation=activation, upsample=True,
                            num_classes=num_classes)  
        self.block4 = GenBlock(num_features * 2, num_features,
                            activation=activation, upsample=True,
                            num_classes=num_classes)  
        self.b5 = nn.BatchNorm2d(num_features)
        self.conv5 = nn.Conv2d(num_features, 3, 1, 1)  

    def _initialize(self):
        init.xavier_uniform_(self.l1.weight.data)
        init.xavier_uniform_(self.conv5.weight.data)

    def forward(self, z, y=None, **kwargs):
        h = self.l1(z).view(z.size(0), -1, self.bottom_width, self.bottom_width)
        for i in range(2, 5):
            h = getattr(self, 'block{}'.format(i))(h, y, **kwargs)
        h = self.activation(self.b5(h))
        return torch.tanh(self.conv5(h))


class SNResNetProjectionDiscriminator48(nn.Module):
    def __init__(self, num_features=64, num_classes=0, activation=F.relu):
        super(SNResNetProjectionDiscriminator48, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.activation = activation

        self.block1 = OptimizedBlock(3, num_features)
        self.block2 = DiscBlock(num_features, num_features * 2,
                            activation=activation, downsample=True)
        self.block3 = DiscBlock(num_features * 2, num_features * 4,
                            activation=activation, downsample=True)
        self.block4 = DiscBlock(num_features * 4, num_features * 8,
                            activation=activation, downsample=True)
        self.block5 = DiscBlock(num_features * 8, num_features * 16,
                                activation=activation, downsample=True)
        self.proj = utils.spectral_norm(nn.Linear(num_features*16, 1, bias=False))
        self._initialize()

    def _initialize(self):
        init.orthogonal_(self.proj.weight.data)
        optional_l_y = getattr(self, 'l_y', None)
        if optional_l_y is not None:
            init.orthogonal_(optional_l_y.weight.data)

    def forward(self, x, y=None):
        h = x  
        h = self.block1(h)  
        h = self.block2(h)  
        h = self.block3(h) 
        h = self.block4(h) 
        h = self.block5(h) 
        h = self.activation(h)
        # Global pooling
        h = torch.sum(h, dim=(2, 3))
        output = self.proj(h)
        return output


# ========  define SNGAN structure for 64 CelebA ========
class ResNetGenerator64(nn.Module):
    """Generator generates 64x64."""

    def __init__(self, num_features=64, dim_z=128, bottom_width=4,
                 activation=F.relu, num_classes=0, distribution='normal'):
        super(ResNetGenerator64, self).__init__()
        self.num_features = num_features
        self.dim_z = dim_z
        self.bottom_width = bottom_width
        self.activation = activation
        self.num_classes = num_classes
        self.distribution = distribution

        self.l1 = nn.Linear(dim_z, 16 * num_features * bottom_width ** 2)
        self.block2 = GenBlock(num_features * 16, num_features * 8,
                            activation=activation, upsample=True,
                            num_classes=num_classes)
        self.block3 = GenBlock(num_features * 8, num_features * 4,
                            activation=activation, upsample=True,
                            num_classes=num_classes)
        self.block4 = GenBlock(num_features * 4, num_features * 2,
                            activation=activation, upsample=True,
                            num_classes=num_classes)
        self.block5 = GenBlock(num_features * 2, num_features,
                            activation=activation, upsample=True,
                            num_classes=num_classes)
        self.b6 = nn.BatchNorm2d(num_features)
        self.conv6 = nn.Conv2d(num_features, 3, 1, 1)
        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.l1.weight.data)
        init.xavier_uniform_(self.conv6.weight.data)

    def forward(self, z, y=None, **kwargs):
        h = self.l1(z).view(z.size(0), -1, self.bottom_width, self.bottom_width)
        for i in range(2, 6):
            h = getattr(self, 'block{}'.format(i))(h, y, **kwargs)
        h = self.activation(self.b6(h))
        return torch.tanh(self.conv6(h))


class SNResNetProjectionDiscriminator64(nn.Module):

    def __init__(self, num_features=64, num_classes=0, activation=F.relu):
        super(SNResNetProjectionDiscriminator64, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.activation = activation

        self.block1 = OptimizedBlock(3, num_features)
        self.block2 = DiscBlock(num_features, num_features * 2,
                            activation=activation, downsample=True)
        self.block3 = DiscBlock(num_features * 2, num_features * 4,
                            activation=activation, downsample=True)
        self.block4 = DiscBlock(num_features * 4, num_features * 8,
                            activation=activation, downsample=True)
        self.block5 = DiscBlock(num_features * 8, num_features * 16,
                            activation=activation, downsample=True)
        self.proj = utils.spectral_norm(nn.Linear(num_features*16, 1, bias=False))
        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.proj.weight.data)
        optional_l_y = getattr(self, 'l_y', None)
        if optional_l_y is not None:
            init.xavier_uniform_(optional_l_y.weight.data)

    def forward(self, x, y=None):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.activation(h)
        # Global pooling
        h = torch.sum(h, dim=(2, 3))
        # print h.shape
        output = self.proj(h)
        return output



# ====== SN for cifar DCGAN =======
class SNDCGenerator32(nn.Module):
    # initializers
    def __init__(self, dim_z=128, num_features=64, channel=3, first_kernel=4):
        super(SNDCGenerator32, self).__init__()
        self.dim_z = dim_z
        self.num_features = num_features
        self.first_kernel = first_kernel
        self.l1 = nn.Linear(dim_z, num_features*8*first_kernel*first_kernel)
        self.deconv1 = nn.ConvTranspose2d(num_features*8, num_features*4, 4, 2, 1)
        self.deconv1_bn = nn.BatchNorm2d(num_features*4)
        self.deconv2 = nn.ConvTranspose2d(num_features*4, num_features*2, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(num_features*2)
        self.deconv3 = nn.ConvTranspose2d(num_features*2, num_features, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(num_features)
        self.conv4 = nn.Conv2d(num_features, channel, 3, 1, 1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.l1.weight.data)
        init.xavier_uniform_(self.deconv1.weight.data)
        init.xavier_uniform_(self.deconv2.weight.data)
        init.xavier_uniform_(self.deconv3.weight.data)
        init.xavier_uniform_(self.conv4.weight.data)

    # forward method
    def forward(self, input):
        x = self.l1(input)
        x = x.view(-1, self.num_features*8, self.first_kernel, self.first_kernel)
        x = F.relu(self.deconv1_bn(self.deconv1(x)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.tanh(self.conv4(x))
        return x


class SNDCDiscriminator32(nn.Module):
    # initializers
    def __init__(self, num_features=64, channel=3, first_kernel=4):
        super(SNDCDiscriminator32, self).__init__()
        self.num_features = num_features
        self.first_kernel = first_kernel
        self.conv1 = utils.spectral_norm(nn.Conv2d(channel, num_features, 3, 1, 1))
        self.conv2 = utils.spectral_norm(nn.Conv2d(num_features, num_features, 4, 2, 1))
        self.conv3 = utils.spectral_norm(nn.Conv2d(num_features, num_features * 2, 3, 1, 1))
        self.conv4 = utils.spectral_norm(nn.Conv2d(num_features * 2, num_features * 2, 4, 2, 1))
        self.conv5 = utils.spectral_norm(nn.Conv2d(num_features * 2, num_features * 4, 3, 1, 1))
        self.conv6 = utils.spectral_norm(nn.Conv2d(num_features * 4, num_features * 4, 4, 2, 1))
        self.conv7 = utils.spectral_norm(nn.Conv2d(num_features * 4, num_features * 8, 3, 1, 1))
        self.proj = utils.spectral_norm(nn.Linear(num_features * 8 * first_kernel * first_kernel, 1, bias=False))
        self.ortho_initialize()

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.1)  # (50, 128, 32, 32)
        x = F.leaky_relu(self.conv2(x), 0.1)  # (50, 256, 16, 16)
        x = F.leaky_relu(self.conv3(x), 0.1)  # (50, 512, 8, 8)
        x = F.leaky_relu(self.conv4(x), 0.1)  # (50, 512, 8, 8)
        x = F.leaky_relu(self.conv5(x), 0.1)
        x = F.leaky_relu(self.conv6(x), 0.1)
        x = F.leaky_relu(self.conv7(x), 0.1)
        x = x.view(x.size(0), -1)
        y = self.proj(x)
        return y

    def ortho_initialize(self):
        init.orthogonal_(self.conv1.weight.data)
        init.orthogonal_(self.conv2.weight.data)
        init.orthogonal_(self.conv3.weight.data)
        init.orthogonal_(self.conv4.weight.data)
        init.orthogonal_(self.conv5.weight.data)
        init.orthogonal_(self.conv6.weight.data)
        init.orthogonal_(self.conv7.weight.data)
        init.orthogonal_(self.proj.weight.data)


# ===========  DCGAN for 64 CelebA ================
class DCGenerator64(nn.Module):
    # initializers
    def __init__(self, dim_z=128, num_features=64, channel=3):
        super(DCGenerator64, self).__init__()
        self.dim_z = dim_z
        self.deconv1 = nn.ConvTranspose2d(dim_z, num_features*8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(num_features*8)
        self.deconv2 = nn.ConvTranspose2d(num_features*8, num_features*4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(num_features*4)
        self.deconv3 = nn.ConvTranspose2d(num_features*4, num_features*2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(num_features*2)
        self.deconv4 = nn.ConvTranspose2d(num_features*2, num_features, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(num_features)
        self.deconv5 = nn.ConvTranspose2d(num_features, channel, 4, 2, 1)

    # forward method
    def forward(self, input):
        x = input.view(-1, self.dim_z, 1, 1)
        x = F.relu(self.deconv1_bn(self.deconv1(x)))  # (50, 512, 4, 4)
        x = F.relu(self.deconv2_bn(self.deconv2(x)))  # (50, 256, 8, 8)
        x = F.relu(self.deconv3_bn(self.deconv3(x)))  # (50, 128, 16, 16)
        x = F.relu(self.deconv4_bn(self.deconv4(x)))  # (50, 64, 32, 32)
        x = F.tanh(self.deconv5(x))  # (50, 3, 64, 64)
        return x


class DCDiscriminator64(nn.Module):
    # initializers
    def __init__(self, num_features=64, channel=3):
        super(DCDiscriminator64, self).__init__()
        self.num_features = num_features
        self.conv1 = utils.spectral_norm(nn.Conv2d(channel, num_features, 4, 2, 1))
        self.conv2 = utils.spectral_norm(nn.Conv2d(num_features, num_features*2, 4, 2, 1))
        self.conv3 = utils.spectral_norm(nn.Conv2d(num_features*2, num_features*4, 4, 2, 1))
        self.conv4 = utils.spectral_norm(nn.Conv2d(num_features*4, num_features*8, 4, 2, 1))
        self.proj = utils.spectral_norm(nn.Linear(num_features*8*4*4, 1, bias=False))

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.1)  
        x = F.leaky_relu(self.conv2(x), 0.1)  
        x = F.leaky_relu(self.conv3(x), 0.1) 
        x = F.leaky_relu(self.conv4(x), 0.1) 
        x = x.view(x.size(0), -1)
        y = self.proj(x)
        return y


def getGD_SN(structure, dataset, Gnum_features, Dnum_features, ignoreD=False):
    leaky_relu = lambda x: F.leaky_relu(x, negative_slope=0.1)
    if structure == 'resnet':
        if dataset == 'mnist':
            netG = ResNetGenerator32(num_features=Gnum_features, channel=1)
            if not ignoreD:
                netD = SNResNetProjectionDiscriminator32(num_features=Dnum_features,
                                                                 channel=1, activation=leaky_relu)
        elif dataset == 'cifar':
            netG = ResNetGenerator32(num_features=Gnum_features)
            if not ignoreD:
                netD = SNResNetProjectionDiscriminator32(num_features=Dnum_features//2, activation=leaky_relu)
        elif dataset == 'stl':
            netG = ResNetGenerator48(num_features=Gnum_features)
            if not ignoreD:
                netD = SNResNetProjectionDiscriminator48(num_features=Dnum_features, activation=leaky_relu)
        elif dataset == 'celeba':
            netG = ResNetGenerator64(num_features=Gnum_features)
            if not ignoreD:
                netD = SNResNetProjectionDiscriminator64(num_features=Dnum_features, activation=leaky_relu)

    if structure == 'dcgan':
        if dataset == 'mnist':
            netG = SNDCGenerator32(num_features=Gnum_features, channel=1)
            if not ignoreD:
                netD = SNDCDiscriminator32(num_features=Dnum_features, channel=1)
        elif dataset == 'cifar':
            netG = SNDCGenerator32(num_features=Gnum_features)
            if not ignoreD:
                netD = SNDCDiscriminator32(num_features=Dnum_features)
        elif dataset == 'stl':
            netG = SNDCGenerator32(num_features=Gnum_features, first_kernel=6)
            if not ignoreD:
                netD = SNDCDiscriminator32(num_features=Dnum_features, first_kernel=6)
        elif dataset == 'celeba':
            netG = DCGenerator64(num_features=Gnum_features)
            if not ignoreD:
                netD = DCDiscriminator64(num_features=Dnum_features)
        
    if ignoreD:
        netD = None
    return netG, netD




# ============================================================================================================
# batch norm or no normalization
# ============================================================================================================


# ===========  DCGAN for 32 MNIST ================

class DCGenerator32mnist(nn.Module):
    def __init__(self, dim_z=128, num_features=64, channel=3, first_kernel=4):
        super(DCGenerator32mnist, self).__init__()
        self.dim_z = dim_z
        self.num_features = num_features
        self.l1 = nn.Linear(dim_z, num_features*8)
        self.deconv1 = nn.ConvTranspose2d(num_features*8, num_features*4, first_kernel, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(num_features*4)
        self.deconv2 = nn.ConvTranspose2d(num_features*4, num_features*2, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(num_features*2)
        self.deconv3 = nn.ConvTranspose2d(num_features*2, num_features, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(num_features)
        self.deconv4 = nn.ConvTranspose2d(num_features, channel, 4, 2, 1)

    def forward(self, input):
        x = self.l1(input)
        x = x.view(-1, self.num_features*8, 1, 1)
        x = F.relu(self.deconv1_bn(self.deconv1(x)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.tanh(self.deconv4(x))
        return x


class DCDiscriminator32mnist(nn.Module):
    # initializers
    def __init__(self, num_features=64, channel=3, first_kernel=4):
        super(DCDiscriminator32mnist, self).__init__()
        self.num_features = num_features
        # self.first_kernel = first_kernel
        self.conv1 = nn.Conv2d(channel, num_features, 4, 2, 1)
        self.conv2 = nn.Conv2d(num_features, num_features*2, 4, 2, 1)
        self.conv3 = nn.Conv2d(num_features*2, num_features*4, 4, 2, 1)
        self.proj = nn.Linear(num_features*4*first_kernel*first_kernel, 1, bias=False)

    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = x.view(x.size(0), -1)
        y = self.proj(x)
        return y


# ===========  DCGAN for 32 CIFAR and 48 STL ================

# this is the disc that sngan paper use
class DCDiscriminator32(nn.Module):
    # initializers
    def __init__(self, num_features=64, channel=3, first_kernel=4):
        super(DCDiscriminator32, self).__init__()
        self.num_features = num_features
        self.first_kernel = first_kernel
        self.conv1 = nn.Conv2d(channel, num_features, 3, 1, 1)
        # self.convbn1 = nn.BatchNorm2d(num_features)
        self.conv2 = nn.Conv2d(num_features, num_features, 4, 2, 1)
        # self.convbn2 = nn.BatchNorm2d(num_features)
        self.conv3 = nn.Conv2d(num_features, num_features * 2, 3, 1, 1)
        # self.convbn3 = nn.BatchNorm2d(num_features*2)
        self.conv4 = nn.Conv2d(num_features * 2, num_features * 2, 4, 2, 1)
        # self.convbn4 = nn.BatchNorm2d(num_features*2)
        self.conv5 = nn.Conv2d(num_features * 2, num_features * 4, 3, 1, 1)
        # self.convbn5 = nn.BatchNorm2d(num_features*4)
        self.conv6 = nn.Conv2d(num_features * 4, num_features * 4, 4, 2, 1)
        # self.convbn6 = nn.BatchNorm2d(num_features*4)
        self.conv7 = nn.Conv2d(num_features * 4, num_features * 8, 3, 1, 1)
        # self.convbn7 = nn.BatchNorm2d(num_features*8)
        self.proj = nn.Linear(num_features * 8 * first_kernel * first_kernel, 1, bias=False)
        self.initilize()

    # forward method
    def forward(self, input):
        # x = F.leaky_relu(self.convbn1(self.conv1(input)), 0.1) 
        # x = F.leaky_relu(self.convbn2(self.conv2(x)), 0.1) 
        # x = F.leaky_relu(self.convbn3(self.conv3(x)), 0.1) 
        # x = F.leaky_relu(self.convbn4(self.conv4(x)), 0.1) 
        # x = F.leaky_relu(self.convbn5(self.conv5(x)), 0.1)
        # x = F.leaky_relu(self.convbn6(self.conv6(x)), 0.1)
        # x = F.leaky_relu(self.convbn7(self.conv7(x)), 0.1)
        # x = x.view(x.size(0), -1)
        # y = self.proj(x)
        # return y
        x = F.leaky_relu(self.conv1(input), 0.1) 
        x = F.leaky_relu(self.conv2(x), 0.1) 
        x = F.leaky_relu(self.conv3(x), 0.1) 
        x = F.leaky_relu(self.conv4(x), 0.1) 
        x = F.leaky_relu(self.conv5(x), 0.1)
        x = F.leaky_relu(self.conv6(x), 0.1)
        x = F.leaky_relu(self.conv7(x), 0.1)
        x = x.view(x.size(0), -1)
        y = self.proj(x)
        return y


    def initilize(self):
        init.orthogonal_(self.conv1.weight.data)
        init.orthogonal_(self.conv2.weight.data)
        init.orthogonal_(self.conv3.weight.data)
        init.orthogonal_(self.conv4.weight.data)
        init.orthogonal_(self.conv5.weight.data)
        init.orthogonal_(self.conv6.weight.data)
        init.orthogonal_(self.conv7.weight.data)
        init.orthogonal_(self.proj.weight.data)



def getGD_batchnorm(structure, dataset, num_Gfeatures, num_Dfeatures, ignoreD=False, dim_z=128):
    leaky_relu = lambda x: F.leaky_relu(x, negative_slope=0.1)
    
    if structure == 'dcgan':
        if dataset == 'mnist':
            netG = DCGenerator32mnist(num_features=num_Gfeatures, channel=1, dim_z=dim_z)
            if not ignoreD:
                netD = DCDiscriminator32mnist(num_features=num_Dfeatures, channel=1)
        elif dataset == 'cifar':
            netG = SNDCGenerator32(num_features=num_Gfeatures, channel=3)
            if not ignoreD:
                netD = DCDiscriminator32(num_features=num_Dfeatures, channel=3)
        elif dataset == 'stl':
            netG = SNDCGenerator32(num_features=num_Gfeatures, channel=3, first_kernel=6)
            if not ignoreD:
                netD = DCDiscriminator32(num_features=num_Dfeatures, channel=3, first_kernel=6)
    if ignoreD:
      netD = None
    return netG, netD







