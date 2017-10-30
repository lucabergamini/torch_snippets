import torch
import numpy
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.autograd import Variable
import cPickle
from tensorboard import SummaryWriter
from torch.optim import SGD, Adam
import progressbar


class ConvLayer(nn.Module):
    """
    blocco convolutivo
    se compress e True esegue compressione prima della convoluzione atraverso un filtro convolutivo 1x1
    """
    def __init__(self, in_features, k, compress=True):
        super(ConvLayer, self).__init__()
        self.compress = compress
        if compress:
            # bn
            self.bn_0 = nn.BatchNorm2d(num_features=in_features)
            self.bn_1 = nn.BatchNorm2d(num_features=4 * k)
            # conv
            self.conv_0 = nn.Conv2d(in_channels=in_features, out_channels=4 * k, kernel_size=1, padding=0,bias=False)
            self.conv_1 = nn.Conv2d(in_channels=4 * k, out_channels=k, kernel_size=3, padding=1,bias=False)
        else:
            # bn
            self.bn_0 = nn.BatchNorm2d(num_features=in_features)
            # conv
            self.conv_0 = nn.Conv2d(in_channels=in_features, out_channels=k, kernel_size=3, padding=1,bias=False)

    def forward(self, ten):
        if self.compress:
            ten = self.bn_0(ten)
            ten = F.relu(ten,inplace=True)
            ten = self.conv_0(ten)
            ten = self.bn_1(ten)
            ten = F.relu(ten,inplace=True)
            ten = self.conv_1(ten)
            ten = F.dropout(ten, p=0.2, training=self.training)
        else:
            ten = self.bn_0(ten)
            ten = F.relu(ten, inplace=True)
            ten = self.conv_0(ten)
            ten = F.dropout(ten, p=0.2, training=self.training)
        return ten


class DenseBlock(nn.Module):
    def __init__(self, in_features, num_layers, k,compress = True):
        super(DenseBlock, self).__init__()
        for i in xrange(num_layers):
            self.add_module("conv_{}".format(i),ConvLayer(in_features=in_features,k=k,compress= compress))
            in_features += k
        self.out_features = in_features

    def forward(self, ten):
        for m in self.children():
            ten_input = ten
            ten = m(ten)
            ten = torch.cat((ten,ten_input),1)
        return ten
    def __call__(self, *args, **kwargs):
        return super(DenseBlock,self).__call__(*args,**kwargs)


class TransitionLayer(nn.Module):
    """
    dimezza dimensione spaziale, se half True dimezza anche quella dei filtri
    """
    def __init__(self, in_features, half = True):
        super(TransitionLayer, self).__init__()
        self.bn_0 = nn.BatchNorm2d(num_features=in_features)
        if half:
            self.conv_0 = nn.Conv2d(in_channels=in_features, out_channels=in_features//2, kernel_size=1,bias=False)
        else:
            self.conv_0 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=1,
                                        bias=False)

    def forward(self, ten):
        ten = self.bn_0(ten)
        ten = F.relu(ten,inplace=True)
        ten = self.conv_0(ten)
        ten = F.avg_pool2d(input=ten, kernel_size=2, stride=2)
        return ten


class ClassificationLayer(nn.Module):
    def __init__(self, in_features, num_classes):
        super(ClassificationLayer, self).__init__()
        self.bn_0 = nn.BatchNorm2d(num_features=in_features)
        self.fc_0 = nn.Linear(in_features=in_features, out_features=num_classes)

    def forward(self, ten):
        ten = F.relu(ten,inplace=True)
        ten = self.bn_0(ten)
        ten = F.avg_pool2d(ten, kernel_size=ten.size()[-2:])
        ten = self.fc_0(ten.view(len(ten), -1))
        ten = F.log_softmax(ten)
        return ten


class DenseNet(nn.Module):
    """
    Densenet as Encoder, 4 blocks
    """
    def __init__(self, k, in_features, layers=(6, 12, 36, 24),num_classes=None,imagenet=True):
        """

        :param k: filtri convolutivi
        :param layers: quanti blocchi mettere e quanti filtri in ognuno
        :param imagenet: se True applica una convoluzione 7x7 stride 2 e un maxpool iniziale, per ridurre overhead
        """
        super(DenseNet, self).__init__()
        # convoluzione iniziale
        # aggiungo moduli
        #diverse liste per poter accedere agli intermedi
        self.blocks_pre = nn.ModuleList()
        self.blocks_dense = nn.ModuleList()
        self.blocks_tran = nn.ModuleList()
        #dimensioni output
        self.output_size_denset = []

        if imagenet:
            self.blocks_pre.append(nn.Sequential(nn.Conv2d(in_channels=in_features,out_channels=2*k,kernel_size=7,stride=2,padding=3),nn.MaxPool2d(kernel_size=2,stride=2)))
        else:
            self.blocks_pre.append(nn.Conv2d(in_channels=in_features,out_channels=2*k,kernel_size=3,padding=1))

        in_features = 2 * k
        for i, j in enumerate(layers):
            dense = DenseBlock(in_features=in_features, num_layers=j, k=k)
            in_features = dense.out_features
            self.output_size_denset.append(in_features)
            if i == len(layers) - 1 and num_classes is not None:
                tran = ClassificationLayer(in_features=in_features, num_classes=num_classes)

            else:
                tran = TransitionLayer(in_features=in_features)
                in_features = in_features//2
            self.blocks_dense.append(dense)
            self.blocks_tran.append(tran)

        self.output_size = in_features

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                #module.bias.data.zero_()
                module.weight.data = torch.randn(module.weight.size()) * (2.0 / (
                module.kernel_size[0] * module.kernel_size[1] * module.out_channels)) ** 0.5
            elif isinstance(module, nn.Linear):
                module.bias.data.zero_()

    def forward(self, ten):

        for block in self.blocks_pre:
            ten = block(ten)
        for block_dense,block_tran in zip(self.blocks_dense,self.blocks_tran):
            ten = block_dense(ten)
            ten = block_tran(ten)
        return ten