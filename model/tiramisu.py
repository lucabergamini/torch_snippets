import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F

from densenet import DenseBlock, TransitionLayer


class UpLayer(nn.Module):
    def __init__(self, in_features):
        super(UpLayer, self).__init__()
        self.conv = nn.ConvTranspose2d(kernel_size=2, stride=2, in_channels=in_features, out_channels=in_features)

    def forward(self, ten):
        return self.conv(ten)


class Tiramisu(nn.Module):
    def __init__(self, in_features=3, num_classes=10, layers=(4, 5, 7, 10, 12), bottleneck=15, compress=False):
        super(Tiramisu, self).__init__()
        # convoluzione iniziale
        self.conv_0 = nn.Conv2d(in_channels=in_features, out_channels=48, kernel_size=3, padding=1)
        init_size = 48
        self.down_dense = nn.ModuleList()
        self.down_tran = nn.ModuleList()
        self.up_dense = nn.ModuleList()
        self.up_tran = nn.ModuleList()
        # DOWN
        for l in layers:
            # print init_size
            block = DenseBlock(in_features=init_size, num_layers=l, k=16, compress=compress)
            self.down_dense.append(block)
            init_size = block.out_features
            block = TransitionLayer(in_features=init_size, half=False)
            self.down_tran.append(block)
        # bottleneck
        # da qui sego della roba in output e quindi mi serve mantere dei temporanei
        init_size_temp = init_size
        self.bottle = DenseBlock(in_features=init_size, num_layers=bottleneck, k=16, compress=compress)
        # print self.bottle.out_features
        init_size = self.bottle.out_features - init_size_temp

        # UP
        for i, l in enumerate(layers[::-1]):
            block = UpLayer(in_features=init_size)
            init_size += self.down_dense[len(self.down_dense) - i - 1].out_features
            self.up_tran.append(block)
            init_size_temp = init_size
            block = DenseBlock(in_features=init_size, num_layers=l, k=16, compress=compress)
            # print block.out_features
            if i < len(layers) - 1:
                init_size = block.out_features - init_size_temp
            else:
                init_size = block.out_features
            self.up_dense.append(block)
        self.classification_conv = nn.Conv2d(in_channels=init_size, out_channels=num_classes, kernel_size=1)
        #INIT PARAMETRI REGISTRATI
        self.init_parameters()

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m,(nn.Conv2d,nn.ConvTranspose2d,nn.Linear)):
                if hasattr(m,"weight") and m.weight is not None and m.weight.requires_grad:
                    nn.init.xavier_normal(m.weight)
                if hasattr(m,"bias") and m.bias is not None and m.bias.requires_grad:
                    nn.init.normal(m.bias,mean=0.0,std=0.05)


    def forward(self, ten):
        ten = self.conv_0(ten)
        # DOWN
        ten_down = []
        for dense, tran in zip(self.down_dense, self.down_tran):
            ten = dense(ten)
            ten_down.append(ten)
            ten = tran(ten)

        # BOTTLE
        # qui devo togliere un pezzo dell output, cioe le prime tot
        input_size = ten.size(1)
        ten = self.bottle(ten)
        ten = ten[:, :ten.size(1) - input_size].contiguous()

        # UP
        for i, (dense, tran) in enumerate(zip(self.up_dense, self.up_tran)):
            ten = tran(ten)
            # qui devo togliere un pezzo dell output, cioe le prime tot
            ten = torch.cat((ten, ten_down[len(ten_down) - i - 1]), 1)
            input_size = ten.size(1)
            ten = dense(ten)
            if i < len(self.up_dense) - 1:
                ten = ten[:, :ten.size(1) - input_size].contiguous()

        ten = self.classification_conv(ten)

        # adesso sara batchx224x224xnum_classes
        ten = F.log_softmax(ten)
        return ten
