# PyTorch implementation for the Stacked Deconvolutional Network
# Reference https://arxiv.org/abs/1708.04943

import torch
from datetime import datetime
import torch.nn as nn
import torch.nn.functional as F
import numpy
import progressbar
import os
from torch.autograd import Variable
from torch.optim import Adam
from tensorboard import SummaryWriter
from densenet import DenseNet


class ConvLayer(nn.Module):
    """
    Layer convolutivo di base come in DENSENET
    """

    def __init__(self, in_channels,out_channels, drop=True):
        """

        :param in_channels: numero canali in ingresso
        :param drop: se applicare drop_out,per riuso codice
        """
        super(ConvLayer, self).__init__()
        # conv con padding 1 per non cambiare shape
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding=1)
        # bn
        self.bn = nn.BatchNorm2d(num_features=in_channels)
        # drop
        self.drop = drop

    def forward(self, ten):
        ten = self.bn(ten)
        ten = F.relu(ten)
        ten = self.conv(ten)
        if self.drop:
            ten = F.dropout(ten, p=0.2, training=self.training)
        return ten

    def __call__(self, *args, **kwargs):
        return super(ConvLayer, self).__call__(*args, **kwargs)


class ClassificationLayer(nn.Module):
    """
    Layer per categorizzazione supervisionata
    """

    def __init__(self, in_channels, num_classes, up_sample):
        """

        :param in_channels: canali in ingresso per conv
        :param num_classes: numero di classi
        :param up_sample: di quanto ingradire l'immagine
        """
        super(ClassificationLayer, self).__init__()
        # conv per classi
        self.up_sample = up_sample
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=num_classes, kernel_size=(3, 3),padding=1)
        self.soft = nn.Softmax2d()

    def forward(self, ten, ten_classification=None):
        """

        :param ten: tensore input
        :param ten_classification: tensore precedente da sommare
        :return: tensore da sommare nella prossima unita, tensore output
        """
        ten = self.conv(ten)
        if ten_classification is not None:
            ten_classification = ten + ten_classification
        else:
            ten_classification = ten
        ten = F.upsample(ten_classification,scale_factor= self.up_sample)
        print "upsample {}".format(ten.size())
        ten = self.soft(ten)
        return (ten_classification, ten)

    def __call__(self, *args, **kwargs):
        return  super(ClassificationLayer, self).__call__( *args, **kwargs)


class CompressionLayer(nn.Module):
    """
    Layer di compressione, e una semplice convoluzione
    """

    def __init__(self, in_channels, out_channels):
        super(CompressionLayer, self).__init__()
        # conv
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3),padding=1)

    def forward(self, ten):
        ten = self.conv(ten)
        return ten

    def __call__(self, *args, **kwargs):
        return super(CompressionLayer, self).__call__( *args, **kwargs)


class Encoder(nn.Module):
    """
    Blocco encoder
    """

    def __init__(self, input_inter_sizes, input_size,k=48):
        """
        :param input_inter_sizes: dimensioni input inter-link da decoder
        :param input_size: dimensione input
        """
        super(Encoder, self).__init__()

        # la prima volta metto input le alre 48 da concatenazione
        init_size = input_size
        init_size += input_inter_sizes[0]

        self.conv_layer_0 = ConvLayer(in_channels=init_size, out_channels=k)
        init_size += k
        self.conv_layer_1 = ConvLayer(in_channels=init_size, out_channels=k)
        init_size += k
        self.comp_layer_0 = CompressionLayer(in_channels=init_size, out_channels=768)

        init_size = 768+ input_inter_sizes[1]

        self.conv_layer_2 = ConvLayer(in_channels=init_size, out_channels=k)
        init_size += k
        self.conv_layer_3 = ConvLayer(in_channels=init_size, out_channels=k)
        init_size += k
        self.comp_layer_1 = CompressionLayer(in_channels=init_size, out_channels=1024)

        self.output_size= 1024

    def forward(self, ten, ten_inputs):
        """

        :param ten: tensore input
        :param ten_inputs: tensori inter-link da decoder
        :return: tensore compresso
        """
        # dimezza dimensioni
        ten = F.max_pool2d(ten, kernel_size=(2, 2), stride=2)
        # inter-link da decoder precedente da concatenare sui canali
        ten = torch.cat((ten, ten_inputs[0]), dim=1)
        # intra-link
        ten_intra = ten
        ten = self.conv_layer_0(ten)
        ten = torch.cat((ten, ten_intra), dim=1)
        # intra-link
        ten_intra = ten
        ten = self.conv_layer_1(ten)
        ten = torch.cat((ten, ten_intra), dim=1)
        ten = self.comp_layer_0(ten)

        # dimezza dimensioni
        ten = F.max_pool2d(ten, kernel_size=(2, 2), stride=2)
        # inter-link da decoder precedente da concatenare sui canali
        ten = torch.cat((ten, ten_inputs[1]), dim=1)
        # intra-link
        ten_intra = ten
        ten = self.conv_layer_2(ten)
        ten = torch.cat((ten, ten_intra), dim=1)
        # intra-link
        ten_intra = ten
        ten = self.conv_layer_3(ten)
        ten = torch.cat((ten, ten_intra), dim=1)
        ten = self.comp_layer_1(ten)

        return ten

    def __call__(self, *args, **kwargs):
        return super(Encoder, self).__call__( *args, **kwargs)


class Decoder(nn.Module):
    """
    Blocco decoder
    """

    def __init__(self, input_inter_sizes, input_size,k=48):
        """
        :param input_inter_sizes: dimensioni input inter-link da primo encoder DENSENET
        :param input_size: dimensione input
        """
        super(Decoder, self).__init__()
        #dimensioni output per encoder successivo
        self.output_size_encoder = [input_size,]
        # la prima volta metto input le alre 48 da concatenazione
        init_size = input_size
        self.class_0 = ClassificationLayer(in_channels=init_size,num_classes=50,up_sample=16)
        self.deconv_0 = nn.ConvTranspose2d(in_channels=init_size, out_channels=init_size, kernel_size=(2, 2), stride=2)
        init_size += input_inter_sizes[0]

        self.conv_layer_0 = ConvLayer(in_channels=init_size, out_channels=k)
        init_size += k
        self.conv_layer_1 = ConvLayer(in_channels=init_size, out_channels=k)
        init_size += k
        self.comp_layer_0 = CompressionLayer(in_channels=init_size, out_channels=768)
        init_size = 768
        self.class_1 = ClassificationLayer(in_channels=init_size,num_classes=50,up_sample=8)
        self.output_size_encoder.append(init_size)

        self.deconv_1 = nn.ConvTranspose2d(in_channels=init_size, out_channels=init_size, kernel_size=(2, 2), stride=2)
        init_size += input_inter_sizes[1]

        self.conv_layer_2 = ConvLayer(in_channels=init_size, out_channels=k)
        init_size += k
        self.conv_layer_3 = ConvLayer(in_channels=init_size, out_channels=k)
        init_size += k
        self.comp_layer_1 = CompressionLayer(in_channels=init_size, out_channels=576)
        init_size = 576
        self.class_2 = ClassificationLayer(in_channels=init_size,num_classes=50,up_sample=4)

        self.output_size= init_size

    def forward(self, ten, ten_inputs, ten_classification=(None,None,None)):
        """

        :param ten: tensore input
        :param ten_inputs: tensori inter-link da primo encoder DENSENET
        :param ten_classification: tensori classificazione decoder precedente da sommare
        :return: tensore compresso,tensori classificazione,tensori per encoders
        """
        output_classification = []
        output_encoder = [ten,]
        output_classification.append(self.class_0(ten, ten_classification[0]))

        # raddoppia dimensioni
        ten = self.deconv_0(ten)
        # inter-link da decoder precedente da concatenare sui canali

        ten = torch.cat((ten, ten_inputs[0]), dim=1)
        # intra-link
        ten_intra = ten
        ten = self.conv_layer_0(ten)
        ten = torch.cat((ten, ten_intra), dim=1)
        # intra-link
        ten_intra = ten
        ten = self.conv_layer_1(ten)
        ten = torch.cat((ten, ten_intra), dim=1)
        ten = self.comp_layer_0(ten)
        output_encoder.append(ten)
        output_classification.append(self.class_1(ten,ten_classification[1]))

        # raddoppia dimensioni
        ten = self.deconv_1(ten)
        # inter-link da decoder precedente da concatenare sui canali
        ten = torch.cat((ten, ten_inputs[1]), dim=1)
        # intra-link
        ten_intra = ten
        ten = self.conv_layer_2(ten)
        ten = torch.cat((ten, ten_intra), dim=1)
        # intra-link
        ten_intra = ten
        ten = self.conv_layer_3(ten)
        ten = torch.cat((ten, ten_intra), dim=1)
        ten = self.comp_layer_1(ten)
        output_classification.append(self.class_2(ten,ten_classification[2]))


        return (ten,output_encoder,output_classification)

    def __call__(self, *args, **kwargs):
        return super(Decoder, self).__call__( *args, **kwargs)




class SDN(nn.Module):
    def __init__(self,k=48):
        super(SDN, self).__init__()
        #primo encoder DENSENET
        self.enc_0 = DenseNet(k=k,in_features=3,layers=(6,12))
        # ho i due filtri convolutivi
        self.conv_filters = nn.ModuleList([ConvLayer(in_channels=self.enc_0.output_size_denset[0],out_channels=k, drop=False),
                        ConvLayer(in_channels=self.enc_0.output_size_denset[1],out_channels=k, drop=False)])
        self.dec_0 = Decoder(input_inter_sizes=(k,k),input_size=self.enc_0.output_size,k=k)

        self.enc_1 = Encoder(input_inter_sizes=self.dec_0.output_size_encoder[::-1],input_size=self.dec_0.output_size,k=k)
        self.dec_1 = Decoder(input_inter_sizes=(k,k),input_size=self.enc_1.output_size,k=k)

        self.enc_2 = Encoder(input_inter_sizes=self.dec_1.output_size_encoder[::-1], input_size=self.dec_0.output_size,k=k)
        self.dec_2 = Decoder(input_inter_sizes=(k, k), input_size=self.enc_2.output_size,k=k)



    def forward(self, ten):
        output_classification_final = []
        print "0"
        ten,output_densenet_for_decoder = self.enc_0(ten)
        print "densenet"
        #devo scambiarli
        output_densenet_for_decoder = [conv(output_densenet_for_decoder[i]) for i,conv in enumerate(self.conv_filters)]
        ten,output_dencoder_for_encoder,output_classification=self.dec_0(ten,output_densenet_for_decoder[::-1])
        output_classification_final.append(output_classification)
        print "1"
        ten = self.enc_1(ten,output_dencoder_for_encoder[::-1])
        ten, output_dencoder_for_encoder, output_classification = self.dec_1(ten, output_densenet_for_decoder[::-1])
        output_classification_final.append(output_classification)
        print "2"
        ten = self.enc_2(ten,output_dencoder_for_encoder[::-1])
        ten, output_dencoder_for_encoder, output_classification = self.dec_2(ten, output_densenet_for_decoder[::-1])
        output_classification_final.append(output_classification)

        return output_classification_final

    def __call__(self, *args, **kwargs):
        super(SDN,self).__call__(*args,**kwargs)


net = SDN(k=48).cuda()
print "net done"
a = Variable(torch.randn(8,3,224,224)).cuda()
net(a)