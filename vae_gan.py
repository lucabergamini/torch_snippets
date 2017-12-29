import torch
import numpy
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from torch.optim import SGD, Adam, RMSprop
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import progressbar
from torchvision.utils import make_grid
import cv2
from utils.metrics import classification_accuracy, RollingMeasure
from sdn.densenet import ConvLayer
from generator.flowers import FLOWER
from matplotlib import pyplot
import torchvision
from torchvision import transforms
from utils.callbacks import ModelCheckpoint

numpy.random.seed(2)


class EncoderBlock(nn.Module):
    def __init__(self,channel_in,channel_out):
        super(EncoderBlock,self).__init__()
        self.conv = nn.Conv2d(in_channels=channel_in,out_channels=channel_out,kernel_size=3,padding=1)
        self.bn = nn.BatchNorm2d(num_features=channel_out)

    def forward(self,ten):
        ten = self.conv(ten)
        ten = self.bn(ten)
        ten = F.relu(ten,inplace=True)
        return ten

class DecoderBlock(nn.Module):
    def __init__(self,channel_in,channel_out):
        super(DecoderBlock,self).__init__()
        self.conv = nn.ConvTranspose2d(channel_in, channel_out, 7, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(channel_out)

    def forward(self,ten):
        ten = self.conv(ten)
        ten = self.bn(ten)
        ten = F.relu(ten,inplace=True)
        return ten

class Encoder(nn.Module):

    def __init__(self,channel_in=3,z_size=2):
        super(Encoder,self).__init__()
        #224->112->64->32
        self.size = channel_in
        layers_list = []
        for i in range(5):
            if i == 0:
                layers_list.append(EncoderBlock(channel_in=self.size,channel_out=16))
                layers_list.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

                self.size=16
            else:
                layers_list.append(EncoderBlock(channel_in=self.size, channel_out=self.size*2))
                layers_list.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
                self.size *=2

        self.conv = nn.Sequential(*layers_list)
        self.l_mu = nn.Linear(in_features=4096,out_features=z_size)
        self.l_var = nn.Linear(in_features=4096,out_features=z_size)


    def forward(self,ten):

        ten = self.conv(ten)

        ten = ten.view(len(ten),-1)
        mu = self.l_mu(ten)
        logvar = self.l_var(ten)
        return mu,logvar

    def __call__(self, *args, **kwargs):
        return super(Encoder,self).__call__(*args,**kwargs)


class Decoder(nn.Module):

    def __init__(self,z_size):
        super(Decoder,self).__init__()
        #B*z_size*1*1
        self.z_size = z_size

        self.conv = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(z_size, self.z_size * 8, 7, 1, 0, bias=False),
            nn.BatchNorm2d(self.z_size * 8),
            nn.ReLU(True),
            # state size. (self.z_size*8) x 7 x 7
            nn.ConvTranspose2d(self.z_size * 8, self.z_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.z_size * 4),
            nn.ReLU(True),
            # state size. (self.z_size*4) x 14 x 14
            nn.ConvTranspose2d(self.z_size * 4, self.z_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.z_size * 2),
            nn.ReLU(True),
            # state size. (self.z_size*2) x 28 x 28
            nn.ConvTranspose2d(self.z_size * 2, self.z_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.z_size),
            nn.ReLU(True),
            # state size. (self.z_size*2) x 56 x 56
            nn.ConvTranspose2d(self.z_size, 3, 4, 2, 1, bias=False),
            # state size. (nc) x 112 x 112
        )
    def forward(self,ten):
        ten = ten.view(-1,self.z_size,1,1)
        ten = self.conv(ten)
        ten = F.tanh(ten)
        #ten = F.sigmoid(ten)
        return ten
    def __call__(self, *args, **kwargs):
        return super(Decoder,self).__call__(*args,**kwargs)

class Discriminator(nn.Module):

    def __init__(self,channel_in=3):
        super(Discriminator,self).__init__()
        #224->112->64->32
        self.size = channel_in
        layers_list = []
        for i in range(5):
            if i == 0:
                layers_list.append(EncoderBlock(channel_in=self.size,channel_out=8))
                layers_list.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

                self.size=8
            else:
                layers_list.append(EncoderBlock(channel_in=self.size, channel_out=self.size*2))
                layers_list.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
                self.size *=2

        self.conv = nn.Sequential(*layers_list)
        self.fc = nn.ModuleList()
        self.fc.append(nn.Linear(in_features=2048,out_features=256))
        self.fc.append(nn.Linear(in_features=256,out_features=32))
        self.fc.append(nn.Linear(in_features=32,out_features=1))



    def forward(self,ten):

        ten = self.conv(ten)
        ten = ten.view(len(ten),-1)
        for i,l in enumerate(self.fc):
            if i != len(self.fc)-1:
                ten = l(ten)
                ten = F.relu(ten,inplace=True)
            else:
                #mando in output questo, che e 32
                inner_layer = ten
                ten = l(ten)
                ten = F.sigmoid(ten)

        return ten,inner_layer

    def __call__(self, *args, **kwargs):
        return super(Discriminator,self).__call__(*args,**kwargs)


class VAE_GAN(nn.Module):
    def __init__(self):
        super(VAE_GAN, self).__init__()
        self.z_size = 128
        self.encoder = Encoder(z_size=self.z_size)
        self.decoder = Decoder(z_size=self.z_size)
        self.discriminator = Discriminator()
        self.init_parameters()

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                if hasattr(m, "weight") and m.weight is not None and m.weight.requires_grad:
                    nn.init.xavier_normal(m.weight)
                if hasattr(m, "bias") and m.bias is not None and m.bias.requires_grad:
                    nn.init.normal(m.bias, mean=0.0, std=0.05)

    def forward(self, ten,gen_size=10):
        # ten di shape Bx784
        if self.training:
            imgs_original = ten
            #codifico
            mus,log_variances = self.encoder(ten)
            #mus e vars ha dimensione B*z_size
            #sampling da normale, tanti quanti B*z_size
            variances = torch.exp(log_variances*0.5)
            ten_from_normal = Variable(torch.randn(len(ten),self.z_size).cuda(),requires_grad=False)
            #sposto gaussiana e la scalo
            ten = ten_from_normal*(variances)+mus
            #decodifico
            imgs_rec = self.decoder(ten)
            #concateno
            imgs = torch.cat((imgs_original,imgs_rec))
            #discriminatore con entrambe
            imgs_classes,inner_layers = self.discriminator(imgs)
            return imgs_rec,imgs_classes,inner_layers,mus, log_variances

        else:
            ten = Variable(torch.randn(gen_size,self.z_size).cuda(),requires_grad=False)
            #questi sono i z che mando al decoder
            ten = self.decoder(ten)

            return ten,None,None,None

    def __call__(self, *args, **kwargs):
        return super(VAE_GAN, self).__call__(*args, **kwargs)

    def vae_gan_loss(self,ten_original,ten_predict,mus,variances,classes_predicted,layer_predicted):
        band =0.1
        #ricostruzione
        nle = torch.mean(torch.sum((ten_original.view(len(ten_original),-1)-ten_predict.view(len(ten_predict),-1))**2,1)/band)
        #KL
        kl = torch.mean((torch.sum(variances.exp(),1)+torch.sum(mus**2,1)-torch.sum(variances,1)-self.z_size))
        #discriminatore
        #cat mette prima originale e poi ric, quindi le prime B devono essere 1,le seconde B 0
        classes_original = Variable(torch.zeros_like(classes_predicted.data),requires_grad=False).cuda()
        classes_original[0:len(classes_predicted)//2] = 1
        classes_original[len(classes_predicted)//2:] = 0
        disc = nn.BCELoss()(classes_predicted,classes_original)
        #layers
        #qui confronto tra prime B e seconde B
        lay = torch.mean(torch.sum((layer_predicted[0:len(layer_predicted)//2]-layer_predicted[len(layer_predicted)//2:])**2,1))
        return nle,kl,disc,lay





if __name__ == "__main__":


    writer = SummaryWriter(comment="_FLOWERS")
    check = ModelCheckpoint()
    net = VAE_GAN().cuda()
    # DATASET
    dataloader = torch.utils.data.DataLoader(FLOWER("/home/lapis/Desktop/flowers/",data_aug=True), batch_size=32,
                                             shuffle=True, num_workers=2)
    # DATASET
    dataloader_test = torch.utils.data.DataLoader(FLOWER("/home/lapis/Desktop/flowers/"), batch_size=32,
                                             shuffle=False, num_workers=2)
    TEST = True


    # OPTIM-LOSS
    #disc sul classificatore
    #le altre sul resto
    optimizer_enc = Adam(params=[p for p in net.encoder.parameters()], lr=0.00035)
    optimizer_dec = Adam(params=[p for p in net.decoder.parameters()], lr=0.00035)
    optimizer_dis = Adam(params=[p for p in net.discriminator.parameters()], lr=0.00035)

    #lr_sc = StepLR(optimizer,step_size=30,gamma=0.9)
    #lr_sc = MultiStepLR(optimizer,milestones=[10,20,30,40,50,60,75],gamma=0.85)
    batch_number = len(dataloader)

    num_epochs = 500
    step_index=0
    widgets = [

        'Batch: ', progressbar.Counter(),
        '/', progressbar.FormatCustomText('%(total)s', {"total": batch_number}),
        ' ', progressbar.Bar(marker="-", left='[', right=']'),
        ' ', progressbar.ETA(),
        ' ',
        progressbar.DynamicMessage('loss_nle'),
        ' ',
        progressbar.DynamicMessage('loss_kl'),
        ' ',
        progressbar.DynamicMessage('loss_disc'),
        ' ',
        progressbar.DynamicMessage('loss_lay'),
        ' ',
        progressbar.DynamicMessage("epoch")
    ]

    a_lay = 0
    a_vae = 1
    a_d1 = 0
    a_d2 = 0
    for i in range(num_epochs):
        progress = progressbar.ProgressBar(min_value=0, max_value=batch_number, initial_value=0, widgets=widgets).start()
        loss_nle_mean = RollingMeasure()
        loss_kl_mean = RollingMeasure()
        loss_disc_mean = RollingMeasure()
        loss_lay_mean = RollingMeasure()

        for j, (data_batch, labels_batch) in enumerate(dataloader):
            net.train()
            # trasformo in variabili

            data_target = Variable(torch.squeeze(data_batch),requires_grad=False).float().cuda()
            data_in = Variable(data_batch, requires_grad=True).float().cuda()


            # azzero gradiente
            net.zero_grad()
            # calcolo uscita
            out, out_classes, inner_layers, mus, log_variances = net(data_in)
            # loss
            loss_nle_value, loss_kl_value, loss_disc_value, loss_lay_value = net.vae_gan_loss(data_target,out,mus,log_variances,out_classes,inner_layers)
            # propago
            #aggiorno encoder
            (a_vae+loss_nle_value+a_vae * loss_kl_value + a_lay * loss_lay_value).backward(retain_graph=True)
            optimizer_enc.step()
            net.zero_grad()
            # aggiorno decoder
            (a_vae+loss_nle_value+a_lay * loss_lay_value-a_d1*loss_disc_value).backward(retain_graph=True)
            optimizer_dec.step()
            net.zero_grad()
            # aggiorno discriminator
            (a_d2*loss_disc_value).backward()
            optimizer_dis.step()

            # LOGGING

            progress.update(progress.value + 1, loss_nle=loss_nle_mean(loss_nle_value.data.cpu().numpy()[0]),
                            loss_kl=loss_kl_mean(loss_kl_value.data.cpu().numpy()[0]),
                            loss_disc=loss_disc_mean(loss_disc_value.data.cpu().numpy()[0]),
                            loss_lay=loss_lay_mean(loss_lay_value.data.cpu().numpy()[0]),
                             epoch=i + 1)


        # FINE EPOCA
        progress.finish()
        #check(net,loss_nle_mean.measure)

        writer.add_scalar('loss_reconstruction', loss_nle_mean.measure, step_index)
        writer.add_scalar('loss_KLD', loss_kl_mean.measure, step_index)
        writer.add_scalar('loss_disc', loss_disc_mean.measure, step_index)
        writer.add_scalar('loss_lay', loss_lay_mean.measure, step_index)

        for j, (data_batch, labels_batch) in enumerate(dataloader_test):

            data_in = Variable(data_batch, requires_grad=True).float().cuda()
            out = net(data_in)
            out = out[0].data.cpu()
            #riporto in 0-1
            out = (out+1)/2
            out = make_grid(out,nrow=4)
            writer.add_image("reconstructed", out, step_index)

            net.eval()
            out = net(None, 32)
            out = out[0].data.cpu()
            out = (out+1)/2
            out = make_grid(out, nrow=4)
            writer.add_image("generated", out, step_index)

            out = data_in.data.cpu()
            out = (out+1)/2
            out = make_grid(out, nrow=4)
            writer.add_image("original", out, step_index)
            break

        step_index += 1
