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

class Encoder(nn.Module):

    def __init__(self,channel_in=3,z_size=2):
        super(Encoder,self).__init__()
        #224->112->64->32
        self.size = channel_in
        layers_list = []
        for i in range(5):
            layers_list.append(ConvLayer(in_features=self.size,k=28,compress=False))
            layers_list.append(nn.MaxPool2d(kernel_size=3,stride=2,padding=1))
            self.size+=28
        #143*4*4 che mandero in avgpool
        self.conv = nn.Sequential(*layers_list)
        self.l_mu = nn.Linear(in_features=2288,out_features=z_size)
        self.l_var = nn.Linear(in_features=2288,out_features=z_size)


    def forward(self,ten):

        ten = self.conv(ten)
        #ten = F.avg_pool3d(ten,kernel_size=(ten.shape[1],1,1))
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



class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.z_size = 128
        self.encoder = Encoder(z_size=self.z_size)
        self.decoder = Decoder(z_size=self.z_size)
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
            mus,log_variances = self.encoder(ten)

            #mus e vars ha dimensione B*z_size
            #sampling da normale, tanti quanti B*z_size
            variances = torch.exp(log_variances*0.5)
            ten_from_normal = Variable(torch.randn(len(ten),self.z_size).cuda(),requires_grad=False)
            #sposto gaussiana e la scalo
            ten = ten_from_normal*(variances)+mus
            ten = self.decoder(ten)
            return ten, mus, log_variances

        else:
            ten = Variable(torch.randn(gen_size,self.z_size).cuda(),requires_grad=False)
            #questi sono i z che mando al decoder
            ten = self.decoder(ten)

            return ten,None,None

    def __call__(self, *args, **kwargs):
        return super(VAE, self).__call__(*args, **kwargs)

    def vae_loss(self,ten_original,ten_predict,mus,variances):
        #loss da teoria
        #ricostruzione
        band =0.1
        nle = torch.mean(torch.sum((ten_original.view(len(ten_original),-1)-ten_predict.view(len(ten_predict),-1))**2,1)/band)

        #nle = torch.mean(torch.sum((ten_original.view(len(ten_original),-1)-ten_predict.view(len(ten_predict),-1))**2,1)/(2*band**2))


        kl = torch.mean((torch.sum(variances.exp(),1)+torch.sum(mus**2,1)-torch.sum(variances,1)-self.z_size))

        #kl = -0.5*torch.mean(1+variances-variances.exp()-mus**2)
        #kl /=(112*112)
        #kl *=self.z_size
        return nle,kl





if __name__ == "__main__":


    writer = SummaryWriter(comment="_FLOWERS")
    check = ModelCheckpoint()
    net = VAE().cuda()
    # DATASET
    dataloader = torch.utils.data.DataLoader(FLOWER("/home/lapis/Desktop/flowers/",data_aug=True), batch_size=32,
                                             shuffle=True, num_workers=2)
    # DATASET
    dataloader_test = torch.utils.data.DataLoader(FLOWER("/home/lapis/Desktop/flowers/"), batch_size=32,
                                             shuffle=False, num_workers=2)
    TEST = True


    # OPTIM-LOSS
    #optimizer = Adam(params=net.parameters(), lr=0.00255)
    optimizer = Adam(params=net.parameters(), lr=0.0005)

    lr_sc = StepLR(optimizer,step_size=30,gamma=0.95)
    #lr_sc = MultiStepLR(optimizer,milestones=[10,20,30,40,50,60,75],gamma=0.85)
    batch_number = len(dataloader)

    num_epochs = 300
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
        progressbar.DynamicMessage("epoch")
    ]

    for i in range(num_epochs):
        progress = progressbar.ProgressBar(min_value=0, max_value=batch_number, initial_value=0, widgets=widgets).start()
        loss_nle_mean = RollingMeasure()
        loss_kl_mean = RollingMeasure()

        for j, (data_batch, labels_batch) in enumerate(dataloader):
            net.train()
            # trasformo in variabili

            data_target = Variable(torch.squeeze(data_batch),requires_grad=False).float().cuda()
            data_in = Variable(data_batch, requires_grad=True).float().cuda()


            # azzero gradiente
            net.zero_grad()
            optimizer.zero_grad()
            # calcolo uscita
            out,mus,variances = net(data_in)
            # loss
            loss_nle_value,loss_kl_value = net.vae_loss(data_target,out,mus,variances)
            #loss_nle_value,loss_kl_value  = net.vae_loss(data_target,out,mus,variances)
            # propago
            (loss_nle_value+loss_kl_value).backward()
            optimizer.step()

            # LOGGING

            progress.update(progress.value + 1, loss_nle=loss_nle_mean(loss_nle_value.data.cpu().numpy()[0]),
                            loss_kl=loss_kl_mean(loss_kl_value.data.cpu().numpy()[0]),
                             epoch=i + 1)


        # FINE EPOCA
        progress.finish()
        #check(net,loss_nle_mean.measure)

        writer.add_scalar('loss_reconstruction', loss_nle_mean.measure, step_index)
        writer.add_scalar('loss_KLD', loss_kl_mean.measure, step_index)
        lr_sc.step()

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
