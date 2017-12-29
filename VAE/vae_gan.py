import torch
import numpy
numpy.random.seed(2)
torch.manual_seed(2)
torch.cuda.manual_seed(2)
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from torch.optim import SGD, Adam, RMSprop
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ExponentialLR
import progressbar
from torchvision.utils import make_grid
import cv2
from utils.metrics import classification_accuracy, RollingMeasure
from sdn.densenet import ConvLayer
from generator.celeba import CELEBA
from matplotlib import pyplot
import torchvision
from utils.callbacks import ModelCheckpoint

numpy.random.seed(2)
torch.manual_seed(2)

class EncoderBlock(nn.Module):
    def __init__(self,channel_in,channel_out):
        super(EncoderBlock,self).__init__()
        self.conv = nn.Conv2d(in_channels=channel_in,out_channels=channel_out,kernel_size=5,padding=2,stride=2,bias=False)
        self.bn = nn.BatchNorm2d(num_features=channel_out,momentum=0.9)

    def forward(self,ten,out=False):
        if out:
            ten = self.conv(ten)
            ten_out=ten
            ten = self.bn(ten)
            ten = F.relu(ten,False)
            return ten,ten_out
        else:
            ten = self.conv(ten)
            ten = self.bn(ten)
            ten = F.relu(ten, True)
            return ten


class DecoderBlock(nn.Module):
    def __init__(self,channel_in,channel_out):
        super(DecoderBlock,self).__init__()
        self.conv = nn.ConvTranspose2d(channel_in, channel_out, kernel_size=5, padding=2, stride=2,output_padding=1,bias=False)
        self.bn = nn.BatchNorm2d(channel_out,momentum=0.9)


    def forward(self,ten):
        ten = self.conv(ten)
        ten = self.bn(ten)
        ten = F.relu(ten,True)
        return ten


class Encoder(nn.Module):

    def __init__(self,channel_in=3,z_size=2):
        super(Encoder,self).__init__()
        #64->32->16
        self.size = channel_in
        layers_list = []
        for i in range(3):
            if i == 0:
                layers_list.append(EncoderBlock(channel_in=self.size,channel_out=64))
                self.size= 64
            else:
                layers_list.append(EncoderBlock(channel_in=self.size,channel_out=self.size*2))
                self.size *=2

        #256*8*8
        self.conv = nn.Sequential(*layers_list)
        self.fc = nn.Sequential(nn.Linear(in_features=8*8*self.size,out_features=1024,bias=False),
                                  nn.BatchNorm1d(num_features=1024),
                                nn.ReLU(True))

        self.l_mu = nn.Linear(in_features=1024,out_features=z_size)
        self.l_var = nn.Linear(in_features=1024,out_features=z_size)

    def forward(self,ten):

        ten = self.conv(ten)
        ten = ten.view(len(ten),-1)
        ten = self.fc(ten)
        mu = self.l_mu(ten)
        logvar = self.l_var(ten)
        return mu,logvar

    def __call__(self, *args, **kwargs):
        return super(Encoder,self).__call__(*args,**kwargs)


class Decoder(nn.Module):

    def __init__(self,z_size,size):
        super(Decoder,self).__init__()
        #B*z_size
        self.fc = nn.Sequential(nn.Linear(in_features=z_size,out_features=8*8*size,bias=False),
                                nn.BatchNorm1d(num_features=8*8*size),
                                nn.ReLU(True))


        self.size = size
        layers_list = []
        for i in range(3):
            layers_list.append(DecoderBlock(channel_in=self.size,channel_out=self.size//2))
            self.size //=2

        #aggiungo conv finale
        layers_list.append(nn.Sequential(
            nn.Conv2d(in_channels=self.size,out_channels=3,kernel_size=5,stride=1,padding=2),
            nn.Tanh()
        ))

        self.conv = nn.Sequential(*layers_list)

    def forward(self,ten,ten_from_normal=None):
        if self.training:

            ten = self.fc(ten)
            ten = ten.view(len(ten),-1,8,8)
            ten = self.conv(ten)

            ten_from_normal = self.fc(ten_from_normal)
            ten_from_normal = ten_from_normal.view(len(ten_from_normal),-1,8,8)
            ten_from_normal = self.conv(ten_from_normal)

            return ten,ten_from_normal
        else:
            ten = self.fc(ten)
            ten = ten.view(len(ten), -1, 8, 8)
            ten = self.conv(ten)
            return ten
    def __call__(self, *args, **kwargs):
        return super(Decoder,self).__call__(*args,**kwargs)



class Discriminator(nn.Module):

    def __init__(self,channel_in=3):
        super(Discriminator,self).__init__()
        #64->32->16
        self.size = channel_in
        self.conv = nn.ModuleList()
        self.conv.append(nn.Sequential(
                           nn.Conv2d(in_channels=3,out_channels=32,kernel_size=5,stride=1,padding=2),
                                     nn.ReLU(inplace=True)))
        self.size=32
        self.conv.append(EncoderBlock(channel_in=self.size, channel_out=128))
        self.size=128
        self.conv.append(EncoderBlock(channel_in=self.size, channel_out=256))
        self.size=256
        self.conv.append(EncoderBlock(channel_in=self.size, channel_out=256))





        self.fc = nn.Sequential(
            nn.Linear(in_features=8 * 8 * self.size, out_features=512,bias=False),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=1),

        )

    def forward(self,ten_original,ten,ten_from_normal):
        ten = torch.cat((ten_original,ten,ten_from_normal),0)
        for i,lay in enumerate(self.conv):
            if i== 3:
                ten,layer_ten = lay(ten,True)

                #porto in output layer
                layer_ten = layer_ten[:-len(ten_from_normal)]
                layer_ten = layer_ten.view(len(layer_ten),-1)
            else:
                ten = lay(ten)

        ten = ten.view(len(ten),-1)
        ten = self.fc(ten)
        return F.sigmoid(ten),layer_ten

    def __call__(self, *args, **kwargs):
        return super(Discriminator,self).__call__(*args,**kwargs)




class VAE_GAN(nn.Module):
    def __init__(self):
        super(VAE_GAN, self).__init__()
        self.z_size = 128
        self.encoder = Encoder(z_size=self.z_size)
        self.decoder = Decoder(z_size=self.z_size,size=self.encoder.size)
        self.discriminator = Discriminator(channel_in=3)

        self.init_parameters()

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                if hasattr(m, "weight") and m.weight is not None and m.weight.requires_grad:
                    nn.init.xavier_normal(m.weight)
                if hasattr(m, "bias") and m.bias is not None and m.bias.requires_grad:
                    nn.init.constant(m.bias,0.0)
                    #nn.init.normal(m.bias, mean=0.0, std=0.02)

    def forward(self, ten,gen_size=10):
        # ten di shape Bx784
        if self.training:
            ten_original = ten
            mus,log_variances = self.encoder(ten)
            #mus e vars ha dimensione B*z_size
            #sampling da normale, tanti quanti B*z_size
            variances = torch.exp(log_variances*0.5)
            ten_from_normal = Variable(torch.randn(len(ten),self.z_size).cuda(),requires_grad=True)
            #sposto gaussiana e la scalo
            ten = ten_from_normal*(variances)+mus
            ten,ten_from_normal = self.decoder(ten,ten_from_normal)
            #discriminator
            ten_class,ten_layer = self.discriminator(ten_original,ten,ten_from_normal)

            return ten,ten_class,ten_layer, mus, log_variances

        else:
            ten = Variable(torch.randn(gen_size,self.z_size).cuda(),requires_grad=False)
            #questi sono i z che mando al decoder
            ten = self.decoder(ten)

            return ten,None,None

    def __call__(self, *args, **kwargs):
        return super(VAE_GAN, self).__call__(*args, **kwargs)
    #
    # def loss(self,ten_original,ten_predict,layer_original,layer_predicted,labels_original,labels_predicted,labels_sampled,mus,variances):
    #     #loss da teoria
    #     #ricostruzione fuori dal decoder(non dovrebbe servire)
    #     nle = torch.mean((ten_original.view(len(ten_original),-1)-ten_predict.view(len(ten_predict),-1))**2)
    #     #kl-divergence
    #     kl = 0.5*torch.sum(variances.exp()+mus**2-variances-1,1)
    #
    #     #mse layer
    #     mse = torch.sum((layer_original-layer_predicted)**2,1)
    #     #bce varie (input,target)
    #
    #     bce_dis_original = -torch.log(labels_original + 1e-03).view(-1)
    #     bce_dis_sampled = -torch.log((1-labels_sampled) + 1e-03).view(-1)
    #     bce_dis_predicted = -torch.log((1-labels_predicted) + 1e-03).view(-1)
    #
    #     #bce_dis_sampled = -torch.log((1-labels_sampled) + 10e-06)
    #     #print("{}:{}".format(bce_dis_original.data[0],(bce_dis_predicted+bce_dis_sampled).data[0]/2))
    #     return nle,kl,mse,bce_dis_original,bce_dis_sampled,bce_dis_predicted


    def loss(self,ten_original,ten_predict,layer_original,layer_predicted,labels_original,labels_predicted,labels_sampled,mus,variances):
        #loss da teoria
        band =1
        #ricostruzione fuori dal decoder(non dovrebbe servire)
        nle = torch.mean((ten_original.view(len(ten_original),-1)-ten_predict.view(len(ten_predict),-1))**2/band)
        #kl-divergence
        kl = torch.mean(0.5*torch.sum(variances.exp()+mus**2-variances-1,1))
        #mse layer
        mse = torch.mean(torch.sum((layer_original-layer_predicted)**2,1)/band)
        #bce varie (input,target)
        bce_gen_predicted = nn.BCELoss()(labels_predicted,Variable(torch.ones_like(labels_predicted.data).cuda(),requires_grad=False))
        bce_gen_sampled = nn.BCELoss()(labels_sampled,Variable(torch.ones_like(labels_sampled.data).cuda(),requires_grad=False))
        bce_dis_original = nn.BCELoss()(labels_original,Variable(torch.ones_like(labels_original.data).cuda(),requires_grad=False))
        bce_dis_predicted = nn.BCELoss()(labels_predicted,Variable(torch.zeros_like(labels_predicted.data).cuda(),requires_grad=False))
        bce_dis_sampled = nn.BCELoss()(labels_sampled,Variable(torch.zeros_like(labels_sampled.data).cuda(),requires_grad=False))

        return nle,kl,mse,bce_gen_predicted,bce_gen_sampled,bce_dis_original,bce_dis_predicted,bce_dis_sampled




if __name__ == "__main__":


    writer = SummaryWriter(comment="_CELEBA_loss_1_full")
    #check = ModelCheckpoint()
    net = VAE_GAN().cuda()
    # DATASET
    dataloader = torch.utils.data.DataLoader(CELEBA("/home/lapis/Desktop/img_align_celeba/"), batch_size=64,
                                             shuffle=True, num_workers=2)
    # DATASET
    dataloader_test = torch.utils.data.DataLoader(CELEBA("/home/lapis/Desktop/img_align_celeba/"), batch_size=64,
                                             shuffle=False, num_workers=2)
    TEST = True
    num_epochs = 13


    # OPTIM-LOSS
    optimizer_encoder = RMSprop(params=net.encoder.parameters(), lr=0.0001)
    lr_encoder = ExponentialLR(optimizer_encoder,gamma=0.985)
    optimizer_decoder = RMSprop(params=net.decoder.parameters(), lr=0.0001)
    lr_decoder = ExponentialLR(optimizer_decoder,gamma=0.985)
    optimizer_discriminator = RMSprop(params=net.discriminator.parameters(), lr=0.0001)
    lr_discriminator = ExponentialLR(optimizer_discriminator,gamma=0.985)


    #lr_sc = MultiStepLR(optimizer,milestones=[5,10,15],gamma=0.5)
    batch_number = len(dataloader)

    step_index=0
    widgets = [

        'Batch: ', progressbar.Counter(),
        '/', progressbar.FormatCustomText('%(total)s', {"total": batch_number}),
        ' ', progressbar.Bar(marker="-", left='[', right=']'),
        ' ', progressbar.ETA(),
        ' ',
        progressbar.DynamicMessage('loss_nle'),
        ' ',
        progressbar.DynamicMessage('loss_encoder'),
        ' ',
        progressbar.DynamicMessage('loss_decoder'),
        ' ',
        progressbar.DynamicMessage('loss_discriminator'),
        ' ',
        progressbar.DynamicMessage("epoch")
    ]

    for i in range(num_epochs):
        progress = progressbar.ProgressBar(min_value=0, max_value=batch_number, initial_value=0, widgets=widgets).start()
        loss_nle_mean = RollingMeasure()
        loss_encoder_mean = RollingMeasure()
        loss_decoder_mean= RollingMeasure()
        loss_discriminator_mean = RollingMeasure()
        print("LR:{}".format(lr_encoder.get_lr()))

        for j, (data_batch, labels_batch) in enumerate(dataloader):
            net.train()

            # trasformo in variabili

            data_target = Variable(torch.squeeze(data_batch),requires_grad=False).float().cuda()
            data_in = Variable(data_batch, requires_grad=True).float().cuda()

            # azzero gradiente
            net.zero_grad()
            # calcolo uscita
            out,out_labels,out_layer,mus,variances = net(data_in)
            out_layer_original = out_layer[:len(out_layer)//2]
            out_layer_predicted = out_layer[len(out_layer)//2:]
            out_labels_original = out_labels[:len(out_labels)//3]
            out_labels_predicted = out_labels[len(out_labels)//3:(len(out_labels)//3)*2]
            out_labels_sampled = out_labels[-len(out_labels)//3:]
            # loss
            nle_value, kl_value, mse_value, bce_gen_predicted_value, bce_gen_sampled_value, bce_dis_original_value,\
            bce_dis_predicted_value, bce_dis_sampled_value= net.loss(data_target,out,out_layer_original,out_layer_predicted,out_labels_original,out_labels_predicted,out_labels_sampled,mus,variances)
            loss_encoder = (kl_value+mse_value)
            loss_decoder = (1e-06*mse_value-(bce_dis_original_value+bce_dis_sampled_value))
            loss_discriminator = (bce_dis_original_value+bce_dis_sampled_value)

            #train di alcune parti solo se ho certe condizioni di equilibrio
            train_dis = True
            train_dec = True
            if bce_dis_original_value.data[0] < 0.33 or bce_dis_sampled_value.data[0] < 0.33:
                train_dis = False
            if bce_dis_original_value.data[0] > 0.33 or bce_dis_sampled_value.data[0] > 0.33:
                train_dec = False
            if train_dec is False and train_dis is False:
                train_dis =True
                train_dec = True

            #BACK
            # VAE e GAN uniti
            # propago loss_encoder
            net.zero_grad()
            loss_encoder.backward(retain_graph=True)
            #[p.grad.data.clamp_(-1,1) for p in net.encoder.parameters()]
            # questo non serve piu quindi posso aggiornarlo
            optimizer_encoder.step()
            # pulisco gli altri
            net.zero_grad()
            # propago loss decoder
            if train_dec:
                loss_decoder.backward(retain_graph=True)
                #[p.grad.data.clamp_(-1,1) for p in net.decoder.parameters()]
                # questo non serve piu quindi posso aggiornarlo
                optimizer_decoder.step()
                # pulisco discriminaotor
                net.discriminator.zero_grad()
            # propago loss discriminator
            if train_dis:
                loss_discriminator.backward()
                #[p.grad.data.clamp_(-1,1) for p in net.discriminator.parameters()]
                optimizer_discriminator.step()


            # LOGGING
            progress.update(progress.value + 1, loss_nle=loss_nle_mean(nle_value.data.cpu().numpy()[0]),
                            loss_encoder=loss_encoder_mean(loss_encoder.data.cpu().numpy()[0]),
                            loss_decoder=loss_decoder_mean(loss_decoder.data.cpu().numpy()[0]),
                            loss_discriminator=loss_discriminator_mean(loss_discriminator.data.cpu().numpy()[0]),

                            epoch=i + 1)


        # FINE EPOCA
        lr_encoder.step()
        lr_decoder.step()
        lr_discriminator.step()
        progress.finish()
        #lr_sc.step()
        #check(net,loss_nle_mean.measure)

        writer.add_scalar('loss_encoder',loss_encoder_mean.measure,step_index)
        writer.add_scalar('loss_decoder',loss_decoder_mean.measure,step_index)
        writer.add_scalar('loss_discriminator',loss_discriminator_mean.measure,step_index)
        writer.add_scalar('loss_reconstruction', loss_nle_mean.measure, step_index)
        # writer.add_scalar('loss_KLD', loss_kl_mean.measure, step_index)
        # writer.add_scalar('loss_layer', loss_mse_mean.measure, step_index)
        # writer.add_scalar('loss_bce_gen', loss_bce_gen_mean.measure, step_index)
        # writer.add_scalar('loss_bce_dis', loss_bce_dis_mean.measure, step_index)


        for j, (data_batch, labels_batch) in enumerate(dataloader_test):

            data_in = Variable(data_batch, requires_grad=True).float().cuda()
            out = net(data_in)
            out = out[0].data.cpu()
            #riporto in 0-1
            out = (out+1)/2
            out = make_grid(out,nrow=4)
            writer.add_image("reconstructed", out, step_index)

            net.eval()
            out = net(None, 64)
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
