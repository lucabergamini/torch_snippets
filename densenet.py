import torch
import numpy
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F
from torch.autograd import Variable
import _pickle
from tensorboard import SummaryWriter
from torch.optim import SGD,Adam
import progressbar

class ConvLayer(nn.Module):
    def __init__(self,in_features,k):
        super(ConvLayer,self).__init__()
        #bn
        self.bn_0 = nn.BatchNorm2d(num_features=in_features)
        self.bn_1 = nn.BatchNorm2d(num_features=4*k)
        #conv
        self.conv_0 = nn.Conv2d(in_channels=in_features,out_channels=4*k,kernel_size=1,padding=0)
        self.conv_1 = nn.Conv2d(in_channels=4*k,out_channels=k,kernel_size=3,padding=1)

    def forward(self,ten):
        ten = self.bn_0(ten)
        ten = F.relu(ten)
        ten = self.conv_0(ten)
        ten = self.bn_1(ten)
        ten = F.relu(ten)
        ten = self.conv_1(ten)
        ten = F.dropout(ten,p=0.2,training=self.training)
        return ten


class DenseBlock(nn.Module):
    def __init__(self, in_features,num_layers,k):
        super(DenseBlock, self).__init__()
        #lista moduli
        self.ml = nn.ModuleList()
        for i in range(num_layers):
            self.ml.append(ConvLayer(in_features=in_features,k=k))
            #self.add_module("conv_{}".format(i),ConvLayer(in_features=in_features,k=k))
            in_features += k
        self.out_features = in_features

    def forward(self, ten):
        tens = [ten,]

        for m in self.ml:
            tens.append(m(torch.cat(tens,1)))


        return torch.cat(tens,1)

class TransitionLayer(nn.Module):
    def __init__(self, in_features):
        super(TransitionLayer, self).__init__()
        self.bn_0 = nn.BatchNorm2d(num_features=in_features)
        self.conv_0 = nn.Conv2d(in_channels=in_features,out_channels=in_features,kernel_size=1)

    def forward(self, ten):
        ten = self.bn_0(ten)
        ten = F.relu(ten)
        ten = self.conv_0(ten)
        ten = F.avg_pool2d(input=ten,kernel_size=2,stride=2)
        return ten

class ClassificationLayer(nn.Module):
    def __init__(self, in_features,num_classes):
        super(ClassificationLayer, self).__init__()
        self.bn_0 = nn.BatchNorm2d(num_features=in_features)
        self.fc_0 = nn.Linear(in_features=in_features,out_features=num_classes)
    def forward(self, ten):
        ten = F.relu(ten)
        ten = self.bn_0(ten)
        ten = F.avg_pool2d(ten,kernel_size=ten.size()[-2:])
        ten = self.fc_0(ten.view(len(ten),-1))
        ten = F.log_softmax(ten)
        return ten


class DenseNet(nn.Module):

    def __init__(self,k,in_features,layers=(6,12,36,24)):
        """
        
        :param k: filtri convolutivi 
        :param layers: quanti blocchi mettere e quanti filtri in ognuno
        """
        super(DenseNet, self).__init__()
        #convoluzione iniziale
        self.conv_0 = nn.Conv2d(in_channels=in_features,out_channels=2*k,kernel_size=3,padding=1)
        #self.conv_0 = nn.Conv2d(in_channels=in_features,out_channels=2*k,kernel_size=7,stride=2,padding=3)
        in_features = 2*k
        #aggiungo moduli
        self.blocks = nn.ModuleList()
        for i,j in enumerate(layers):
            dense = DenseBlock(in_features=in_features,num_layers=j,k=k)
            in_features = dense.out_features
            if i == len(layers)-1:
                tran = ClassificationLayer(in_features=in_features,num_classes=1000)
            else:
                tran = TransitionLayer(in_features=in_features)
            self.blocks.append(nn.Sequential(dense,tran))
        for module in self.modules():
            if isinstance(module,nn.Conv2d):

                module.bias.data.zero_()
                module.weight.data = torch.randn(module.weight.size())* (2.0/(module.kernel_size[0]*module.kernel_size[1]*module.out_channels))**0.5
            elif isinstance(module,nn.Linear):
                module.bias.data.zero_()

    def forward(self,ten):
        #preprocessing
        ten = self.conv_0(ten)
        #ten = F.max_pool2d(ten,kernel_size=2,stride=2)
        for block in self.blocks:
            ten = block(ten)
        return ten


class CIFAR(Dataset):
    def __init__(self):
        dict = _pickle.load(open("data/cifar100"))
        data = dict["data"].astype("float32")
        #ogni immagine e una riga
        #i canali sono messi uno in coda all altro
        #li normalizzo
        channels = []
        for i in range(3):
            channel = data[:,1024*i:1024*(i+1)]
            channel = (channel - numpy.mean(channel)) / numpy.std(channel)
            channel = channel[:, numpy.newaxis]
            channels.append(channel)

        self.data = numpy.concatenate(channels,1).reshape((len(data),3,32,32))
        self.labels = dict["fine_labels"]

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        return (self.data[index],self.labels[index])



def classification_accuracy(out,labels):
    #mi servono argmax
    _,out = torch.max(out,1)
    accuracy = torch.sum(out==labels).float()
    accuracy/= len(out)
    return accuracy




writer = SummaryWriter("runs")
net = DenseNet(in_features=3,k=12,layers=[32,32,32]).cuda()
#DATASET
dataset = CIFAR()
loader = DataLoader(dataset,batch_size=32,shuffle=True)
#OPTIM-LOSS
#optimizer = Adam(params=net.parameters(),lr=0.01,weight_decay=10e-4)
optimizer = SGD(params=net.parameters(),lr=0.1,momentum=0.9,weight_decay=10e-4,nesterov=True)
loss = nn.NLLLoss()
#IL GRAFO NON SI RIESCE A FARE
#writer.add_graph(net,net(Variable(torch.rand(1,3,32,32), requires_grad=True).cuda()))

batch_number = len(loader)
num_epochs = 300
logging_step = 50
#logging_image_step = 100
widgets = [

    'Batch: ', progressbar.Counter(),
        '/', progressbar.FormatCustomText('%(total)s',{"total":batch_number}),
        ' ', progressbar.Bar(marker="-", left='[', right=']'),
        ' ', progressbar.ETA(),
        ' ',
         progressbar.DynamicMessage('loss'),
        ' ',
        progressbar.DynamicMessage("accuracy"),' ',
    progressbar.DynamicMessage("epoch")
    ]

for i in range(num_epochs):
    progress = progressbar.ProgressBar(min_value=0, max_value=batch_number, initial_value=0,widgets=widgets).start()

    # sistemo lr dividendo per 10 a 1/2 e 2/3
    if i % 5 == 0:
        optimizer.param_groups[0]["lr"] *= 0.93
        #0.93 ogni 5
    for j,(data_batch,labels_batch) in enumerate(loader):
        net.train()
        #trasformo in variabili
        data_batch = Variable(data_batch, requires_grad=True).float().cuda()
        labels_batch = Variable(labels_batch).long().cuda()

        #azzero gradiente
        net.zero_grad()
        optimizer.zero_grad()

        #calcolo uscita
        out = net(data_batch)
        #loss
        loss_value = loss(out, labels_batch)
        #accuracy
        accuracy_value = classification_accuracy(out,labels_batch)
        # propago
        loss_value.backward()
        # adesso ho accesso al gradiente e posso aggiornare anche i pesi
        optimizer.step()
        #LOGGING
        progress.update(progress.value+1,loss=loss_value.data.cpu().numpy()[0],accuracy=accuracy_value.data.cpu().numpy()[0],epoch=i+1)

        if j % logging_step == 0:
            #LOSS ACCURACY
            writer.add_scalar('loss', loss_value.data[0], i*batch_number+j)
            writer.add_scalar('accuracy', accuracy_value.data[0], i*batch_number+j)
            #PARAMS
            #for name, param in net.named_parameters():
            #   writer.add_histogram(name, param.clone().cpu().data.numpy(), i*batch_number+j)

    progress.finish()



