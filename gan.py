import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from tensorboard import SummaryWriter
from datetime import datetime
import numpy
from torch.optim import SGD
from torch.nn.init import xavier_normal
from torchvision import utils
from torch.nn import Parameter
from torchvision.utils import make_grid
from torch.utils.data import DataLoader,TensorDataset
import cPickle
import progressbar


class Conv_block(nn.Module):

    def __init__(self,in_ch,out_ch,k_size,stride=1,pool=3):
        super(Conv_block,self).__init__()

        self.conv_0 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=k_size, stride=stride)
        self.pool = nn.MaxPool2d(pool)
        # self.red = nn.Sequential(*[
        #     nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=k_size, stride=stride),
        #     nn.ReLU(),
        #     nn.MaxPool2d(pool),
        # ])

    def forward(self,x):

        x = self.conv_0(x)
        x = F.relu(x)
        x = self.pool(x)
        return x

class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()

        self.conv_0 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=2, stride=1)
        self.conv_1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=2, stride=1)
        self.conv_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=1,padding=1)
        self.conv_3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=1,padding=1)
        self.conv_4 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=2, stride=1,padding=0)

    def forward(self,i):
        #down
        i = self.conv_0(i)
        i = F.relu(i)
        i = F.max_pool2d(i,2)
        #down
        i = self.conv_1(i)
        i = F.relu(i)
        i = F.max_pool2d(i, 2)
        #up
        i = self.conv_2(i)
        i = F.relu(i)
        i = F.upsample(i, scale_factor=2)
        #up
        i = self.conv_3(i)
        i = F.relu(i)
        #i = F.upsample(i, scale_factor=2)
        #up
        i = self.conv_4(i)
        i = F.relu(i)
        i = F.upsample(i, scale_factor=2)

        return i


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()

        self.add_module("conv_0",Conv_block(1,20,2))
        self.add_module("conv_1",Conv_block(20,32,2))

        self.fc1 = nn.Linear(in_features=128,out_features=64)
        self.fc2 = nn.Linear(in_features=64,out_features=32)
        self.fc3 = nn.Linear(in_features=32,out_features=2)

    def forward(self,i):
        i = self.conv_0(i)
        i = self.conv_1(i)

        i = i.view(len(i),-1)
        i = self.fc1(i)
        i = F.relu(i)
        i = self.fc2(i)
        i = F.relu(i)
        i = self.fc3(i)
        i = F.log_softmax(i)
        return i





class Net(nn.Module):

    def __init__(self):
        super(Net,self).__init__()
        #layers
        self.add_module("generator",Generator())
        self.add_module("discriminator",Discriminator())

        self.imgs = {}
        #setto pesi conv
        for m in self.modules():
            #qui vedo tutto
            if isinstance(m,nn.Conv2d):
                #setto weight
                m.weight = Parameter(xavier_normal(m.weight.data))
            if isinstance(m,nn.Linear):
                # setto weight
                m.weight = Parameter(xavier_normal(m.weight.data))

        #train mode 1D 0G
        self.train_discrimator = True
        #batch_origin 1T 0F
        self.batch_real = True

    def forward(self,i):
        #i rappresenta input
        if self.batch_real:
            #immagini vere passano solo nel discriminatore
            i = self.discriminator(i)
        else:
            i = self.generator(i)
            self.imgs["gen"] = i.data
            i = self.discriminator(i)
        #print list(self.generator.parameters())[0][0]
        return i


def classification_accuracy(out,labels):
    #mi servono argmax
    _,out = torch.max(out,1)
    accuracy = torch.sum(out==labels).float()
    accuracy/= len(out)
    return accuracy

writer = SummaryWriter('runs/'+datetime.now().strftime('%B%d  %H:%M:%S'))
net = Net()
#net = net.cuda()
optimizer_discriminator = SGD(params=net.discriminator.parameters(),lr=0.01,momentum=0.85)
optimizer_generator = SGD(params=net.generator.parameters(),lr=0.005,momentum=0.85)
#definisco input da numpy
train_raw,_,test_raw = cPickle.load(open("data/mnist.pkl"))
data_train = train_raw[0].reshape(-1,1,28,28)
labels_train = train_raw[1]


#converto a variabili

#loader
loader = DataLoader(TensorDataset(torch.FloatTensor(data_train[0:]),torch.LongTensor(labels_train[0:])),batch_size=16)

#loss
loss = nn.NLLLoss()
#due grafi?
net.batch_real = True
writer.add_graph(net, net(Variable(torch.FloatTensor(data_train[0:1]), requires_grad=True)))
net.batch_real = False
writer.add_graph(net, net(Variable(torch.FloatTensor(data_train[0:1]), requires_grad=True)))

batch_number = len(loader)
num_epochs = 1
logging_step = 50
logging_image_step = 25
widgets = [
        'Batch: ', progressbar.Counter(),
        '/', progressbar.FormatCustomText('%(total)s',{"total":batch_number}),
        ' ', progressbar.Bar(marker="-", left='[', right=']'),
        ' ', progressbar.ETA(),
        ' ', progressbar.DynamicMessage('loss_discriminator'),
        ' ', progressbar.DynamicMessage('loss_generator'),
        ' ', progressbar.DynamicMessage("accuracy_discriminator"),
        ' ', progressbar.DynamicMessage('accuracy_generator'),
    ]
for i in xrange(num_epochs):
    progress = progressbar.ProgressBar(min_value=0, max_value=batch_number, initial_value=0,widgets=widgets).start()

    for j,(data_batch,labels_batch) in enumerate(loader):
        #REAL
        net.batch_real = True
        #trasformo in variabili
        data_batch = Variable(data_batch, requires_grad=True)
        #calcolo uscita
        out_real = net(data_batch)
        #FAKE
        net.batch_real = False
        # genero rumore
        data_batch = Variable(torch.rand(data_batch.size()), requires_grad=True)
        # calcolo uscita
        out_fake = net(data_batch)
        #concateno
        out_cat = torch.cat((out_real,out_fake),0)
        labels_cat = torch.cat((Variable(torch.ones(labels_batch.size())),Variable(torch.zeros(labels_batch.size()))),0).long()
        labels_cat_reverse = labels_cat.clone()
        labels_cat_reverse[labels_cat == 0] = 1
        labels_cat_reverse[labels_cat == 1] = 0
        #LOSS
        loss_discriminator = loss(out_cat, labels_cat)
        loss_generator = loss(out_cat, labels_cat_reverse)
        #accuracy
        accuracy_discriminator = classification_accuracy(out_cat, labels_cat)
        accuracy_generator = classification_accuracy(out_cat, labels_cat_reverse)
        #BACKPROP
        optimizer_discriminator.zero_grad()
        optimizer_generator.zero_grad()
        #discriminator
        if j % 5 ==0:
            net.zero_grad()
            loss_discriminator.backward(retain_graph=True)
            optimizer_discriminator.step()
        # generator
        net.zero_grad()
        loss_generator.backward()
        optimizer_generator.step()

        #LOGGING
        progress.update(progress.value + 1,
                        loss_discriminator=loss_discriminator.data.cpu().numpy()[0],
                        loss_generator=loss_generator.data.cpu().numpy()[0],
                        accuracy_discriminator=accuracy_discriminator.data.cpu().numpy()[0],
                        accuracy_generator=accuracy_generator.data.cpu().numpy()[0]
                        )

        if j % logging_step == 0:
            #LOSS ACCURACY
            writer.add_scalar('loss_discriminator', loss_discriminator.data[0], i*batch_number+j)
            writer.add_scalar('loss_generator', loss_generator.data[0], i*batch_number+j)
            writer.add_scalar('accuracy_discriminator', accuracy_discriminator.data[0], i * batch_number + j)
            writer.add_scalar('accuracy_generator', accuracy_generator.data[0], i * batch_number + j)
            #PARAMS
            for name, param in net.named_parameters():
                writer.add_histogram(name, param.clone().cpu().data.numpy(), i*batch_number+j)
            #IMGS
        if j % logging_image_step == 0:
            for (name,imgs) in net.imgs.iteritems():
                imgs = imgs.view(imgs.size()[0]*imgs.size()[1],1,imgs.size()[2],imgs.size()[3]).cpu()
                grid = make_grid(imgs,nrow=10)
                writer.add_image(name,grid,i*batch_number+j)
    progress.finish()


