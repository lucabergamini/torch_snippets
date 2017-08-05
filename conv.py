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

class Net(nn.Module):

    def __init__(self):
        super(Net,self).__init__()
        #layers
        #rete convolutiva
        self.add_module("conv_0",Conv_block(1,20,2))
        self.add_module("conv_1",Conv_block(20,32,2))
        self.add_module("conv_r1",Conv_block(1,32,4,4))

        #rete fc
        self.fc1 = nn.Linear(in_features=256,out_features=128)
        self.fc2 = nn.Linear(in_features=128,out_features=10)

        self.imgs = {}
        #setto pesi conv
        for m in self.modules():
            #qui vedo tutto
            if isinstance(m,nn.Conv2d):
                #setto weight
                m.weight = Parameter(xavier_normal(m.weight.data))



    def forward(self,i):
        #i rappresenta input
        x = self.conv_0(i)
        self.imgs["conv_0"] = x.data
        x = self.conv_1(x)
        self.imgs["conv_1"] = x.data
        y = self.conv_r1(i)
        self.imgs["conv_r1"] = y.data

        x = torch.cat((y,x),1)
        x = x.view(len(x),-1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.log_softmax(x)
        return x


def classification_accuracy(out,labels):
    #mi servono argmax
    _,out = torch.max(out,1)
    accuracy = torch.sum(out==labels).float()
    accuracy/= len(out)
    return accuracy

writer = SummaryWriter('runs/'+datetime.now().strftime('%B%d  %H:%M:%S'))
net = Net()
#net = net.cuda()
optimizer = SGD(params=net.parameters(),lr=0.01,momentum=0.85)
#definisco input da numpy
train_raw,_,test_raw = cPickle.load(open("data/mnist.pkl"))
data_train = train_raw[0].reshape(-1,1,28,28)
labels_train = train_raw[1]


#converto a variabili

#loader
loader = DataLoader(TensorDataset(torch.FloatTensor(data_train[0:]),torch.LongTensor(labels_train[0:])),batch_size=32)

#loss
loss = nn.NLLLoss()
writer.add_graph(net, net(Variable(torch.FloatTensor(data_train[0:1]), requires_grad=True)))

batch_number = len(loader)
num_epochs = 10
logging_step = 50
logging_image_step = 100
widgets = [
        'Batch: ', progressbar.Counter(),
        '/', progressbar.FormatCustomText('%(total)s',{"total":batch_number}),
        ' ', progressbar.Bar(marker="-", left='[', right=']'),
        ' ', progressbar.ETA(),
        ' ',
         progressbar.DynamicMessage('loss'),
        ' ',
        progressbar.DynamicMessage("accuracy"),
    ]
for i in xrange(num_epochs):
    progress = progressbar.ProgressBar(min_value=0, max_value=batch_number, initial_value=0,widgets=widgets).start()

    for j,(data_batch,labels_batch) in enumerate(loader):

        #trasformo in variabili
        data_batch = Variable(data_batch, requires_grad=True)
        labels_batch = Variable(labels_batch)

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
        progress.update(progress.value+1,loss=loss_value.data.cpu().numpy()[0],accuracy=accuracy_value.data.cpu().numpy()[0])

        if j % logging_step == 0:
            #LOSS ACCURACY
            writer.add_scalar('loss', loss_value.data[0], i*batch_number+j)
            writer.add_scalar('accuracy', accuracy_value.data[0], i*batch_number+j)
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


