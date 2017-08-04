import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from tensorboard import SummaryWriter
from datetime import datetime
import numpy
from torch.optim import SGD
from torchvision import utils
from time import sleep

def to_categorical(y,num_classes):
    y = numpy.array(y, dtype='int').ravel()
    n = y.shape[0]

    categorical = numpy.zeros((n, num_classes),dtype="float32")
    categorical[numpy.arange(n), y] = 1
    return categorical


class Net(nn.Module):

    def __init__(self):
        super(Net,self).__init__()

        self.weight_1 = nn.Parameter(nn.init.xavier_normal(torch.Tensor(2, 2)), requires_grad=True)
        self.bias_1 = nn.Parameter(torch.ones(2),requires_grad=True)
        #self.weight_2 = nn.Parameter(nn.init.xavier_normal(torch.Tensor(2, 8)), requires_grad=True)
        #self.bias_2 = nn.Parameter(torch.ones(2),requires_grad=True)

        self.register_parameter("weight_1", self.weight_1)
        self.register_parameter("bias_1", self.bias_1)
        #self.register_parameter("weight_2", self.weight_2)
        #self.register_parameter("bias_2", self.bias_2)

    def forward(self,x):
        #x e input

        x = F.linear(x,weight=self.weight_1,bias=self.bias_1)
        #x = F.relu(x)

        #x = F.linear(x,weight=self.weight_2,bias=self.bias_2)

        x = F.log_softmax(x)

        return x


writer = SummaryWriter('data/runs/'+datetime.now().strftime('%B%d  %H:%M:%S'))

net = Net()
optimizer = SGD(params=net.parameters(),lr=0.5,momentum=0.85)
#definisco input da numpy
#data = numpy.random.randint(-2,2,120)

data = numpy.random.randint(0,2,(120,2))
labels = data[:,0:1] | data[:,1:]
#labels = to_categorical(labels,num_classes=2)

#converto a variabili
data = Variable(torch.FloatTensor(data),requires_grad=True)
labels = Variable(torch.LongTensor(labels)).squeeze()

loss = nn.NLLLoss()

writer.add_graph(net, net(Variable(torch.rand(120,2), requires_grad=True)))

for i in xrange(120):
    #calcolo uscita
    out = net(data)

    #azzero gradiente
    net.zero_grad()
    optimizer.zero_grad()
    #loss
    loss_value = loss(out,labels)
    writer.add_scalar('loss_NLL', loss_value.data[0], i)
    #propago
    loss_value.backward()
    #adesso ho accesso al gradiente e posso aggiornare anche i pesi
    optimizer.step()

    for name, param in net.named_parameters():
        writer.add_histogram(name, param.clone().cpu().data.numpy(), i)




