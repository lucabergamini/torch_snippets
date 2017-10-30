
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
from model.densenet import DenseNet
from generator.imagenet import Imagenet
from utils.metrics import classification_accuracy


writer = SummaryWriter()
#net = testDense().cuda()

net = DenseNet(in_features=3,k=32,layers=[6,12,24,16],num_classes=1000).cuda()
print "net done"
#DATASET
dataset = Imagenet("/home/lapis-ml/Desktop/imagenet/train_224/")
loader = DataLoader(dataset,batch_size=64,shuffle=True)
#OPTIM-LOSS
optimizer = Adam(params=net.parameters(),lr=0.01,weight_decay=10e-4)
#optimizer = SGD(params=net.parameters(),lr=0.1,momentum=0.9,weight_decay=10e-4,nesterov=True)
loss = nn.NLLLoss()
#IL GRAFO NON SI RIESCE A FARE
#writer.add_graph(net,net(Variable(torch.rand(1,3,32,32), requires_grad=True).cuda()))

batch_number = len(loader)
num_epochs = 300
logging_step = 100
#logging_image_step = 100
step = 0
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

for i in xrange(num_epochs):
    progress = progressbar.ProgressBar(min_value=0, max_value=batch_number, initial_value=0,widgets=widgets).start()

    # sistemo lr dividendo per 10 a 1/2 e 2/3
    if i  >  0:
        optimizer.param_groups[0]["lr"] *= 0.95
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
        #qui mi serve solo uscita finale
        #out = out[0]
        #loss
        loss_value = loss(out, labels_batch)
        #accuracy
        accuracy_value = classification_accuracy(out,labels_batch)
        # propago
        loss_value.backward()
        # adesso ho accesso al gradiente e posso aggiornare anche i pesi
        optimizer.step()
        #LOGGING
        progress.update(progress.value+1,loss=loss_value.data.cpu().numpy()[0],accuracy=accuracy_value,epoch=i+1)

        if j % logging_step == 0:
            #LOSS ACCURACY
            writer.add_scalar('loss', loss_value.data[0], step)
            writer.add_scalar('accuracy', accuracy_value, step)
            step += 1
            #PARAMS
            #for name, param in net.named_parameters():
            #   writer.add_histogram(name, param.clone().cpu().data.numpy(), i*batch_number+j)

    progress.finish()



