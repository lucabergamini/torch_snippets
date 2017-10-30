import torch
import numpy
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F
from torch.autograd import Variable
import cPickle
from tensorboard import SummaryWriter
from torch.optim import SGD, Adam,RMSprop
from torch.optim.lr_scheduler import StepLR,MultiStepLR
import progressbar
from generator.lapis import LAPIS
from torchvision.utils import make_grid
import cv2

from utils.metrics import classification_accuracy,RollingMeasure
from utils.callbacks import ModelCheckpoint
from model.tiramisu import Tiramisu
from collections import OrderedDict

def from_classes_to_color(batch_array):
    """
    
    :param batch_array: array di dimensione BxCxWxH 
    :return: 
    """

    batch_array = torch.squeeze(batch_array, 1).numpy()
    # LABEL COLOR
    label_c = numpy.zeros((len(out), 224, 224, 3))
    for index_color, (name, color) in enumerate(colors.items()):
        # devo metterlo in scala
        color = numpy.array(color[0:3]) * 255

        color = color.astype("int32")


        # trovo maschera
        mask = batch_array == index_color
        label_c[mask] = color
        # mi serve sapere il numero di elementi a True
        # out_c[mask] = numpy.repeat(color,numpy.sum(mask[:,0]))
    #metto channel first
    label_c = numpy.transpose(label_c, (0, 3, 1, 2))
    return label_c


LABELS = OrderedDict([
    ("bolt",(1.0,0.0,0.0,1)),
    ("cup",(1.0,0.501960754395,0.501960754395,1)),
    ("hex",(1.0,0.433333396912,0.0,1)),
    ("mouse",(1.0,0.717777848244,0.501960754395,1)),
    ("pen",(1.0,0.866666734219,0.0,1)),
    ("remote",(1.0,0.933594822884,0.501960754395,1)),
    ("scrissor",(0.699999928474,1.0,0.0,1)),
    ("washer",(0.850588202477,1.0,0.501960754395,1)),
    ("bottle",(0.266666531563,1.0,0.0,1)),
    ("fork",(0.634771168232,1.0,0.501960754395,1)),
    ("keys",(0.0,1.0,0.16666674614,1)),
    ("nails",(0.501960754395,1.0,0.584967374802,1)),
    ("plate",(0.0,1.0,0.600000143051,1)),
    ("screw",(0.501960754395,1.0,0.800784349442,1)),
    ("spoon",(0.0,0.966666460037,1.0,1)),
    ("wrench",(0.501960754395,0.983398616314,1.0,1)),
    ("cellphone",(0.0,0.533333063126,1.0,1)),
    ("hammer",(0.501960754395,0.767581582069,1.0,1)),
    ("knife",(0.0,0.0999999046326,1.0,1)),
    ("nut",(0.501960754395,0.551764667034,1.0,1)),
    ("pliers",(0.333333492279,0.0,1.0,1)),
    ("screwdriver",(0.667973935604,0.501960754395,1.0,1)),
    ("tootbrush",(0.766666889191,0.0,1.0,1)),
])

LABELS_BACK = OrderedDict([
    ("back", (0, 0, 0, 1)),
    ("table", (1, 1, 1, 1))
])

WEIGHTS = OrderedDict([
    ("back", 0.2),
    ("table", 0.002),
    ("bolt", 0.1),
    ("cup", 0.05),
    ("hex", 0.1),
    ("mouse", 0.05),
    ("pen", 0.1),
    ("remote", 0.025),
    ("scrissor", 0.025),
    ("washer", 0.05),
    ("bottle", 0.025),
    ("fork", 0.025),
    ("keys", 0.05),
    ("nails", 0.025),
    ("plate", 0.025),
    ("screw", 0.05),
    ("spoon", 0.025),
    ("wrench", 0.025),
    ("cellphone", 0.025),
    ("hammer", 0.025),
    ("knife", 0.025),
    ("nut", 0.1),
    ("pliers", 0.025),
    ("screwdriver", 0.025),
    ("tootbrush", 0.025),


])

WEIGHTS = [0.02,]*2+[0.96/23,]*23

colors = [(key,val) for key,val in LABELS_BACK.items()]
colors.extend([(key,val) for key,val in LABELS.items()])


colors = OrderedDict(colors)



if __name__ == "__main__":


    writer = SummaryWriter(comment="_LAPIS")

    net = Tiramisu(in_features=3,num_classes=25,layers=(4,5,7,10),bottleneck=15,compress=False).cuda()
    print "net done"
    # DATASET
    TEST = True
    dataset = LAPIS("/home/lapis-ml/Desktop/LAPIS_dataset/train_2/data/", "/home/lapis-ml/Desktop/LAPIS_dataset/train_2/labels/",data_aug=True,reshape=False)

    loader = DataLoader(dataset,batch_size=3,shuffle=True)

    dataset = LAPIS("/home/lapis-ml/Desktop/LAPIS_dataset/test_2/data/", "/home/lapis-ml/Desktop/LAPIS_dataset/test_2/labels/",data_aug=True,reshape=False)
    loader_test_plain = DataLoader(dataset,batch_size=3,shuffle=False)

    dataset = LAPIS("/home/lapis-ml/Desktop/LAPIS_dataset/test_2/data/", "/home/lapis-ml/Desktop/LAPIS_dataset/test_2/labels/",data_aug=False,reshape=True)
    loader_test_reshape = DataLoader(dataset,batch_size=3,shuffle=False)

    # OPTIM-LOSS
    #ottimo per ora con lr=0.0008 e step=1000 con gamma=0.5
    optimizer = Adam(params=net.parameters(), lr=0.008)
    #lr_schedul = StepLR(optimizer,step_size=1000,gamma=0.5)
    #questo andava bene su 300 immagini con lr 0.008 e 350 epoche
    #lr_schedul = MultiStepLR(optimizer,milestones=[150,200,250,300],gamma=0.5)
    lr_schedul = MultiStepLR(optimizer,milestones=[15,20,25,30,60],gamma=0.5)
    #lr_schedul = StepLR(optimizer,step_size=100,gamma=1)
    loss = nn.NLLLoss2d(weight=torch.FloatTensor(WEIGHTS)).cuda()
    #modelcheck
    check = ModelCheckpoint()
    batch_number = len(loader)

    #num_epochs = 1000
    num_epochs = 100
    step_index=0
    widgets = [

        'Batch: ', progressbar.Counter(),
        '/', progressbar.FormatCustomText('%(total)s', {"total": batch_number}),
        ' ', progressbar.Bar(marker="-", left='[', right=']'),
        ' ', progressbar.ETA(),
        ' ',
        progressbar.DynamicMessage('loss'),
        ' ',
        progressbar.DynamicMessage("accuracy"), ' ',
        progressbar.DynamicMessage("epoch")
    ]

    for i in xrange(num_epochs):
        progress = progressbar.ProgressBar(min_value=0, max_value=batch_number, initial_value=0, widgets=widgets).start()
        accuracy_mean = RollingMeasure()
        loss_mean = RollingMeasure()
        grad_max_mean = RollingMeasure()
        grad_min_mean = RollingMeasure()

        for j, (data_batch, labels_batch) in enumerate(loader):
            net.train()
            # trasformo in variabili
            data_batch = Variable(data_batch, requires_grad=True).float().cuda()

            labels_batch = Variable(labels_batch).long().cuda()

            # azzero gradiente
            net.zero_grad()
            optimizer.zero_grad()
            loss.zero_grad()
            # calcolo uscita
            out = net(data_batch)
            # qui mi serve solo uscita finale
            # out = out[0]
            # loss
            loss_value = loss(out, labels_batch)
            accuracy_value = classification_accuracy(out, labels_batch)
            # propago
            loss_value.backward()
            # adesso ho accesso al gradiente e posso aggiornare anche i pesi
            #grads = ([torch.max(p._grad.data) for p in net.parameters()])
            #grad_max_mean(max(grads))
            #grad_min_mean(min(grads))

            optimizer.step()
            lr_schedul.step()
            # LOGGING

            progress.update(progress.value + 1, loss=loss_mean(loss_value.data.cpu().numpy()[0]),
                            accuracy=accuracy_mean(accuracy_value), epoch=i + 1)

        # FINE EPOCA
        progress.finish()
        #check di salvataggio
        check(net,loss_mean(loss_value.data.cpu().numpy()[0]))
        #log
        # LOSS ACCURACY
        writer.add_scalar('loss', loss_mean.measure, step_index)
        writer.add_scalar('accuracy', accuracy_mean.measure, step_index)
        #writer.add_scalar("max_grad",grad_max_mean.measure,step_index)
        #writer.add_scalar("min_grad",grad_min_mean.measure,step_index)


        if TEST:
            # TEST 1
            label_test_color = []
            out_test_color = []
            in_test_color = []
            for k, (data_batch, labels_batch) in enumerate(loader_test_plain):
                net.eval()
                #
                data_batch = Variable(data_batch, requires_grad=False, volatile=True).float().cuda()
                # calcolo uscita
                out = net(data_batch)
                _, out = torch.max(out, 1)
                #LOG tensorboard
                label_test_color.append(from_classes_to_color(labels_batch))
                out_test_color.append(from_classes_to_color(out.data.cpu()))
                in_test_color.append(data_batch.data.cpu().numpy())

            grid = make_grid(torch.FloatTensor(numpy.concatenate(label_test_color)), nrow=5)
            writer.add_image("labels_plain_test_color", grid / 255.0, step_index)

            grid = make_grid(torch.FloatTensor(numpy.concatenate(out_test_color)), nrow=5)
            writer.add_image("out_plain_test_color", grid / 255.0, step_index)

            grid = make_grid(torch.FloatTensor(numpy.concatenate(in_test_color)), nrow=5)
            writer.add_image("in_plain_test_color", grid, step_index)

            # TEST 2
            label_test_color = []
            out_test_color = []
            in_test_color = []
            for k, (data_batch, labels_batch) in enumerate(loader_test_reshape):
                net.eval()
                #
                data_batch = Variable(data_batch, requires_grad=False, volatile=True).float().cuda()
                # calcolo uscita
                out = net(data_batch)
                _, out = torch.max(out, 1)
                #LOG tensorboard
                label_test_color.append(from_classes_to_color(labels_batch))
                out_test_color.append(from_classes_to_color(out.data.cpu()))
                in_test_color.append(data_batch.data.cpu().numpy())


            grid = make_grid(torch.FloatTensor(numpy.concatenate(label_test_color)), nrow=5)
            writer.add_image("labels_reshaped_test_color", grid / 255.0, step_index)

            grid = make_grid(torch.FloatTensor(numpy.concatenate(out_test_color)), nrow=5)
            writer.add_image("out_reshaped_test_color", grid / 255.0, step_index)

            grid = make_grid(torch.FloatTensor(numpy.concatenate(in_test_color)), nrow=5)
            writer.add_image("in_reshaped_test_color", grid, step_index)



        step_index += 1

