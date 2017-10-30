import torch
import numpy
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F
from torch.autograd import Variable
import cPickle
from tensorboard import SummaryWriter
from torch.optim import SGD, Adam,RMSprop
import progressbar
from generator.camvid import CamVid
from torchvision.utils import make_grid
import cv2

from utils.metrics import classification_accuracy,RollingMeasure
from utils.callbacks import ModelCheckpoint
from model.tiramisu import Tiramisu




if __name__ == "__main__":


    writer = SummaryWriter()
    #net = testDense().cuda()

    net = Tiramisu(in_features=3,num_classes=12).cuda()
    print "net done"
    # DATASET
    #train_data = numpy.load('../data/train_data.npy').reshape(-1,3,224,224).astype("float")

    #train_labels = numpy.load('../data/train_label.npy').reshape((367, 224, 224)).astype("int")

    #loader = DataLoader(
    #    TensorDataset(data_tensor=torch.FloatTensor(train_data), target_tensor=torch.LongTensor(train_labels)),
    #    batch_size=3, shuffle=True)

    dataset = CamVid("/home/lapis-ml/Desktop/camvid/train/data/", "/home/lapis-ml/Desktop/camvid/train/labels/",
                     "/home/lapis-ml/Desktop/camvid/labels")

    loader = DataLoader(dataset,batch_size=3,shuffle=True)

    dataset = CamVid("/home/lapis-ml/Desktop/camvid/test/data/", "/home/lapis-ml/Desktop/camvid/test/labels/",
                     "/home/lapis-ml/Desktop/camvid/labels",data_aug=False,reshape=True)

    loader_test = DataLoader(dataset,batch_size=3,shuffle=False)
    #
    #test_labels = numpy.load('../data/test_label.npy').reshape((233, 224, 224)).astype("int")
    #test_data = numpy.load('../data/test_data.npy').reshape(-1,3,224,224).astype("float")

    #loader_test = DataLoader(
    #    TensorDataset(data_tensor=torch.FloatTensor(test_data), target_tensor=torch.LongTensor(test_labels)), batch_size=3,
    #    shuffle=False)

    # OPTIM-LOSS
    optimizer = RMSprop(params=net.parameters(), lr=0.001,weight_decay=10e-04)
    # optimizer = SGD(params=net.parameters(),lr=0.1,momentum=0.9,weight_decay=10e-4,nesterov=True)
    loss = nn.NLLLoss2d(weight=dataset.class_weight).cuda()
    #modelcheck
    check = ModelCheckpoint()
    batch_number = len(loader)
    num_epochs = 750
    logging_step = 120
    logging_image_step = 50
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
        # sistemo lr dividendo per 10 a 1/2 e 2/3
        if i >0 :
            optimizer.param_groups[0]["lr"] *= 0.995
        # 0.93 ogni 5
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
            optimizer.step()
            # LOGGING

            progress.update(progress.value + 1, loss=loss_mean(loss_value.data.cpu().numpy()[0]),
                            accuracy=accuracy_mean(accuracy_value), epoch=i + 1)

        # FINE EPOCA
        progress.finish()
        #check di salvataggio
        check(net,loss_mean(loss_value.data.cpu().numpy()[0]))
        #log
        # LOSS ACCURACY
        writer.add_scalar('loss', loss_value.data[0], step_index)
        writer.add_scalar('accuracy', accuracy_value, step_index)
        # TEST
        net.eval()
        # prendo solo il primo
        for k, (data_batch, labels_batch) in enumerate(loader_test):
            if k == 0:
                net.eval()
                #
                data_batch = Variable(data_batch, requires_grad=False, volatile=True).float().cuda()
                labels_batch = Variable(labels_batch).long().cuda()

                # calcolo uscita
                out = net(data_batch)
                _, out = torch.max(out, 1)
                out = torch.unsqueeze(out, 1)
                out = out.data.float().cpu().numpy()

                # OUT
                out_g = out / 12.0 * 255.0
                # OUT COLOR
                out_c = numpy.zeros((len(out), 3, 224, 224))
                # replico 3 volte in out
                out_temp = numpy.repeat(out, 3, axis=1)

                for index_color, color in enumerate(dataset.labels):
                    out_c = numpy.where(out_temp == index_color, numpy.asarray(color).reshape(3, 1, 1), out_c)

                # LABELS
                labels_batch = torch.unsqueeze(labels_batch, 1)
                labels_batch = labels_batch.data.float().cpu().numpy()
                labels_batch = labels_batch / 12.0 * 255.0

                grid = make_grid(torch.FloatTensor(out_c), nrow=3)
                writer.add_image("out_test_color", grid, step_index)

                grid = make_grid(torch.FloatTensor(out_g), nrow=3)
                writer.add_image("out_test_gray", grid, step_index)
                grid = make_grid(torch.FloatTensor(labels_batch), nrow=3)
                writer.add_image("label_test_gray", grid, step_index)

        step_index += 1

