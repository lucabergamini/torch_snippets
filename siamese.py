import torch
import numpy
from datetime import datetime
from tensorboard import SummaryWriter
import progressbar
import cPickle

import torch.nn.functional as F
from torch.nn import Module
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader
from torchvision.utils import make_grid
from torch.optim import Adam
from matplotlib import pyplot
from sklearn.decomposition import PCA


class MNISTSiameseDataset(Dataset):
    def __init__(self,path_name,classes,num_elements_total,num_elements,epoch_len=100):
        """
        dataset siamese per MNIST, in ogni batch devo avere un tot di esempi di ogni classe
        per farlo ogni item del dataset e un intera classe
        :param path_name: path al dataset
        :param classes: classi da includere
        :param num_elements_total: numero elementi totali in ogni classe
        :param num_elements: quanti campioni da prendere per ogni classe nel batch
        """
        self.len = len(classes)
        self.epoch_len=epoch_len
        train_raw, _, test_raw = cPickle.load(open(path_name))
        self.data_raw = train_raw[0].reshape(-1, 1, 28, 28)
        self.labels_raw = train_raw[1]
        self.data = numpy.zeros((self.len,num_elements_total,1,28,28),dtype="float")
        self.num_elements = num_elements
        self.labels = numpy.asarray(classes)

        for num_class,i in enumerate(classes):
            #prendo num_el_total dalla classe i, presi a caso tra quelli della classe
            data_class_i = self.data_raw[self.labels_raw == i]
            self.data[num_class] = data_class_i[numpy.random.permutation(len(data_class_i))][0:num_elements_total]



    def __len__(self):
        return self.epoch_len

    def __getitem__(self, item):
        #item [0-epoch-len]
        item = item % self.len
        #devo prendere da data
        data = self.data[item]
        data = data[numpy.random.permutation(len(data))][0:self.num_elements]
        label = numpy.full(len(data),self.labels[item])
        sample = {"data":data,"label":label}
        return sample


class Net(Module):
    def __init__(self):
        super(Net,self).__init__()
        self.bn1 = torch.nn.BatchNorm2d(num_features=1)
        self.conv1 = torch.nn.Conv2d(in_channels=1,out_channels=64,kernel_size=2)
        self.bn2 = torch.nn.BatchNorm2d(num_features=64)
        self.conv2 = torch.nn.Conv2d(in_channels=64,out_channels=128,kernel_size=2)
        self.bn3 = torch.nn.BatchNorm2d(num_features=128)
        self.conv3 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2)
        self.bn4 = torch.nn.BatchNorm1d(num_features=512)
        self.do1 = torch.nn.Dropout(p=0.5)
        self.fc1 = torch.nn.Linear(in_features=512,out_features=256)
        self.bn5 = torch.nn.BatchNorm1d(num_features=256)
        self.fc2 = torch.nn.Linear(in_features=256,out_features=128)
        self.bn6 = torch.nn.BatchNorm1d(num_features=128)
        self.fc3 = torch.nn.Linear(in_features=128,out_features=3)


    def forward(self, i):
        i = self.bn1(i)
        i = self.conv1(i)
        i = F.relu(i)
        i = F.max_pool2d(i,kernel_size=2)

        i = self.bn2(i)
        i = self.conv2(i)
        i = F.relu(i)
        i = F.max_pool2d(i,kernel_size=2)

        i = self.bn3(i)
        i = self.conv3(i)
        i = F.relu(i)
        i = F.max_pool2d(i, kernel_size=2)
        i = i.view(len(i),-1)

        i = self.bn4(i)
        i = self.fc1(i)
        #i = self.do1(i)
        i = F.relu(i)

        i = self.bn5(i)
        i = self.fc2(i)
        i = F.relu(i)

        i = self.bn6(i)
        i = self.fc3(i)
        i = F.tanh(i)

        return i


class N_pairLoss(Module):
    """
    calcola una loss che dovrebbe portare vicini elementi della stessa classe usando il prodotto scalare
    """

    def forward(self,out,labels):
        """
        :param out: vettore uscita Bx128
        :param labels: vettore labels B
        :return:
        """
        value = 0.0
        for embedding,label in zip(out,labels):
            #prendo tutti quelli della classe
            embedding_same_class = out[torch.nonzero(labels.data == label.data),:]
            embedding_same_class = torch.squeeze(embedding_same_class,1)
            #calcolo exp prodotti scalari
            exp_same = torch.exp(torch.sum(embedding_same_class*embedding,1))
            exp_all = torch.exp(torch.sum(out * embedding, 1))
            value -= torch.log(torch.sum(exp_same/torch.sum(exp_all)))
        value /=len(out)

        return value

#shuffle false qui e fondamentale
#proviamo a cambiare altro
loader = DataLoader(MNISTSiameseDataset("data/mnist.pkl",classes=[i for i in xrange(10)],num_elements_total=500,num_elements=8,epoch_len=1000),batch_size=5,shuffle=True)
loader_test = DataLoader(MNISTSiameseDataset("data/mnist.pkl",classes=[i for i in xrange(10)],num_elements_total=500,num_elements=16,epoch_len=10),batch_size=10,shuffle=False)
net = Net().cuda()
loss = N_pairLoss()
optimizer = Adam(params=net.parameters(),lr=0.00025)
#paraetri
batch_number = len(loader)
num_epochs = 100
logging_step = 50
logging_image_step = 25
widgets = [
    progressbar.DynamicMessage("epoch"), ' ',
    'Batch: ', progressbar.Counter(),
    '/', progressbar.FormatCustomText('%(total)s', {"total": batch_number}),
    ' ', progressbar.Bar(marker="-", left='[', right=']'),
    ' ', progressbar.ETA(),
    ' ', progressbar.DynamicMessage('loss'),
]
#logger
writer_name = datetime.now().strftime('%B%d  %H:%M:%S')
writer = SummaryWriter('runs/{}'.format(writer_name))
for i in xrange(num_epochs):
    progress = progressbar.ProgressBar(min_value=0, max_value=batch_number, initial_value=0, widgets=widgets).start()

    for j,sample in enumerate(loader):
        net.train()
        # azzero gradiente
        net.zero_grad()
        optimizer.zero_grad()
        #carico batch
        data_batch = sample["data"].float()
        data_batch = data_batch.view((data_batch.size()[0]*data_batch.size()[1],)+data_batch.size()[2:])
        label_batch = sample["label"]
        label_batch = label_batch.view(label_batch.numel())

        #calcolo uscita
        out = net(Variable(data_batch).cuda())
        #loss back
        loss_value = loss(out,Variable(label_batch,requires_grad=True).cuda())
        loss_value.backward()
        optimizer.step()
        # LOGGING
        progress.update(progress.value + 1,
                        loss=loss_value.data.cpu().numpy()[0],
                        epoch=i+1
                        )
        if j %logging_image_step == 0:
            net.eval()
            #drawn dal test
            sample= loader_test.__iter__().next()
            data_batch = sample["data"].float()
            data_batch = data_batch.view((data_batch.size()[0] * data_batch.size()[1],) + data_batch.size()[2:])
            label_batch = sample["label"]
            label_batch = label_batch.view(label_batch.numel())

            out = net(Variable(data_batch).cuda())
            p = PCA(n_components=3)
            out_pca = p.fit_transform(out.data.cpu().numpy())

            writer.add_embedding(mat=torch.FloatTensor(out_pca),metadata=label_batch,label_img=data_batch,global_step=(i*batch_number)+j)
        # imgs = sample["data"]
        # imgs = imgs.view(imgs.size()[0] * imgs.size()[1], 1, imgs.size()[3], imgs.size()[4]).cpu()
        # grid = make_grid(imgs, nrow=5)
        # grid = grid.permute(1,2,0)
        # pyplot.imshow(grid.numpy())
        # pyplot.show()
    progress.finish()
