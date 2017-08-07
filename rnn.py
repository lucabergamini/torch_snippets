import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from tensorboard import SummaryWriter
from datetime import datetime
import numpy
from torch.optim import SGD, Adam
from torch.nn.init import xavier_normal
from torch.nn import Parameter
from torchvision.utils import make_grid
from torch.utils.data import DataLoader, TensorDataset, Dataset
import progressbar

def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
    # Returns
        A binary matrix representation of the input.
    """
    y = numpy.array(y, dtype='int').ravel()
    if not num_classes:
        num_classes = numpy.max(y) + 1
    n = y.shape[0]
    categorical = numpy.zeros((n, num_classes))
    categorical[numpy.arange(n), y] = 1
    return categorical


def text_to_numpy(text_name,seq_len=20):
    #apro file
    f = open(text_name,"r")
    #lego file
    text = f.read()
    #tolgo roba inutile
    text = text.replace("\n","")
    text = text.replace("\t","")

    i = 0
    #preparo spazio dati finali
    data = numpy.zeros((len(text)/seq_len,seq_len,91))
    while True:
        #provo a prendere seq_len caratteri
        token = text[i*seq_len:(i)*seq_len+seq_len]
        #se non riesco ho finito
        if len(token) < seq_len:
            break
        #sostituisco ogni lettera con ASCII-32 (spazio primo carattere utile)
        token = numpy.asarray([ ord(c) for c in token]) - 32
        #trasformo in one-hot
        token = to_categorical(token,num_classes=91)
        #aggiorno data
        data[i] = token
        i+=1
    #per sicurezza di non avere roba in piu tutta a 0
    data = data[:i]
    return data

class MyDataset(Dataset):
    def __init__(self,data,transform):
        self.data = data
        self.len = len(data)
        self.transform = transform
    def __len__(self):
        return self.len
    def __getitem__(self, item):
        sample = {"data":self.data[item],"label":0}
        return self.transform(sample)

class dataMultiplier(object):

    def __init__(self,seq_len=10,label_len = 1):
        self.seq_len = seq_len
        self.label_len = label_len

    def __call__(self, sample):
        #prende dato
        data_raw = sample["data"]
        i = 0
        data = numpy.zeros((len(data_raw) , self.seq_len, 91))
        labels = numpy.zeros((len(data_raw), self.label_len, 91))
        while True:
            #prende dato e label
            token = data_raw[i:i+self.seq_len]
            label = data_raw[i+self.seq_len:i+self.seq_len+self.label_len]
            #se non riesce ho finito
            if len(token) < self.seq_len or len(label) < self.label_len:
                break
            data[i] = token
            labels[i] = label
            i +=1
        data = data[:i]
        labels = labels[:i]
        return {"data":data,"label":labels}




class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.bn1 = nn.BatchNorm1d(num_features=91)
        self.lstm = nn.LSTM(input_size=91,hidden_size=256,num_layers=1,batch_first=True,dropout=0.2)
        self.bn2 = nn.BatchNorm1d(num_features=256)
        self.fc2 = nn.Linear(in_features=256,out_features=91)

    def forward(self,i):
        #devo girare i per bn
        #i = i.transpose(1,2).contiguous()
        #i = self.bn1(i)
        #i = i.transpose(1,2)
        i,_ = self.lstm(i)
        i = i[:,-1]
        #i = self.fc1(i)
        #i = F.relu(i)
        #i = F.dropout(i,0.35)
        i = self.fc2(i)

        i = F.log_softmax(i)
        return i

def classification_accuracy(out, labels):
    # mi servono argmax
    _, out = torch.max(out, 1)
    accuracy = torch.sum(out == labels).float()
    accuracy /= len(out)
    return accuracy


net = Net().cuda()

writer = SummaryWriter('runs/'+datetime.now().strftime('%B%d  %H:%M:%S'))

data = text_to_numpy("data/text_1",seq_len=51)
loader = DataLoader(MyDataset(data,transform=dataMultiplier(seq_len=50,label_len=1)),batch_size=128,shuffle=True)

# net = net.cuda()
optimizer = Adam(params=net.parameters(), lr=0.01)

# loss
loss = nn.NLLLoss()

batch_number = len(loader)
num_epochs = 200
logging_step = 50
logging_text_step = 50
widgets = [
    'Batch: ', progressbar.Counter(),
    '/', progressbar.FormatCustomText('%(total)s', {"total": batch_number}),
    ' ', progressbar.Bar(marker="-", left='[', right=']'),
    ' ', progressbar.ETA(),
    ' ', progressbar.DynamicMessage('loss'),
    ' ', progressbar.DynamicMessage("accuracy"),
]

for i in xrange(num_epochs):
    progress = progressbar.ProgressBar(min_value=0, max_value=batch_number, initial_value=0, widgets=widgets).start()

    for j,sample in enumerate(loader):
        optimizer.zero_grad()
        net.zero_grad()
        #reshape
        data_batch = sample["data"].view(-1,sample["data"].size()[2],sample["data"].size()[3]).float()
        labels_batch = sample["label"].view(-1,sample["label"].size()[3]).float()
        #tolgo one-hot
        labels_batch = torch.max(labels_batch,1)[1]
        #trasofrmo in variabili
        data_batch = Variable(data_batch.cuda(), requires_grad=True)
        labels_batch = Variable(labels_batch.long().cuda())
        # calcolo uscita
        out = net(data_batch)
        #calcolo loss
        loss_value= loss(out, labels_batch)
        # accuracy
        accuracy_value = classification_accuracy(out, labels_batch)
        # BACKPROP
        #optimizer.zero_grad()
        #net.zero_grad()
        loss_value.backward()
        optimizer.step()

        # LOGGING
        progress.update(progress.value + 1,
                        loss=loss_value.data.cpu().numpy()[0],
                        accuracy=accuracy_value.data.cpu().numpy()[0],
                        )

        if j % logging_step == 0:
            # LOSS ACCURACY
            writer.add_scalar('loss', loss_value.data[0], i * batch_number + j)
            writer.add_scalar('accuracy', accuracy_value.data[0], i * batch_number + j)
            # PARAMS
            for name, param in net.named_parameters():
                writer.add_histogram(name, param.clone().cpu().data.numpy(), i * batch_number + j)

        if j % logging_text_step == 0:
            # STEP
            s = "Non sopporto i giovani impertinenti che non cedono il posto ai vecchi in autobus"[0:50]
            s_final = s
            s = numpy.asarray([ord(c) for c in s]) - 32
            s = to_categorical(s, num_classes=91)
            for k in xrange(50):
                c = net(Variable(torch.FloatTensor(s[numpy.newaxis,...])).cuda()).cpu().data.numpy()
                c = numpy.where(c < numpy.max(c),0,1)

                s = numpy.append(s,c,0)
                s = s[1:]
                c = (numpy.argmax(c,1)+32)[0]
                s_final += str(unichr(c))

            writer.add_text("text_sample",s_final,i * batch_number + j)
    progress.finish()



