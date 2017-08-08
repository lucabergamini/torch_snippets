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

CHARS = ['\n', ' ', '!', '"', "'", '(', ')', '*', ',', '-', '.', ':', ';', '?', '[', ']', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
FILL_CHAR = ' '
features_size = len(CHARS)

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



class MyDataset(Dataset):
    def __init__(self,text_path,input_len,output_len):
        # lego file
        self.text = open(text_path, "r").read()
        # tolgo roba inutile
        self.text = self.text.replace("\n", " ")
        self.text = self.text.replace("\t", " ")
        self.text = self.text.lower()

        self.input_len = input_len
        self.output_len = output_len
        #TODO check this
        self.len = len(self.text) - (self.input_len+self.output_len)
    def __len__(self):
        return self.len
    def __getitem__(self, item):
        # provo a prendere seq_len caratteri
        token = self.text[item :item  + self.input_len+self.output_len]
        assert len(token) == self.input_len + self.output_len
        # sostituisco ogni lettera con codice
        token = numpy.asarray([CHARS.index(c) if c in CHARS else CHARS.index(FILL_CHAR) for c in token])
        # trasformo in one-hot
        token = to_categorical(token, num_classes=features_size)

        sample = {"data":token[:self.input_len],"label":token[-self.output_len:]}
        return sample



class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.bn1 = nn.BatchNorm1d(num_features=features_size)
        self.lstm = nn.LSTM(input_size=features_size,hidden_size=256,num_layers=1,batch_first=True,dropout=0.2)
        self.bn2 = nn.BatchNorm1d(num_features=256)
        self.fc2 = nn.Linear(in_features=256,out_features=features_size)

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


net = Net()

writer = SummaryWriter('runs/'+datetime.now().strftime('%B%d  %H:%M:%S'))

loader = DataLoader(MyDataset("data/text_1",input_len=50,output_len=1),batch_size=64,shuffle=True)

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
        data_batch = sample["data"].float()
        labels_batch = torch.squeeze(sample["label"])
        #tolgo one-hot
        labels_batch = torch.max(labels_batch,1)[1]
        #trasofrmo in variabili
        data_batch = Variable(data_batch, requires_grad=True)
        labels_batch = Variable(labels_batch.long())
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
            s = "the cry was so horrible in its agony that the frig"[0:50]
            s_final = s
            s = numpy.asarray([CHARS.index(c) if c in CHARS else CHARS.index(FILL_CHAR) for c in s])
            s = to_categorical(s, num_classes=features_size)
            for k in xrange(50):
                c = net(Variable(torch.FloatTensor(s[numpy.newaxis,...]))).cpu().data.numpy()
                c = numpy.where(c < numpy.max(c),0,1)
                s = numpy.append(s,c,0)
                s = s[1:]
                c = numpy.argmax(c,1)[0]
                s_final += CHARS[c]

            writer.add_text("text_sample",s_final,i * batch_number + j)
    progress.finish()



