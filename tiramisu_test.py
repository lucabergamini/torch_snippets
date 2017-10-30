import numpy
import torch
from tensorboard import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from generator.camvid import CamVid
from model.tiramisu import Tiramisu

writer = SummaryWriter()
net = Tiramisu(in_features=3,num_classes=12).cuda()
net.load_state_dict(torch.load(open("test")))

print "net done"

dataset = CamVid("/home/lapis-ml/Desktop/camvid/train/data/", "/home/lapis-ml/Desktop/camvid/train/labels/",
                 "/home/lapis-ml/Desktop/camvid/labels")

loader = DataLoader(dataset,batch_size=1,shuffle=True)

dataset = CamVid("/home/lapis-ml/Desktop/camvid/test/data/", "/home/lapis-ml/Desktop/camvid/test/labels/",
                 "/home/lapis-ml/Desktop/camvid/labels",data_aug=False,reshape=True)

loader_test = DataLoader(dataset,batch_size=1,shuffle=False)

outg_v = []
outc_v = []
label_v = []
index_log = 0

for j, (data_batch, labels_batch) in enumerate(loader_test):

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
    outg_v.append(out_g)
    # OUT COLOR
    out_c = numpy.zeros((len(out),3,224,224))
    #replico 3 volte in out
    out_temp = numpy.repeat(out,3,axis=1)

    for index_color, color in enumerate(dataset.labels):
        out_c = numpy.where(out_temp == index_color, color.reshape(3, 1, 1), out_c)
    outc_v.append(out_c)

    # LABELS
    labels_batch = torch.unsqueeze(labels_batch, 1)
    labels_batch = labels_batch.data.float().cpu().numpy()
    labels_batch = labels_batch / 12.0 * 255.0
    label_v.append(labels_batch)


    if j % 9 == 0 and j != 0:
        outg_v = numpy.asarray(outg_v).squeeze(1)
        label_v = numpy.asarray(label_v).squeeze(1)
        outc_v= numpy.asarray(outc_v).squeeze(1)
        grid = make_grid(torch.FloatTensor(outc_v), nrow=int((len(outg_v)) ** 0.5))
        writer.add_image("out_test_color", grid, index_log)

        grid = make_grid(torch.FloatTensor(outg_v), nrow=int((len(outg_v)) ** 0.5))
        writer.add_image("out_test_gray", grid, index_log)
        grid = make_grid(torch.FloatTensor(label_v), nrow=int((len(outg_v)) ** 0.5))
        writer.add_image("label_test_gray", grid, index_log)
        index_log  +=1
        outg_v = []
        label_v = []
        outc_v = []


#writer.add_image("out_gray", grid, i * batch_number + j)
