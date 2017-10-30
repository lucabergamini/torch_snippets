import torch
import numpy
import os
import re
from torch.utils.data import Dataset,DataLoader
import cv2
numpy.random.seed(5)

class CamVid(Dataset):
    """
    camvid loader, ha bisogno del file di corrispondenza
    in base_folder ci sono direttamente le immagini
    """
    def __init__(self, data_folder, labels_folder, labels_corr_path,data_aug = True,reshape=False):
        #len e semplicemente il numero di file
        self.len = len(os.listdir(data_folder))
        assert self.len == len(os.listdir(labels_folder))
        self.data_aug = data_aug
        self.reshape = reshape

        #mantengo nomi file e labels, cosi sono piu veloce
        self.data_names = [os.path.join(data_folder,name)for name in sorted(os.listdir(data_folder)) ]
        self.labels_names = [os.path.join(labels_folder,name)for name in sorted(os.listdir(labels_folder)) ]
        #devo estrarre con regex le labels per poterle convertire
        # self.labels = []
        # with open(labels_corr_path) as f:
        #     for i,l in enumerate(f.readlines()):
        #         match = re.search('(?P<r>\d+) (?P<g>\d+) (?P<b>\d+)',l).groups()
        #         self.labels.append(numpy.asarray([int(el) for el in match]))

        self.class_weight  = torch.FloatTensor([0.58872014284134, 0.51052379608154, 2.6966278553009, 0.45021694898605, 1.1785038709641, 0.77028578519821, 2.4782588481903, 2.5273461341858, 1.0122526884079, 3.2375309467316, 4.1312313079834, 0])
        self.mean = numpy.asarray([0.41189489566336, 0.4251328133025, 0.4326707089857])
        self.std = numpy.asarray([0.27413549931506, 0.28506257482912, 0.28284674400252])
        self.labels = [(128, 128, 128),(128, 0, 0),(192, 192, 128),(128, 64, 128),(0, 0, 192),(128, 128, 0),(192, 128, 128),(64, 64, 128),(64, 0, 128),(64, 64, 0),(0, 128, 192),(0, 0, 0),]

    def __len__(self):
        return self.len
    def __iter__(self):
        return self

    def __getitem__(self, item):
        """
        
        :param item: indice immagine da caricare 
        :return: 
        """
        #carico immagine e label e la porto rgb
        data = cv2.cvtColor(cv2.imread(self.data_names[item]),cv2.COLOR_BGR2RGB)
        label = cv2.imread(self.labels_names[item],cv2.IMREAD_GRAYSCALE)

        #reshape
        if self.reshape:
            data = cv2.resize(data,(224,224))
            label = cv2.resize(label,(224,224))

        # #RGB NORM
        data = data.astype("float")
        # r = data[::,0].copy()
        # g = data[::,1].copy()
        # b = data[::,2].copy()
        # data[::,0] = r / (r+g+b+10e-06)
        # data[::,1] = g / (r+g+b+10e-06)
        # data[::,2] = b / (r+g+b+10e-06)
        data /=255.0
        data -= self.mean
        data /= self.std

        #qui dipende se ho data aug o no
        if self.data_aug:
            #devo fare crop uguale in entrambi
            w  = numpy.random.randint(0,data.shape[1]-224)
            h = numpy.random.randint(0,data.shape[0]-224)
        else:
            #setto w e h a 0
            w = 0
            h = 0
        data = data[h:h+224,w:w+224]
        label = label[h:h+224,w:w+224]

        #CHANNEL FIRST
        data = data.transpose(2,0,1)

        return (data,label)

if __name__ == "__main__":
    dataset = CamVid("/home/lapis-ml/Desktop/camvid/train/data/","/home/lapis-ml/Desktop/camvid/train/labels/","/home/lapis-ml/Desktop/camvid/labels")
    gen = DataLoader(dataset,batch_size=32,shuffle=True)

    import time

    for i in xrange(10):
        tick = time.time()
        a = gen.__iter__().next()
        exit()
        print a
        exit()
        print time.time()-tick