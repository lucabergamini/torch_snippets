import torch
import numpy
import os
import re
from torch.utils.data import Dataset, DataLoader
import cv2

numpy.random.seed(5)


class LAPIS(Dataset):
    """
    lapis loader, ha bisogno del file di corrispondenza
    in base_folder ci sono direttamente le immagini
    """

    def __init__(self, data_folder, labels_folder, data_aug=True, reshape=False, offset=(0,0)):
        # len e semplicemente il numero di file
        self.len = len(os.listdir(data_folder))
        assert self.len == len(os.listdir(labels_folder))
        self.data_aug = data_aug
        self.reshape = reshape

        # mantengo nomi file e labels, cosi sono piu veloce
        self.data_names = [os.path.join(data_folder, name) for name in sorted(os.listdir(data_folder))]
        self.labels_names = [os.path.join(labels_folder, name) for name in sorted(os.listdir(labels_folder))]
        # devo estrarre con regex le labels per poterle convertire
        # self.labels = []
        # with open(labels_corr_path) as f:
        #     for i,l in enumerate(f.readlines()):
        #         match = re.search('(?P<r>\d+) (?P<g>\d+) (?P<b>\d+)',l).groups()
        #         self.labels.append(numpy.asarray([int(el) for el in match]))

        #todo  std
        self.mean = numpy.asarray([ 0,  0,  0])

        #self.mean = numpy.asarray([ 0.2927674 ,  0.27401834,  0.25505481])
        self.std = numpy.asarray([1,1,1])
        self.offset = offset

    def __len__(self):
        return self.len

    def __iter__(self):
        return self

    def __getitem__(self, item):
        """

        :param item: indice immagine da caricare 
        :return: 
        """
        # carico immagine e label e la porto rgb
        data = cv2.cvtColor(cv2.imread(self.data_names[item]), cv2.COLOR_BGR2RGB)
        label = cv2.imread(self.labels_names[item], cv2.IMREAD_GRAYSCALE)

        # reshape
        if self.reshape:
            data = cv2.resize(data, (224, 224))
            label = cv2.resize(label, (224, 224),interpolation=cv2.INTER_NEAREST)

        # #RGB NORM
        data = data.astype("float")
        # r = data[::,0].copy()
        # g = data[::,1].copy()
        # b = data[::,2].copy()
        # data[::,0] = r / (r+g+b+10e-06)
        # data[::,1] = g / (r+g+b+10e-06)
        # data[::,2] = b / (r+g+b+10e-06)
        data /= 255.0
        data -= self.mean
        data /= self.std

        # qui dipende se ho data aug o no
        if self.data_aug:
            #prendo un area variabile e faccio il resize a 224x224
            #conviene avere circa dei quadrati, quindi vincolo le due dimensioni ad essere piu o meno simili
            #cosi non distorce troppo le due dimensioni
            size = numpy.random.randint(85,min(data.shape[1],data.shape[0]))
            x = numpy.random.randint(0, data.shape[1] - size)
            y = numpy.random.randint(0, data.shape[0] - size)
            # crop
            data = data[y:y + size, x:x + size]
            label = label[y:y + size, x:x + size]

            #width = numpy.random.randint(100,data.shape[1])
            #height = numpy.random.randint(100,data.shape[0])

            #x = numpy.random.randint(0,data.shape[1]-width)
            #y = numpy.random.randint(0,data.shape[0]-height)
            #crop
            #data = data[y:y+height, x:x+width]
            #label = label[y:y+height, x:x+width]
            #resize
            data = cv2.resize(data, (224, 224))
            label = cv2.resize(label, (224, 224),interpolation=cv2.INTER_NEAREST)

        else:
            # setto w e h da offset
            x = self.offset[0]
            y = self.offset[1]
            data = data[y:y + 224, x:x + 224]
            label = label[y:y + 224, x:x + 224]

        # CHANNEL FIRST
        data = data.transpose(2, 0, 1)

        return (data, label)


if __name__ == "__main__":
    dataset = LAPIS("/home/lapis-ml/Desktop/LAPIS_dataset/train/data/", "/home/lapis-ml/Desktop/LAPIS_dataset/train/labels/")
    gen = DataLoader(dataset, batch_size=32, shuffle=True)
    from matplotlib import pyplot
    import time

    for i in range(10):
        a = gen.__iter__().next()
