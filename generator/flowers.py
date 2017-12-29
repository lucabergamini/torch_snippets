import torch
import numpy
import os
import re
from torch.utils.data import Dataset, DataLoader
import cv2

numpy.random.seed(5)


class FLOWER(Dataset):
    """
    lapis loader, ha bisogno del file di corrispondenza
    in base_folder ci sono direttamente le immagini
    """

    def __init__(self, data_folder,data_aug=False):
        # len e semplicemente il numero di file
        self.len = len(os.listdir(data_folder))
        self.crop_size=112
        self.data_aug = data_aug

        self.scale_fator = 2
        # mantengo nomi file e labels, cosi sono piu veloce
        self.data_names = [os.path.join(data_folder, name) for name in sorted(os.listdir(data_folder))]

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
        #crop 112x112 dal centro
        w = data.shape[1]
        h = data.shape[0]
        dimension = self.crop_size*self.scale_fator
        if not self.data_aug:
            start_h = (h-dimension)//2
            start_w = (w-dimension)//2
            data = data[start_h:start_h + dimension, start_w:start_w + dimension]
        else:
            start_h = numpy.random.randint(0,h-dimension)
            start_w = numpy.random.randint(0,w-dimension)
            data = data[start_h:start_h+dimension,start_w:start_w+dimension]

        data = cv2.resize(data,(self.crop_size,self.crop_size))
        # CHANNEL FIRST
        data = data.transpose(2, 0, 1)
        #tanh
        #data = data.astype("float32")/255.0
        data = data.astype("float32")/255.0*2-1

        return (data, data)


if __name__ == "__main__":
    dataset = FLOWER("/home/lapis/Desktop/flowers/",data_aug=True)
    gen = DataLoader(dataset,batch_size=3)
    from matplotlib import pyplot
    import time

    for i in range(10):
        a = gen.__iter__().next()
        a = a[0]
        a = (a+1)/2
        for el in a:
            pyplot.imshow(numpy.transpose(el.numpy(),(1,2,0)))
            pyplot.show()
