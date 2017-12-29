import torch
import numpy
import os
import re
from torch.utils.data import Dataset, DataLoader
import cv2

numpy.random.seed(5)


class CELEBA(Dataset):
    """
    lapis loader, ha bisogno del file di corrispondenza
    in base_folder ci sono direttamente le immagini
    """

    def __init__(self, data_folder):
        # len e semplicemente il numero di file
        self.len = len(os.listdir(data_folder))

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
        #prendo crop 64x64 dal centro
        c_x = data.shape[1]//2
        c_y = data.shape[0]//2
        data = data[c_y-64:c_y+64,c_x-64:c_x+64]
        data = cv2.resize(data,(64,64))
        # CHANNEL FIRST
        data = data.transpose(2, 0, 1)
        #tanh
        data = data.astype("float32")/255.0*2-1
        return (data, data)


if __name__ == "__main__":
    dataset = CELEBA("/home/lapis/Desktop/img_align_celeba/")
    gen = DataLoader(dataset,batch_size=3,shuffle=True)
    from matplotlib import pyplot
    import time

    for i in range(10):
        a = gen.__iter__().next()
        a = a[0]
        a = (a+1)/2
        for el in a:

            pyplot.imshow(numpy.transpose(el.numpy(),(1,2,0)))
            pyplot.show()
