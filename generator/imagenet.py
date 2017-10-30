import torch
import numpy
import os
from torch.utils.data import Dataset,DataLoader
import cv2
numpy.random.seed(5)

class Imagenet(Dataset):
    """
    dataset per imagenet
    dentro base_folder ci sono le cartelle con le classi E BASTA
    le classi NON sono bilanciate
    """
    def __init__(self,base_folder):
        #mi serve lista cartelle
        #questa la tengo per essere sicuro sia sempre uguale
        list_folder = sorted(os.listdir(base_folder))

        self.labels_folder = [os.path.join(base_folder,j) for i,j in enumerate(list_folder)]
        #mi servono le lunghezze delle classi
        self.folders_len = numpy.asarray([len(os.listdir(os.path.join(base_folder,j))) for j in list_folder],dtype="int")
        self.len = numpy.sum(self.folders_len)
    def __len__(self):
        return self.len
    def __iter__(self):
        return self

    def __getitem__(self, item):
        #item e un numero tra 0 e il numero totale di elementi
        #devo capire in che cartella andare a prenderlo
        for i,folder_len in enumerate(self.folders_len):
            if item - folder_len < 0:
                #significa che item e in quella cartella
                #posso estrarre da li
                #ma devo ancora prendere l'elemento
                img_name = sorted(os.listdir(self.labels_folder[i]))[item]
                data = cv2.cvtColor(cv2.imread(os.path.join(self.labels_folder[i],img_name)),cv2.COLOR_BGR2RGB)
                #devo andare channel first
                data = data.transpose(2,0,1)
                return data,i

            else:
                item -= folder_len
        print "ERROR"

if __name__ == "__main__":
    dataset = Imagenet("/home/lapis-ml/Desktop/imagenet/train_224/")
    gen = DataLoader(dataset,batch_size=32,shuffle=True)

    import time

    for i in xrange(10):
        tick = time.time()
        a = gen.__iter__().next()
        print a
        exit()
        print time.time()-tick