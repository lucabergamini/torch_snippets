import torch
from torch.utils.data import Dataset,DataLoader
import numpy


class MNIST(Dataset):
    def __init__(self,path):
        #carico file
        npz = numpy.load(path)
        data = npz["X_train"].reshape(-1,1,28,28)
        self.data = data
        self.labels = npz["y_train"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return (self.data[item],self.labels[item])


if __name__ == "__main__":
    x= MNIST("../dataset/mnist.npz")
    gen = DataLoader(x,batch_size=3,shuffle=True)
    for data,label in gen:
        pass