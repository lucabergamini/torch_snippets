import torch
import numpy

def classification_accuracy(out,labels):
    #mi servono tensori
    out = out.data
    labels = labels.data
    out = torch.max(out,1)[1]
    accuracy = float(torch.sum(out==labels))
    accuracy /= numpy.prod(list(out.size()))
    return accuracy


class RollingMeasure(object):
    def __init__(self):
        self.measure = 0.0
        self.iter = 0

    def __call__(self, measure):
        #passo nuovo valore e ottengo average
        #se first call inizializzo
        if self.iter == 0:
            self.measure = measure
        else:
            self.measure = (1.0/self.iter*measure)+(1-1.0/self.iter)*self.measure
        self.iter +=1
        return self.measure


