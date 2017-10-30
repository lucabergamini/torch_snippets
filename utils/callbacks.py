import os
import socket
from datetime import datetime
import torch

class ModelCheckpoint(object):
    def __init__(self):
        self.log_dir = os.path.join('models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
        os.makedirs(self.log_dir)
        self.loss = None
    def __call__(self, net,loss=None,comment=''):
        """
        
        :param net: 
        :param loss: 
        :param comment: 
        :return: 
        """
        if loss is not None:
            if self.loss is None or self.loss > loss:
                name = datetime.now().strftime('%b%d_%H-%M-%S') +"_" + str(loss)+ "_"+comment
                path = os.path.join(self.log_dir,name)
                torch.save(net.state_dict(),path)
                self.loss = loss

