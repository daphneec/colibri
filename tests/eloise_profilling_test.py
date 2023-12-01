import os
import sys
sys.path.append('/Users/eloise/workspace/HR-NAS/code/hrnas/')
from utils.secure_model_profiling import *
from utils.fix_hook import fix_hook

import crypten
import crypten.nn as cnn



class DummyNetwork(cnn.Module):

    def __init__(self):
        super(DummyNetwork, self).__init__()
        self.conv1 = cnn.Conv2d(3, 16, 3, padding=1)  # Increased output channels and adjusted kernel size
        self.conv2 = cnn.Conv2d(16, 32, 3, padding=1)  # Added another convolutional layer
        self.pool = cnn.MaxPool2d(2, 2)
        self.relu = cnn.ReLU()
        self.n_params = 0
        self.n_macs = 0
        self.n_seconds = 0
        #self.act1 = mb.SqueezeAndExcitation(2,2, active_fn=cnn.ReLU)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        #x = self.act1(x)
        return x


if __name__ == "__main__":
    # Prepare crypten
    crypten.init()

    # Create the network and encrypt it
    fix_hook(cnn.Module)  
    dummy_network = DummyNetwork().encrypt()
    dummy_input = crypten.randn(1, 3, 28, 28)
    # Encrypt it and add the profiling thingies
    
    # #dummy_network.forward(dummy_input)
    # #a = cnn.from_pytorch(dummy_network, dummy_input)
    model_profiling_hooks = []
    model_profiling(dummy_network, 28, 28)
