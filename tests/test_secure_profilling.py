import os
import sys
sys.path.append('/Users/eloise/workspace/HR-NAS/code/colibri/')
from utils.secure_model_profiling import *
from utils.fix_hook import fix_deps

import crypten
import crypten.nn as cnn


class DummyNetwork(cnn.Module):

    def __init__(self):
        super(DummyNetwork, self).__init__()

        self.conv1 = cnn.Conv2d(3, 16, 3, padding=1)  # Increased output channels and adjusted kernel size
        self.bn = cnn.BatchNorm2d(16)
        self.conv2 = cnn.Conv2d(16, 32, (3,5), padding=1)  # Added another convolutional layer
        self.maxpool = cnn.MaxPool2d(kernel_size=3, stride=2)
        self.avgpool = cnn.AvgPool2d(kernel_size=3)
        self.relu = cnn.ReLU()
        self.flatten = cnn.Flatten(axis=1)
        self.linear = cnn.Linear(in_features=228480, out_features=1)
        self.n_seconds = 0
        #self.act1 = mb.SqueezeAndExcitation(2,2, active_fn=cnn.ReLU)

    def forward(self, x):
        print("Point 1", x.size())
        x = self.conv1(x)
        print("Point 2", x.size())
        x = self.bn(x)
        print("Point 3", x.size())
        x = self.relu(x)

        print("Point 4", x.size())

        x = self.relu(self.conv2(x))

        print("Point 5", x.size())
        # x = self.maxpool(x)
        # x = cnn.MaxPool2d(kernel_size=3, stride=2)(x)
        # print("Point 6", x.size())
        x = self.avgpool(x)
        print("Point 7", x.size())
        x = self.flatten(x)
        print("Point 8", x.size())
        x = self.linear(x)
        print("Point 9", x.size())
        #x = self.act1(x)
        return x


if __name__ == "__main__":
    # Prepare crypten
    fix_deps()
    crypten.init()

    # Create the network
    dummy_network = DummyNetwork()
    dummy_input = crypten.randn(1, 3, 256, 256)
    
    model_profiling(dummy_network, 256, 256, use_cuda=False)

