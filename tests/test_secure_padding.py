#!/usr/bin/env python3
# TEST SECURE PADDING.py
#   by Lut99
#
# Created:
#   16 Nov 2023, 15:59:52
# Last edited:
#   05 Dec 2023, 11:59:27
# Auto updated?
#   Yes
#
# Description:
#   Tests for our crypten zero pad wrapper.
#

import sys

import crypten
import numpy as np
import torch
import torch.nn as nn

sys.path.append(".")
from models.secure_padding import ZeroPad2d

crypten.init()


##### ENTRYPOINT #####
def main():
    # Try in all dimensions first
    nn_input = torch.randn(1, 1, 3, 3)
    nn_output = nn.ZeroPad2d(2).forward(nn_input)
    cnn_input = crypten.cryptensor(nn_input)
    cnn_output = ZeroPad2d(2).encrypt().forward(cnn_input)
    print(nn_input)
    print(nn_output)
    print(cnn_output.get_plain_text())

    # Try subsets
    nn_input = torch.randn(1, 1, 3, 3)
    nn_output = nn.ZeroPad2d((1, 1, 2, 0)).forward(nn_input)
    cnn_input = crypten.cryptensor(nn_input)
    cnn_output = ZeroPad2d((1, 1, 2, 0)).encrypt().forward(cnn_input)
    print(nn_input)
    print(nn_output)
    print(cnn_output.get_plain_text())

    # Done!
    return 0


# Actual entrypoint
if __name__ == "__main__":
    exit(main())
