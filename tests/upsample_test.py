#!/usr/bin/env python3
# UPSAMPLE TEST.py
#   by Lut99
#
# Created:
#   16 Nov 2023, 15:59:52
# Last edited:
#   23 Nov 2023, 13:55:16
# Auto updated?
#   Yes
#
# Description:
#   Tests for our crypten tensor upsampling scheme.
#

import sys

import crypten
import numpy as np
import torch
import torch.nn as nn

sys.path.append(".")
from models.secure_upsample import UpsampleNearest

crypten.init()


##### ENTRYPOINT #####
def main():
    # Create the layers
    nn_model = nn.Upsample(scale_factor = 4, mode = "nearest")
    cnn_model = UpsampleNearest(scale_factor = 4).encrypt()


    input = torch.tensor([[[ 1, 2 ], [ 1, 2 ]]], dtype=torch.uint8)
    nn_res = nn_model.forward(input)
    cnn_res = cnn_model.forward(crypten.cryptensor(input))
    cnn_res = cnn_res.get_plain_text()
    np_res = np.array([[[ 1, 2 ], [ 1, 2 ]]]).repeat(4, axis=2)
    print(input)
    print(nn_res)
    print(cnn_res)
    print(np_res)


    input = torch.tensor([[[ 1, 2, 3 ], [ 4, 5, 6 ], [ 7, 8, 9 ]]], dtype=torch.uint8)
    nn_res = nn_model.forward(input)
    cnn_res = cnn_model.forward(crypten.cryptensor(input))
    cnn_res = cnn_res.get_plain_text()
    np_res = np.array([[[ 1, 2, 3 ], [ 4, 5, 6 ], [ 7, 8, 9 ]]]).repeat(4, axis=2)
    print(input)
    print(nn_res)
    print(cnn_res)
    print(np_res)


    input = torch.tensor([[[ 1, 5, 3 ], [ 88, 42.24, 6 ], [ 7, 8, 9 ]]], dtype=torch.float32)
    nn_res = nn_model.forward(input)
    cnn_res = cnn_model.forward(crypten.cryptensor(input))
    cnn_res = cnn_res.get_plain_text()
    np_res = np.array([[[ 1, 5, 3 ], [ 88, 42.24, 6 ], [ 7, 8, 9 ]]], dtype=np.float32).repeat(4, axis=2)
    print(input)
    print(nn_res)
    print(cnn_res)
    print(np_res)


    # Try for larger dimensions
    nn_model = nn.Upsample(scale_factor = (2, 2), mode = "nearest")
    cnn_model = UpsampleNearest(scale_factor = (2, 2)).encrypt()

    input = torch.tensor([[[ [ 1, 2 ], [ 3, 4 ] ], [ [ 1, 2 ], [ 3, 4 ] ]]], dtype=torch.uint8)
    nn_res = nn_model.forward(input)
    cnn_res = cnn_model.forward(crypten.cryptensor(input))
    cnn_res = cnn_res.get_plain_text()
    np_res = np.array([[[ [ 1, 2 ], [ 3, 4 ] ], [ [ 1, 2 ], [ 3, 4 ] ]]]).repeat(2, axis=2).repeat(2, axis=3)
    print(input)
    print(nn_res)
    print(cnn_res)
    print(np_res)


    return 0


# Actual entrypoint
if __name__ == "__main__":
    exit(main())
