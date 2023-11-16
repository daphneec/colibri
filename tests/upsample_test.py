#!/usr/bin/env python3
# UPSAMPLE TEST.py
#   by Lut99
#
# Created:
#   16 Nov 2023, 15:59:52
# Last edited:
#   16 Nov 2023, 16:22:27
# Auto updated?
#   Yes
#
# Description:
#   Tests for our crypten tensor upsampling scheme.
#

import sys

import crypten
import torch
import torch.nn as nn

sys.path.append(".")
from models.secure_hrnet import UpsampleNearest

crypten.init()


##### ENTRYPOINT #####
def main():
    # Create the layers
    nn_model = nn.Upsample(scale_factor = 2, mode = "nearest")
    cnn_model = UpsampleNearest(scale_factor = 2).encrypt()


    input = torch.tensor([[[ 1, 2 ], [ 1, 2 ]], [[ 1, 2 ], [ 1, 2 ]]], dtype=torch.uint8)
    nn_res = nn_model.forward(input)
    cnn_res = cnn_model.forward(crypten.cryptensor(input))
    cnn_res = cnn_res.get_plain_text()
    print(input)
    print(nn_res)
    print(cnn_res)


    return 0


# Actual entrypoint
if __name__ == "__main__":
    exit(main())
