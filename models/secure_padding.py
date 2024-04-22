# SECURE PADDING.py
#   by Lut99
#
# Created:
#   05 Dec 2023, 11:50:43
# Last edited:
#   05 Dec 2023, 11:55:30
# Auto updated?
#   Yes
#
# Description:
#   Defines a wrapper around [`cnn.ConstantPad2d`] to emulate PyTorch's
#   [`nn.ZeroPad2d`].
#

import typing

import crypten_eloise.nn as cnn
# import crypten.nn as cnn


##### LIBRARY #####
class ZeroPad2d(cnn.Module):
    """
        Wraps around Cryptens' [`cnn.ConstantPad2d`] to emulate PyTorch's [`nn.ZeroPad2d`].
    """

    _layer: cnn.ConstantPad2d

    def __init__(self, padding: typing.Union[int, typing.Tuple[int], typing.Tuple[int, int], typing.Tuple[int, int, int], typing.Tuple[int, int, int, int]]):
        """
            Constructor for the ZeroPad2d.

            # Arguments:
            - `padding`: Determines how many zeroes to pad in each boundary. If an integer, then it pads the same value in all boundaries.

            # Returns
            A new ZeroPad2d layer.
        """

        # Initialize the super
        super(ZeroPad2d, self).__init__()

        # Initialize the actual layer
        self._layer = cnn.ConstantPad2d(padding, 0, mode="constant")

    def forward(self, x):
        """
            Runs a forward pass over the ZeroPad2d module.
        """

        return self._layer.forward(x)
