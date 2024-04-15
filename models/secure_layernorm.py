#!/usr/bin/env python3
# SECURE LAYERNORM.py
#   by Lut99
#
# Created:
#   22 Jan 2024, 14:21:22
# Last edited:
#   22 Jan 2024, 15:54:42
# Auto updated?
#   Yes
#
# Description:
#   Implements a [`LayerNorm`] for Crypten in terms of [`BatchNorm`].
#

import sys
sys.path.append('/Users/eloise/workspace/HR-NAS/code/crypten_eloise/')
import crypten_eloise as crypten
import crypten_eloise.nn as cnn

# import crypten
# import crypten.nn as cnn
import torch


##### TESTS #####
def test_layernorm_theory():
    """
        A simple function to see if we can get away implementing a LayerNorm using BatchNorm.
    """

    import torch.nn as nn

    # Define some input
    inp = torch.rand([2, 3, 4, 5])
    print(f"Input dimensions: {inp.shape}")
    print(f"Input:\n{80 * '-'}\n{inp}\n{80 * '-'}\n")

    # Run it thru a LayerNorm layer
    layer_norm = nn.LayerNorm(inp.shape[1:], eps=0.00001)
    print(f"Layer weights:\n{80 * '-'}\n{layer_norm.weight}\n{80 * '-'}\nLayer bias:\n{80 * '-'}\n{layer_norm.bias}\n{80 * '-'}\n")
    inp_lyr = layer_norm.forward(inp)

    # Run it thru a BatchNorm2D layer with some dope transposing
    inp_t = torch.transpose_copy(inp, 0, 1)
    print(f"Transposed dimensions: {inp_t.shape}")
    batch_norm = nn.BatchNorm2d(inp_t.shape[1], eps=0.00001, momentum=None)
    print(f"Batch weights:\n{80 * '-'}\n{batch_norm.weight}\n{80 * '-'}\nBatch bias:\n{80 * '-'}\n{batch_norm.bias}\n{80 * '-'}\n")
    inp_bch_t = batch_norm.forward(inp_t)
    inp_bch = torch.transpose_copy(inp_bch_t, 0, 1)
    # inp_bch = inp_bch_t

    # Print it
    print(f"Layer output:\n(Shape {inp_lyr.shape})\n{80 * '-'}\n{inp_lyr}\n{80 * '-'}\n")
    print(f"Batch output:\n(Shape {inp_bch.shape})\n{80 * '-'}\n{inp_bch}\n{80 * '-'}\n")

    # Assert they are the same the same
    assert torch.all(torch.isclose(inp_lyr, inp_bch, rtol=0.0001, atol = 0.0001))

def test_layernorm():
    """
        Practical unit test for the [`LayerNorm`] layer.
    """

    import torch.nn as nn
    crypten.init()

    # Define some input
    inp = torch.rand([2, 3, 4, 5])
    print(f"Input:\n{80 * '-'}\n{inp}\n{80 * '-'}\n")

    # Run it thru a LayerNorm layer
    layer_norm = nn.LayerNorm(inp.shape[1:])
    inp_lyr = layer_norm.forward(inp)

    # Run it thru the Crypten layer
    crypt_norm = LayerNorm(inp.shape).encrypt()
    inp_crp = crypt_norm.forward(crypten.cryptensor(inp)).get_plain_text()

    # Print it
    print(f"Layer output:\n(Shape {inp_lyr.shape})\n{80 * '-'}\n{inp_lyr}\n{80 * '-'}\n")
    print(f"Crypten output:\n(Shape {inp_crp.shape})\n{80 * '-'}\n{inp_crp}\n{80 * '-'}\n")

    # Assert they are the same the same
    # TODO Check with Daphnee is these tolerances are OK
    assert torch.all(torch.isclose(inp_lyr, inp_crp, rtol=0.2, atol = 0.2))





##### LIBRARY #####
class LayerNorm(cnn.Module):
    """
        Applies layer normalisation to an input.

        Implemented in terms of the already existing batchnormalization implementation.
    """

    # The wrapper BatchNorm module (either [`cnn.BatchNorm1d`], [`cnn.BatchNorm2d`] or [`cnn.BatchNorm3d`])
    norm: cnn.BatchNorm2d


    def __init__(self, dims, eps=0.00001):
        """
            Constructor for the LayerNorm.

            # Arguments
            - `dims`: The dimensions to normalize over.
            - `eps`: A value added to the denominator for numerical stability.
        """

        # Initialize the `cnn.Module`
        super().__init__()

        # Resolve dims to something indexable
        if type(dims) == int:
            dims = (dims,)

        # Choose the matching batchnorm
        if len(dims) == 4:
            self.norm = cnn.BatchNorm2d(dims[0], eps=eps)
            self.norm.running_mean = None
            self.norm.running_var = None
        else:
            raise ValueError(f"Can only process input 'dims' of 4 dimensions; got {dims} ({len(dims)} dimensions)")

    def forward(self, x):
        """
            Runs the forward pass for the LayerNorm.
        """

        # Assert size is correct
        if len(x.shape) != 4:
            raise ValueError(f"Can only process input with 4 dimensions, got {len(x.shape)})")

        # Alright, transpose the input first along its first two dimensions
        if isinstance(x, crypten.CrypTensor):
            x_t = x.transpose(0, 1)
        else:
            x_t = torch.tranpose_copy(x, 0, 1)

        # Run the batch norm on it
        xn_t = self.norm.forward(x_t)

        # Tranpose back
        if isinstance(xn_t, crypten.CrypTensor):
            xn = xn_t.transpose(0, 1)
        else:
            xn = torch.tranpose_copy(xn_t, 0, 1)

        # Done
        return xn





##### ENTRYPOINT #####
if __name__ == "__main__":
    test_layernorm_theory()
    test_layernorm()
