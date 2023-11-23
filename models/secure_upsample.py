# SECURE UPSAMPLE.py
#   by Tim Müller
#
# Created:
#   21 Nov 2023, 16:57:38
# Last edited:
#   21 Nov 2023, 17:11:14
# Auto updated?
#   Yes
#
# Description:
#   Defines an upsampling layer for Crypten that performs nearest neighbour
#   upsampling.
#   
#   Also comes with a convenient functional equivalent.
# 
#   ## How it works
#   
#

import typing

import crypten
import crypten.nn as cnn
import torch


##### LIBRARY #####
def interpolate(input: crypten.CrypTensor, size: typing.Optional[typing.Union[int, typing.Tuple[int], typing.Tuple[int, int], typing.Tuple[int, int, int]]] = None, scale_factor: typing.Optional[typing.Union[float, typing.Tuple[float], typing.Tuple[float, float], typing.Tuple[float, float, float]]] = None, recompute_scale_factor: bool = False) -> crypten.CrypTensor:
    """
        Crypten equivalent for `torch.nn.functional.interpolate`, which performs nearest-neighbour upsampling on the given tensor.

        See <https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html> for details about the emulated behaviour.

        # Arguments
        - `input`: The input `CrypTensor` to interpolate/upsample. This tensor is expected to have 3, 4 or 5 dimensions, of which 1, 2 or 3 are spatial, respectively. The first two dimensions represent the mini-batch and channel of the sample.
        - `size`: The new size of the spatial part of the input tensor. Mutually exclusive with `scale_factor`.
        - `scale_factor`: The scale for every of the spatial dimensions of the input tensor. Mutually exclusive with `size`.
        - `recompute_scale_factor`: Recomputes the scale_factor for use in interpolation calculation. Put differently: computes the output size based on the scale_factor, then _recomputes_ the scale_factor based on the output size. This is relevant because rounding may subtely change the scale if recomputed.

        # Returns
        A new `CrypTensor` that encodes the input upsampled to the given size.
    """

    ##### INPUT VALIDATION #####
    # First, resolve the size & scale factor to a new size



    ##### UPSAMPLING #####
    # Test

    # Done
    return crypten.rand(4)



class UpsampleNearest(cnn.Module):
    """
        Crypten module for upsampling nearest tensors.

        This only works for nearest because we don't have to worry about the gnarly encrypted case;
        since we're only interpolating to values we've seen in a deterministic way, we can be sure
        that the interpolated values are correctly shared.
    """

    _scale: typing.Union[typing.Tuple[int], typing.Tuple[int, int], typing.Tuple[int, int, int]]

    def __init__(self, scale_factor: typing.Union[int, typing.Tuple[int], typing.Tuple[int, int], typing.Tuple[int, int, int]]):
        super(UpsampleNearest, self).__init__()

        # Only accept scale factors of certain shapes
        if type(scale_factor) is tuple or type(scale_factor) is list:
            scale_factor = tuple([int(f) for f in scale_factor])
            if len(scale_factor) != 1 and len(scale_factor) != 2 and len(scale_factor) != 3:
                raise ValueError("UpsampleNearest(): Can only accept scale factors in 1, 2 or 3 dimensions.")
        elif type(scale_factor) is not int:
            raise TypeError("UpsampleNearest(): Scale factor must either be an int or a sequence (tuple or list)")

        # Store locally
        self._scale = scale_factor

    def forward(self, x):
        """
            Runs a forward pass by running the interpolation on the input tensor.
        """

        # Only accept tensors of certain shapes
        if len(x.shape) < 3 or len(x.shape) > 5:
            raise ValueError(f"UpsampleNearest.forward(): Can only upsample tensors with 3, 4 or 5 dimensions (see https://pytorch.org/docs/stable/generated/torch.nn.Upsample.html)")

        # Resolve the scale factor to a tuple with the correct number of dimensions
        # NOTE: The given X has two bonus dimensions, which are the minibatches and channels before we get to the actual dimensions. As such, number of x dimensions is 3-5, but we only scale 1-3 dimensions (the latter ones).
        n_dims = len(x.shape) - 2
        scale = self._scale
        if type(scale) is int:
            scale = tuple([scale for _ in range(n_dims)])
        elif len(x.shape) != 2 + n_dims:
            raise ValueError(f"UpsampleNearest.forward(): Can only upsample tensors with matching dimensions to scale factor: got scale factor of {n_dims} dimensions, got tensor of {len(x.shape) - 2} scalable dimensions (see https://pytorch.org/docs/stable/generated/torch.nn.Upsample.html)")

        # We stretch the x using repeat(), but be warned; this is different from numpy repeat! E.g.,
        # (n_dims = 1, scale factor 2)
        # [[[4, 5,      ->    [[[4, 5, 4, 5,        # Note this repeat sucks because it doesn't repeat values but rows
        #    6, 7]]]             6, 7, 6, 7]]]
        new_x = x.repeat((1, 1) + scale)

        # But fear not, since now comes the trick: we use `CrypTensor.index_select()` magic to re-orden the output array along the "real" dimensions, e.g.,
        # (n_dims = 1, scale factor 2)
        # [[[4, 5, 4, 5,      ->    [[[4, 4, 5, 5,
        #    6, 7, 6, 7]]]             6 ,6, 7, 7]]]
        # Note that we only reorden "real" dimensions, not the two bonus dimensions
        for dim in range(n_dims):
            # Compute the index vector, which determines the new order of the tensor along this (=`dim`) axis
            # This is built as, for every index in the old x, generate SCALE new indices, which are distanced `x.shape[dim]` spaces from each other; e.g.,
            # (x dim size 2, scale factor 2)
            # [0, 2, 1, 3]
            # (x dim size 2, scale factor 4)
            # [0, 2, 4, 6, 1, 3, 5, 7]
            # (x dim size 4, scale factor 2)
            # [0, 4, 1, 5, 2, 6, 3, 7]
            # (x dim size 3, scale factor 4)
            # [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]
            # 
            # To illustrate using this list of indices, consider:
            # (x dim size 2, scale factor 2)
            # [[[4, 5, 4, 5,      .index_select([0, 2, 1, 3])    [[[4, 4, 5, 5,     # Note that we have swapped the rows according to the indices!
            #    6, 7, 6, 7]]]                                      6, 6, 7, 7]]]
            indices = []
            for i in range(x.shape[2 + dim]):
                indices += [i + x.shape[2 + dim] * j for j in range(scale[dim])]

            # Re-orden the new X accordingly
            new_x = new_x.index_select(dim=2+dim, index=torch.tensor(indices))

        # Done!
        return new_x
