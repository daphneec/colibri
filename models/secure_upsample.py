# SECURE UPSAMPLE.py
#   by Tim Müller
#
# Created:
#   21 Nov 2023, 16:57:38
# Last edited:
#   05 Dec 2023, 11:33:08
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

import math
import typing

import crypten_eloise as crypten
import crypten_eloise.nn as cnn
# import crypten
# import crypten.nn as cnn
import torch


##### LIBRARY #####
def interpolate_nearest(input: crypten.CrypTensor, size: typing.Optional[typing.Union[int, typing.Tuple[int], typing.Tuple[int, int], typing.Tuple[int, int, int]]] = None, scale_factor: typing.Optional[typing.Union[int, typing.Tuple[int], typing.Tuple[int, int], typing.Tuple[int, int, int]]] = None) -> crypten.CrypTensor:
    """
        Crypten equivalent for `torch.nn.functional.interpolate`, which performs nearest-neighbour upsampling on the given tensor.

        See <https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html> for details about the emulated behaviour.

        # Arguments
        - `input`: The input `CrypTensor` to interpolate/upsample. This tensor is expected to have 3, 4 or 5 dimensions, of which 1, 2 or 3 are spatial, respectively. The first two dimensions represent the mini-batch and channel of the sample.
        - `size`: The new size of the spatial part of the input tensor. Mutually exclusive with `scale_factor`.
        - `scale_factor`: The scale for every of the spatial dimensions of the input tensor. Mutually exclusive with `size`.

        # Returns
        A new `CrypTensor` that encodes the input upsampled to the given size.
    """

    ##### INPUT VALIDATION #####
    # Assert the input makes sense
    if len(input.shape) < 3 or len(input.shape) > 5:
        raise ValueError(f"Can only interpolate tensors of 3, 4 or 5 dimensions (got tensor of {len(input.shape)} dimension(s))")
    spatial_dims = len(input.shape) - 2

    # Assert exactly one of the possible new sizes is given
    if size is None and scale_factor is None:
        raise ValueError("You must give either `size` or `scale_factor`")
    elif size is not None and scale_factor is not None:
        raise ValueError("Cannot give both `size` and `scale_factor`; specify one, set the other to None")

    # Compute a new size based on which is given
    if size is not None:
        # Either check if the list has correct dimensionality, or create a correctly dimensionalised list if given a constant
        if isinstance(size, (tuple, list)):
            if len(size) != spatial_dims:
                raise ValueError(f"`size` must have as many dimensions as the `input` has spatial dimensions; `input` has {spatial_dims} spatial dimensions ({len(input.shape)} total dimensions), `size` has {len(size)} dimensions")
            size = tuple([int(s) for s in size])
        elif isinstance(size, int):
            # Create a list with it
            size = tuple([size for _ in range(spatial_dims)])
        else:
            raise TypeError("`size` must either be a single integer or a tuple of 1-3 integers")

        # Then we convert it to a scale_factor
        scale_factor = []
        for dim in range(spatial_dims):
            # Compute the scale
            scale = size[dim] / input.shape[2 + dim]
            if int(scale) != scale:
                print("WARNING: secure_upsample.interpolate(): Given size does not produce a whole scale factor, so it will be rounded to the nearest integer")
            scale_factor.append(int(scale))
        scale_factor = tuple(scale_factor)

    if scale_factor is not None:
        # Either check if the list has correct dimensionality, or create a correctly dimensionalised list if given a constant
        if isinstance(scale_factor, (tuple, list)):
            if len(scale_factor) != spatial_dims:
                raise ValueError(f"`scale_factor` must have as many dimensions as the `input` has spatial dimensions; `input` has {spatial_dims} spatial dimensions ({len(input.shape)} total dimensions), `scale_factor` has {len(size)} dimensions")
            warned = False
            for s in scale_factor:
                if isinstance(s, float):
                    if not warned:
                        print("WARNING: secure_upsample.interpolate(): Given scale factor as a tuple/list with a floating-point, which is rounded to the nearest integer")
                        warned = True
                elif not isinstance(s, int):
                    raise TypeError("`scale_factor` must be a tuple or list of integers or floats")
            scale_factor = tuple([int(s) for s in scale_factor])
        elif isinstance(scale_factor, (int, float)):
            # Assert the scale_factor is an integer
            if isinstance(scale_factor, float):
                print("WARNING: secure_upsample.interpolate(): Given scale factor as a floating-point, which is rounded to the nearest integer")
                scale_factor = int(math.round(scale_factor))
            # Create a list with it
            scale_factor = tuple([int(scale_factor) for _ in range(spatial_dims)])
        else:
            raise TypeError("`scale_factor` must either be a single int/float, or a tuple of 1-3 integers/floats")

        # Now convert to a new size based on the input tensor
        size = tuple([input.shape[2 + dim] * scale_factor[dim] for dim in range(spatial_dims)])


    ##### UPSAMPLING #####
    # We stretch the input using repeat(), but be warned; this is different from numpy repeat! E.g.,
    # (n_dims = 1, scale factor 2)
    # [[[4, 5,      ->    [[[4, 5, 4, 5,        # Note this repeat sucks because it doesn't repeat values but rows
    #    6, 7]]]             6, 7, 6, 7]]]
    new_input = input.repeat((1, 1) + scale_factor)

    # But fear not, since now comes the trick: we use `CrypTensor.index_select()` magic to re-orden the output array along the "real" dimensions, e.g.,
    # (n_dims = 1, scale factor 2)
    # [[[4, 5, 4, 5,      ->    [[[4, 4, 5, 5,
    #    6, 7, 6, 7]]]             6 ,6, 7, 7]]]
    # Note that we only reorden "real" dimensions, not the two bonus dimensions
    for dim in range(spatial_dims):
        # Compute the index vector, which determines the new order of the tensor along this (=`dim`) axis
        # This is built as, for every index in the old input, generate SCALE new indices, which are distanced `input.shape[dim]` spaces from each other; e.g.,
        # (input dim size 2, scale factor 2)
        # [0, 2, 1, 3]
        # (input dim size 2, scale factor 4)
        # [0, 2, 4, 6, 1, 3, 5, 7]
        # (input dim size 4, scale factor 2)
        # [0, 4, 1, 5, 2, 6, 3, 7]
        # (input dim size 3, scale factor 4)
        # [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]
        # 
        # To illustrate using this list of indices, consider:
        # (input dim size 2, scale factor 2)
        # [[[4, 5, 4, 5,      .index_select([0, 2, 1, 3])    [[[4, 4, 5, 5,     # Note that we have swapped the rows according to the indices!
        #    6, 7, 6, 7]]]                                      6, 6, 7, 7]]]
        indices = []
        for i in range(input.shape[2 + dim]):
            indices += [i + input.shape[2 + dim] * j for j in range(scale_factor[dim])]

        # Re-orden the new input accordingly
        new_input = new_input.index_select(dim=2+dim, index=torch.tensor(indices))


    # Done, return!
    return new_input



class UpsampleNearest(cnn.Module):
    """
        Crypten module for upsampling nearest tensors.

        This only works for nearest because we don't have to worry about the gnarly encrypted case;
        since we're only interpolating to values we've seen in a deterministic way, we can be sure
        that the interpolated values are correctly shared.
    """

    _size         : typing.Optional[typing.Union[int, typing.Tuple[int], typing.Tuple[int, int], typing.Tuple[int, int, int]]]
    _scale_factor : typing.Optional[typing.Union[int, typing.Tuple[int], typing.Tuple[int, int], typing.Tuple[int, int, int]]]

    def __init__(self, size: typing.Optional[typing.Union[int, typing.Tuple[int], typing.Tuple[int, int], typing.Tuple[int, int, int]]] = None, scale_factor: typing.Optional[typing.Union[int, typing.Tuple[int], typing.Tuple[int, int], typing.Tuple[int, int, int]]] = None):
        super(UpsampleNearest, self).__init__()

        # Store locally
        self._size = size
        self._scale_factor = scale_factor

    def forward(self, x):
        """
            Runs a forward pass by running the interpolation on the input tensor.
        """

        # Simply call interpolate on x
        return interpolate_nearest(x, self._size, self._scale_factor)
