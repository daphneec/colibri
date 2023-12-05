# FIX HOOK.py
#   by Lut99
#
# Created:
#   09 Nov 2023, 10:41:00
# Last edited:
#   05 Dec 2023, 13:22:46
# Auto updated?
#   Yes
#
# Description:
#   File that implements fixes for Crypten's register_forward_hook().
#

import sys
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar, Union

import crypten
import crypten.nn as cnn
import torch


##### LIBRARY #####
T = TypeVar('T', bound='cnn.Module')

class Hook:
    """
        Wrapper around a (forward) hook that also stores some settings.
    """

    hook: Union[
        Callable[[T, Tuple[Any, ...], Any], Optional[Any]],
        Callable[[T, Tuple[Any, ...], Dict[str, Any], Any], Optional[Any]],
    ]
    with_kwargs: bool
    always_call: bool

    def __init__(
        self,
        hook: Union[
            Callable[[T, Tuple[Any, ...], Any], Optional[Any]],
            Callable[[T, Tuple[Any, ...], Dict[str, Any], Any], Optional[Any]],
        ],
        with_kwargs: bool,
        always_call: bool
    ):
        """
            Constructor for the Hook that creates a new object out of it.

            # Arguments
            - `hook`: The hook/callback to store.
            - `with_kwargs`: Whether to call it with keyword arguments given at some other place or not, not sure what to do with this.
            - `always_call`: Whether to always call it... which, I think, would be sensible? Edit: No I'm assuming these are also called when the forward pass crashes. Let's also warn for this.
        """

        self.hook = hook
        self.with_kwargs = with_kwargs
        self.always_call = always_call

        # Hit some warnings about stuff unimplemented
        if self.always_call:
            print("WARNING: cnn.Module.register_forward_hook(): Asked to add a forward hook with `always_call` set, but the custom Crypten implementation does nothing with this.", file=sys.stderr)



def _mean(tensor, *args, **kwargs):
    """
        Computes the mean of the values in the given tensor.
    """

    tensor.mean(*args, **kwargs)

def _sum(tensor, *args, **kwargs):
    """
        Computes the sum of the values in the given tensor.
    """

    tensor.sum(*args, **kwargs)



def _register_forward_hook(
    self,
    hook: Union[
        Callable[[T, Tuple[Any, ...], Any], Optional[Any]],
        Callable[[T, Tuple[Any, ...], Dict[str, Any], Any], Optional[Any]],
    ],
    *,
    prepend: bool = False,
    with_kwargs: bool = False,
    always_call: bool = False
):
    """
        Registers new hooks that are called *after* the forward pass has commenced.
    """

    # Ensure the hooks list exist for this module
    try:
        getattr(self, "_forward_hooks")
    except AttributeError:
        self._forward_hooks = []

    # Build the Hook object
    handle = Hook(hook, with_kwargs, always_call)

    # Either pre- or append the hook
    if prepend:
        self._forward_hooks.insert(0, handle)
    else:
        self._forward_hooks.append(handle)

    # Alrighty done
    return

def _forward_override(self, *args, **kwargs):
    """
        Override for the normal crypten forward that runs its forward, then calls hooks when a result has been produced.
    """

    # Run the normal forward
    x = self._unhooked_forward(*args, **kwargs)

    # Call any hooks, if any
    try:
        for hook in self._forward_hooks:
            # Alrighty-o, call the hook! (with or without keywords, we're not picky)
            if hook.with_kwargs:
                hook_x = hook.hook(self, args, x)
            else:
                hook_x = hook.hook(self, args, kwargs, x)

            # Propagate the result, if any
            if hook_x is not None:
                x = hook_x
    except AttributeError:
        # Nothing to do get
        pass

    # Done
    return x



def _conv_init(
    self: cnn.Conv2d,
    in_channels: Any,
    out_channels: Any,
    kernel_size: Any,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
    bias: bool = True
):
    """
        Override for the `cnn.Conv2d` constructor to make it remember its input channels.
    """

    # Run the normal init
    self._untouched_init(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

    # Remember the input channels
    self.in_channels = in_channels



def fix_crypten():
    """
        Fixes missing functions in crypten.

        Specifically, applies the following functions for this library:
        - `fix_lib()` to `crypten`;
        - `fix_init()` to `crypten.nn.init`;
        - `fix_hook()` to `crypten.nn.Module`; and
        - `fix_conv()` to `crypten.nn.Conv2d`.

        Simply call this function \*once\* and you should be good to go.
    """

    fix_lib(crypten)
    # fix_init(cnn.init)
    fix_hook(cnn.Module)
    fix_conv(cnn.Conv2d)

def fix_lib(lib):
    """
        Fixes stuff like `mean` and `sum` in the given Crypten library module.

        Specifically, injects:
        - `crypten.mean()` as an alias for `CrypTensor.mean()`
        - `crypten.sum()` as an alias for `CrypTensor.sum()`

        Use it like so:
        ```python
        fix_lib(crypten)
        ```
    """

    # Inject mean if it does not exist
    try:
        getattr(lib, 'mean')
        print("NOTE: utils.fix_hook.fix_lib(): Not fixing `mean` for given library as it apparently already exists")
    except AttributeError:
        lib.mean = _mean

    # Inject sum if it does not exist
    try:
        getattr(lib, 'sum')
        print("NOTE: utils.fix_hook.fix_lib(): Not fixing `sum` for given library as it apparently already exists")
    except AttributeError:
        lib.sum = _sum

# def fix_init(init):
#     """
#         Fixes stuff like `_calculate_fan_in_and_fan_out` not existing in the given Crypten (init) module.

#         Specifically, injects:
#         - `crypten.nn.init._calculate_fan_in_and_fan_out()` as an alias for `torch.nn.init._calculate_fan_in_and_fan_out()`.
#         - `crypten.nn.init.calculate_gain()` as an alias for `torch.nn.init.calculate_gain()`.

#         Use it like so:
#         ```python
#         fix_init(cnn.init)
#         ```
#     """

#     try:
#         getattr(init, '_calculate_fan_in_and_fan_out')
#         print("NOTE: utils.fix_hook.fix_init(): Not fixing `_calculate_fan_in_and_fan_out` for given library as it apparently already exists")
#     except AttributeError:
#         init._calculate_fan_in_and_fan_out = torch.nn.init._calculate_fan_in_and_fan_out

#     try:
#         getattr(init, 'calculate_gain')
#         print("NOTE: utils.fix_hook.fix_init(): Not fixing `calculate_gain` for given library as it apparently already exists")
#     except AttributeError:
#         init.calculate_gain = torch.nn.init.calculate_gain

def fix_hook(ty):
    """
        Fixes `register_forward_hook()` not existing in the given Crypten Module.

        Specifically, injects:
        - `cnn.Module.register_forward_hook()`
        - `cnn.Module.apply()`
        - A wrapper around `cnn.Module.forward()` to implement the hooks. The old forward is re-injected as `cnn.Module._unhooked_forward()`.

        Use it like so:
        ```python
        # Should be all you need, implements it for *all* Crypten modules
        fix_hook(cnn.Module)
        ```
    """

    # Inject the functions if they haven't been injected already
    try:
        getattr(ty, 'register_forward_hook')
        print("NOTE: utils.fix_hook.fix_hook(): Not fixing `register_forward_hook` for given module as it apparently already exists")
    except AttributeError:
        ty.register_forward_hook = _register_forward_hook

    try:
        getattr(ty, '_unhooked_forward')
        print("NOTE: utils.fix_hook.fix_hook(): Not fixing `forward` override for given module as it apparently already exists (or rather, `_unhooked_forward` already exists)")
    except AttributeError:
        ty._unhooked_forward = cnn.Module.forward
        ty.forward = _forward_override
    try:
        getattr(ty, "apply")
        print("NOTE: utils.fix_hook.fix_hook(): Not fixing `apply` because it already exists")
    except AttributeError:
        ty.apply = ty._apply

def fix_conv(conv):
    """
        Fixes `in_channels` not existing in the given Crypten module.

        Specifically, injects:
        - A wrapper around `cnn.Conv2d.__init__()` to make it store its input channels under `in_channels`.

        Use it like so:
        ```python
        fix_conv(cnn.Conv2d)
        ```
    """

    # Inject the wrapper around __init__
    if getattr(conv, "__init__") is not _conv_init:
        conv._untouched_init = cnn.Conv2d.__init__
        conv.__init__ = _conv_init
    else:
        print("NOTE: utils.fix_hook.fix_conv(): Not fixing `__init__` for given module as it is already overwritten")
