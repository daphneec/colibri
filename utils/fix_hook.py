# FIX HOOK.py
#   by Lut99
#
# Created:
#   09 Nov 2023, 10:41:00
# Last edited:
#   16 Nov 2023, 15:12:15
# Auto updated?
#   Yes
#
# Description:
#   File that implements fixes for Crypten's register_forward_hook().
#

import sys
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar, Union

import crypten.nn as cnn


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



def fix_lib(lib):
    """
        Fixes stuff like `mean` and `sum` in the given Crypten library module.

        Use it like so:
        ```python
        fix_lib(crypten)
        ```
    """

    # Inject mean if it does not exist
    try:
        getattr(lib, 'mean')
        print("NOTE: fix_lib(): Not fixing `mean` for given library as it apparently already exists")
    except AttributeError:
        lib.mean = _mean

    # Inject sum if it does not exist
    try:
        getattr(lib, 'sum')
        print("NOTE: fix_lib(): Not fixing `sum` for given library as it apparently already exists")
    except AttributeError:
        lib.sum = _sum


def fix_hook(ty):
    """
        Fixes `register_forward_hook()` not existing in the given Crypten Module.

        Use it like so:
        ```python
        # Should be all you need, implements it for *all* Crypten modules
        fix_hook(cnn.Module)
        ```
    """

    # Inject the functions if they haven't been injected already
    try:
        getattr(ty, 'register_forward_hook')
        print("NOTE: fix_hook(): Not fixing `register_forward_hook` for given module as it apparently already exists")
    except AttributeError:
        ty.register_forward_hook = _register_forward_hook

    try:
        getattr(ty, '_unhooked_forward')
        print("NOTE: fix_hook(): Not fixing `forward` override for given module as it apparently already exists (or rather, `_unhooked_forward` already exists)")
    except AttributeError:
        ty._unhooked_forward = cnn.Module.forward
        ty.forward = _forward_override
    try:
        getattr(ty, "apply")
        print("NOTE: fix_hook(): Not fixing `apply` because it already exists")
    except AttributeError:
        ty.apply = ty._apply
