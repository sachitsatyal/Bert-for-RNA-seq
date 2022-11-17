from argparse import Namespace
from collections.abc import MutableMapping

from torch import Tensor


def tensor_sizes(input=None, **kwargs) -> ...:
    """
    A very useful method to inspect the sizes of tensors in object containing Tensors
    Args:
        input ():
        **kwargs ():

    Returns:

    """
    if kwargs:
        return tensor_sizes(kwargs)

    if isinstance(input, (dict, MutableMapping)):
        return {key: tensor_sizes(v) \
                for key, v in input.items()}
    if isinstance(input, Namespace):
        return {key: tensor_sizes(v) \
                for key, v in input.__dict__.items()}

    elif isinstance(input, tuple):
        return tuple(tensor_sizes(v) for v in input)
    elif isinstance(input, list):
        if len(input) and isinstance(input[0], str):
            return len(input)
        return [tensor_sizes(v) for v in input]
    elif isinstance(input, set):
        if len(input) and isinstance(list(input)[0], str):
            return len(input)
        elif len(input) and isinstance(list(input)[0], tuple):
            return ['.'.join(tup) for tup in input]
        return len(input)

    else:
        if input is not None and hasattr(input, "shape"):
            if isinstance(input, Tensor) and input.dim() == 0:
                return input.item()

            return list(input.shape)
        else:
            return input