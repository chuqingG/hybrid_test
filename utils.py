import numpy as np
from fastop import gather_nd_ori
from rich import print


def gather_nd(input: np.array, indices: np.array):
    output_shape = list(indices.shape[:-1]) + \
                    list(input.shape[indices.shape[-1]:])
    output = np.zeros(output_shape, dtype=np.float32)
    gather_nd_ori(output, indices, input)
    return output