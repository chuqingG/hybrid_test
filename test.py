import numpy as np
import time
from rich import print

from fastop import *
import numpy as np
import time

DEPTH=2
SEQ_LEN=3
BATCH_SIZE=4
HIDDEN_SIZE=5

# size = 100000000
# arr1 = np.linspace(1.0,100.0, size)
# arr2 = np.linspace(1.0,100.0, size)

# arr3 = np.random.random((3,4,5))
# testpy(arr3)
# # runs = 10
# factor = 3.0

# t0 = time.time()
# for _ in range(runs):
#     multiply_vector(arr1, factor)
# print("gpu time: " + str(time.time()-t0))
# t0 = time.time()
# for _ in range(runs):
#     arr2 = arr2 * factor
# print("cpu time: " + str(time.time()-t0))

# print("results match: " + str(np.allclose(arr1,arr2)))

if __name__ == "__main__":
    y_shape = [BATCH_SIZE, DEPTH, SEQ_LEN, HIDDEN_SIZE];
    h0_shape = [BATCH_SIZE, HIDDEN_SIZE];

    ysss = np.zeros(y_shape)
    h0 = np.random.randn(*h0_shape)
    
    idx_l = [[i, 0, 0] for i in range(BATCH_SIZE)]
    indices = np.array(idx_l)
    print(ysss)
    print(h0)
    
    scatter_nd(ysss, indices, h0, "cuda")
    print(ysss)
    
    