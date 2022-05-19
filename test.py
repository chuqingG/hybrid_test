import numpy as np
from rich import print

from fastop import *
import numpy as np
import time

DEPTH=2
SEQ_LEN=3
BATCH_SIZE=4
HIDDEN_SIZE=5

DEMO=False

if __name__ == "__main__":
    
    if DEMO:
        arr1 = np.linspace(1.0,100.0, 5)
        factor = 3.0
        print(arr1)
        multiply_vector(arr1, factor)
        print(arr1)
    else:
        y_shape = [BATCH_SIZE, DEPTH, SEQ_LEN, HIDDEN_SIZE];
        h0_shape = [BATCH_SIZE, HIDDEN_SIZE];
    
        ysss = np.zeros(y_shape)
        h0 = np.random.randn(*h0_shape)
        
        idx_l = [[i, 0, 0] for i in range(BATCH_SIZE)]
        indices = np.array(idx_l)
        print(ysss)
        print(h0)
        
        scatter_nd(ysss, indices, h0)
        print(ysss)
    
    