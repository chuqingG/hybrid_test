import numpy as np
from rich import print

from fastop import scatter_nd
from utils import gather_nd
import numpy as np
import time

DEPTH=2
SEQ_LEN=3
BATCH_SIZE=4
HIDDEN_SIZE=5

SCATTER=False
GATHER=True

if __name__ == "__main__":
    
    if SCATTER:
        
        y_shape = [BATCH_SIZE, DEPTH, SEQ_LEN, HIDDEN_SIZE];
        h0_shape = [BATCH_SIZE, HIDDEN_SIZE];

        ysss = np.zeros(y_shape, dtype=np.float32)
        h0 = np.random.randn(*h0_shape).astype(np.float32)

        idx_l = [[i, 0, 0] for i in range(BATCH_SIZE)]
        indices = np.array(idx_l)
        print(h0)

        scatter_nd(ysss, indices, h0)
        print(ysss)
    
    if GATHER:
        
        x_shape = [BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE];
        xss = np.random.randn(*x_shape).astype(np.float32)
        indices = np.array([[i, 0] for i in range(BATCH_SIZE)])
        out = gather_nd(xss, indices)
        print(out)