import numpy as np
from rich import print

from fastop import scatter_nd, lstm_cxx
from utils import *
import numpy as np
import time

DEPTH=2
SEQ_LEN=16
BATCH_SIZE=128
HIDDEN_SIZE=128

# (2, 16, 128, 128)

SCATTER=False
# SCATTER=True
GATHER=False #True
LSTM=False 
TVMCXX=False #True
LSTMCXX=True


if __name__ == "__main__":
    x_shape = [BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE];
    h0_shape = [DEPTH, BATCH_SIZE, HIDDEN_SIZE];
    w_shape = [DEPTH, 4 * HIDDEN_SIZE, HIDDEN_SIZE];
    y_shape = [BATCH_SIZE, DEPTH, SEQ_LEN, HIDDEN_SIZE];
    
    
    xss = np.random.randn(*x_shape).astype(np.float32)
    h0 = np.random.randn(*h0_shape).astype(np.float32)
    c0 = np.random.randn(*h0_shape).astype(np.float32)
    ws = np.random.randn(*w_shape).astype(np.float32)
    us = np.random.randn(*w_shape).astype(np.float32)
    
    if SCATTER:

        #TODO: unstable error here: 128, 128 can run sometimes,
        ysss = np.zeros(y_shape, dtype=np.float32)
        h0 = np.random.randn(BATCH_SIZE, HIDDEN_SIZE).astype(np.float32)

        idx_l = [[i, 0, 0] for i in range(BATCH_SIZE)]
        indices = np.array(idx_l)
        # print(h0)

        scatter_nd(ysss, indices, h0)
        print(ysss)
    
    if GATHER:
        
        xss = np.random.randn(*x_shape).astype(np.float32)
        indices = np.array([[i, 0] for i in range(BATCH_SIZE)])
        # print(xss)
        out = gather_nd(xss, indices)
        print(out)
        
    if LSTM:
        '''
        .so used in TVMCXX is built here, just a demo
        '''
        target = tvm.target.Target(target="cuda", host="llvm") 
        r0 = True
        f = get_cell(target, BATCH_SIZE, HIDDEN_SIZE)
        
        if r0:
            indices = np.array([[i, 0] for i in range(BATCH_SIZE)])
            xs0 = gather_nd(xss, indices)
            ht, ct = run_cell(f, xs0, h0[0], c0[0], ws[0], us[0], 
                              BATCH_SIZE, HIDDEN_SIZE, target)
        
        
    if TVMCXX:
        # test the generate-data-from-python version
        # failed because of something wrong with the callback from cxx to python
        
        # indices = np.array([[i, 0] for i in range(BATCH_SIZE)])
        # xs0 = gather_nd(xss, indices)
        
        # ht, ct = lstmcell(xs0, h0[0], c0[0], ws[0], us[0])
        
        out = lstm_network(xss, h0, c0, ws, us)
        print(out)
        
    if LSTMCXX:
        # test the pure cxx version LSTM network
        ysss = np.random.randn(*x_shape).astype(np.float32)
        lstm_cxx(ysss, DEPTH, SEQ_LEN, BATCH_SIZE,HIDDEN_SIZE)
        # crash down when trying to access data in python
        # print(ysss)