import os
import tvm
from tvm import te, auto_scheduler, relay
import numpy as np

'''
For the whole LSTM network,
we have the inputs in the following shape:
    x  = [bs, seq,  hs]
    h0 = [dep, bs, hs]
    c0 = [dep, bs, hs]
    w  = [dep, 4 * hs, hs]  # seq: f, i, c, o
    u  = [dep, 4 * hs, hs]
    y  = [bs, dep, seq, hs]

'''

@auto_scheduler.register_workload
def cell(bs, hs, dtype="float32"):
    '''
    A basic cell of LSTM
    [x, h, c, w, u] -> new h, new c
    
    For the shape of tensors:
    
    if we use matmul for all regions:
        x, h, c: (bs, hs)       
        w, u:    (4*hs, hs) 
    if we use batch_amtmul for r3:
        [placeholder]

    In this cell, all matmul is matmul_NT
    e.g. mm(w, x) -> w @ xT : (4*hs, hs) @ (hs, bs)

    :param bs: batch size
    :param hs: hidden size
    :param dtype: default is "float32", because double is 
                unsupported in 'scatter_nd'
    :returns: the te list: [x, h, c, w, u, h_new, c_new]
    '''
    
    x = te.placeholder((bs, hs), name="x", dtype=dtype)
    h = te.placeholder((bs, hs), name="h", dtype=dtype)
    c = te.placeholder((bs, hs), name="c", dtype=dtype)
    w = te.placeholder((4 * hs, hs), name="w", dtype=dtype)
    u = te.placeholder((4 * hs, hs), name="u", dtype=dtype)
    
    rk = te.reduce_axis((0, hs), name='k')
    
    wx_uh = te.compute(
            (4 * hs, bs),
            lambda i, j: te.sum(w[i, rk] * x[j, rk] + u[i, rk] * h[j, rk] , axis=rk),
            name='wx_uh',
            attrs={"layout_free_placeholders": [x, h]}
        )
    ft = te.compute((hs, hs), lambda i, j: te.sigmoid(wx_uh[i, j]), name='ft')
    it = te.compute((hs, hs), lambda i, j: te.sigmoid(wx_uh[i + hs, j]), name='it')
    c_bar = te.compute((hs, hs), lambda i, j: te.tanh(wx_uh[i + 2 * hs, j]), name='c_bar')
    ot = te.compute((hs, hs), lambda i, j: te.sigmoid(wx_uh[i + 3 * hs, j]), name='ot')
    
    fc = te.compute((hs, hs), lambda i, j: ft[i, j] * c[i, j], name="fc")
    ic = te.compute((hs, hs), lambda i, j: it[i, j] * c_bar[i, j], name="ic")
    ct = te.compute((hs, hs), lambda i, j: fc[i, j] + ic[i, j], name="ct")
    ht = te.compute((hs, hs), lambda i, j: te.tanh(ct[i, j]) * ot[i, j], name="ht")
    
    return [x, h, c, w, u, ht, ct]