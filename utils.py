import os
import sys
import tvm
from tvm import te, auto_scheduler, relay
from tvm.contrib import graph_executor
import numpy as np
from time import time
from fastop import gather_nd_ori, lstmcell_ori
from rich import print
from lstm import cell


def gather_nd(input: np.array, indices: np.array):
    output_shape = list(indices.shape[:-1]) + \
                    list(input.shape[indices.shape[-1]:])
    output = np.zeros(output_shape, dtype=np.float32)
    gather_nd_ori(output, indices, input)
    return output


def lstmcell(x: np.array, h:np.array, c:np.array,
             w:np.array, u:np.array):
    h_shape = list(x.shape)
    ht = np.zeros(h_shape, dtype=np.float32)
    ct = np.zeros(h_shape, dtype=np.float32)
    lstmcell_ori(x, h, c, w, u, ht, ct)
    return ht, ct
            

def get_cell(target, bs, hs, dtype="float32"):
    log_file = f'log/cell_b{bs}_h{hs}.json'
    code_file = f"code/cell_b{bs}_h{hs}.cu"
    lib_file = f"lib/cell_b{bs}_h{hs}.so"
    
    if not os.path.exists(log_file):
        tune_option = auto_scheduler.TuningOptions(
            num_measure_trials=200,
            measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
            verbose=2,
        )
        
        start = time()
        task = auto_scheduler.SearchTask( 
            func=cell, args=(bs, hs, dtype), target=target)
        task.tune(tune_option)
        tunetime = time() - start
        
        sch, args = task.apply_best(log_file)
        # print(args)
        with open(code_file, 'w') as f:
            m_l = tvm.lower(sch, args, name='cell')
            mod = tvm.build(m_l, target = target)
            mod.export_library(lib_file)
            print(mod.imported_modules[0].get_source(), file=f)
            print("\n total tuning time: %.6f s\n" % tunetime, file=f)

    f: tvm.runtime.Module = tvm.runtime.load_module(lib_file)
    return f
    

def run_cell(f, x, h, c, w, u, bs, hs, target, dtype="float32"):
    dev = tvm.device(str(target), 0)
    start = time()
    x_tvm = tvm.nd.array(x, dev)
    h_tvm = tvm.nd.array(h, dev)
    c_tvm = tvm.nd.array(c, dev)
    w_tvm = tvm.nd.array(w, dev)
    u_tvm = tvm.nd.array(u, dev)
    ht_tvm = tvm.nd.array(np.zeros((bs, hs), dtype=dtype), dev)
    ct_tvm = tvm.nd.array(np.zeros((bs, hs), dtype=dtype), dev)
    end = time() - start
    f(x_tvm, h_tvm, c_tvm, w_tvm, u_tvm, ht_tvm, ct_tvm)
    
    print("%.5f ms" % (end * 1000))
    
    evaluator = f.time_evaluator(f.entry_name, dev, repeat=20)
    print("bs = %d, hs = %d : %.4f ms" % (bs, hs, 
                np.median(evaluator(x_tvm, h_tvm, c_tvm, w_tvm, u_tvm, ht_tvm, ct_tvm).results) * 1000))
    
    return ht_tvm, ct_tvm