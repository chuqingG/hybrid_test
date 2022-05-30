#pragma once

#include <dlpack/dlpack.h>
#include <tvm/driver/driver_api.h>
#include <tvm/runtime/contrib/papi.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/profiling.h>
#include <tvm/runtime/registry.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <memory>

#include "utils.h"
#include "scatter.cu"

#pragma GCC diagnostic ignored "-Wformat="


__global__ void copykernel1(int **in, int **out, int len, int N)
{
    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    for(; idx<N; idx+=gridDim.x*blockDim.x)
        memcpy(out[idx], in[idx], sizeof(int)*len);

}

// template <typename T>
// __global__ void copykernel(const T *in, T *out, int len){
//   // Casting for improved loads and stores
//   for (int i = 0; i< len; i++) 
//     out[i] = in[i]; 
// }

/** 
 * For the whole LSTM network,
 * we have the inputs in the following shape:
 *      x = [bs, seq,  hs]
 *      h0 = [dep, bs, hs]
 *      c0 = [dep, bs, hs]
 *      w  = [dep, 4 * hs, hs]  # seq: f, i, c, o
 *      u  = [dep, 4 * hs, hs]
 *      y  = [bs, dep, seq, hs]       
 * 
 * real return: out = [bs, seq, hs]  => gather from y
 * 
 * WARNING: dll.so only works from (bs, hs) = (128, 128) now
 */
template <typename T>
void lstm_network(pybind11::array_t<T> x_pb,
                pybind11::array_t<T> h_pb,
                pybind11::array_t<T> c_pb,
                pybind11::array_t<T> w_pb,
                pybind11::array_t<T> u_pb,
                pybind11::array_t<T> out_pb){
    
    pybind11::buffer_info x_bf = x_pb.request();
    pybind11::buffer_info h_bf = h_pb.request();
    pybind11::buffer_info c_bf = c_pb.request();
    pybind11::buffer_info w_bf = w_pb.request();
    pybind11::buffer_info u_bf = u_pb.request();
    pybind11::buffer_info out_bf = out_pb.request();

    int64_t depth = h_bf.shape[0];
    int64_t seq_len = x_bf.shape[1];
    int64_t batch_size = x_bf.shape[0];
    int64_t hidden_size = x_bf.shape[2];

    std::string lib_path = "lib/cell_b" + std::to_string(batch_size) + 
                            "_h" + std::to_string(hidden_size) + ".so";
    std::cout << lib_path << std::endl;

    bool enabled = tvm::runtime::RuntimeEnabled("cuda");
    const tvm::runtime::PackedFunc* graph_executor_create =
                tvm::runtime::Registry::Get("tvm.graph_executor.create");
    
    auto lib = tvm::runtime::Module::LoadFromFile(lib_path);
    tvm::runtime::PackedFunc f = lib.GetFunction("cell");
    ICHECK(f != nullptr);

    // total inputs, pointer to data on cpu
    T* p_x = reinterpret_cast<T*>(x_bf.ptr);
    T* p_h = reinterpret_cast<T*>(h_bf.ptr);
    T* p_c = reinterpret_cast<T*>(c_bf.ptr);
    T* p_w = reinterpret_cast<T*>(w_bf.ptr);
    T* p_u = reinterpret_cast<T*>(c_bf.ptr);
    T* p_out = reinterpret_cast<T*>(out_bf.ptr);

    // copy all data to CUDA
    T *g_x, *g_h, *g_c, *g_w, *g_u; 
    T *p_ysss, *p_csss;

    int64_t g_x_len = batch_size * seq_len * hidden_size;
    int64_t g_h_len = depth * batch_size * hidden_size;
    int64_t g_w_len = 4 * hidden_size * hidden_size;
    int64_t g_ysss_len = batch_size * depth * seq_len * hidden_size;

    cudaCheck(cudaMalloc(&g_x, g_x_len * sizeof(T)));
    cudaCheck(cudaMalloc(&g_h, g_h_len * sizeof(T)));
    cudaCheck(cudaMalloc(&g_c, g_h_len * sizeof(T)));
    cudaCheck(cudaMalloc(&g_w, g_w_len * sizeof(T)));
    cudaCheck(cudaMalloc(&g_u, g_w_len * sizeof(T)));
    cudaCheck(cudaMalloc(&p_ysss, g_ysss_len * sizeof(T)));
    cudaCheck(cudaMalloc(&p_csss, g_ysss_len * sizeof(T)));

    cudaMemcpy(g_x, p_x, g_x_len * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(g_h, p_h, g_h_len * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(g_c, p_c, g_h_len * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(g_w, p_w, g_w_len * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(g_u, p_u, g_w_len * sizeof(T), cudaMemcpyHostToDevice);

    // params for a lstmcell (TVM part)
    int64_t x_shape[2] = {batch_size, hidden_size};
    int64_t w_shape[2] = {4 * hidden_size, hidden_size};
  
    int64_t x_len = batch_size * hidden_size;
    int64_t w_len = 4 * hidden_size * hidden_size;

    int ndim = 2;
    int dtype_code = kDLFloat;
    int dtype_bits = 32;
    int dtype_lanes = 1;
    int device_type = kDLCUDA;
    int device_id = 0;

    // build TVMArray for cell call, alloc mem on CUDA
    DLTensor* x;
    DLTensor* h;
    DLTensor* c;
    DLTensor* w;
    DLTensor* u;
    DLTensor* ht;
    DLTensor* ct;

    TVMArrayAlloc(x_shape, ndim, dtype_code, dtype_bits, 
                dtype_lanes, device_type, device_id, &x);
    TVMArrayAlloc(x_shape, ndim, dtype_code, dtype_bits, 
                dtype_lanes, device_type, device_id, &h);
    TVMArrayAlloc(x_shape, ndim, dtype_code, dtype_bits, 
                dtype_lanes, device_type, device_id, &c);
    TVMArrayAlloc(w_shape, ndim, dtype_code, dtype_bits, 
                dtype_lanes, device_type, device_id, &w);
    TVMArrayAlloc(w_shape, ndim, dtype_code, dtype_bits, 
                dtype_lanes, device_type, device_id, &u);
    TVMArrayAlloc(x_shape, ndim, dtype_code, dtype_bits, 
                dtype_lanes, device_type, device_id, &ht);
    TVMArrayAlloc(x_shape, ndim, dtype_code, dtype_bits, 
                dtype_lanes, device_type, device_id, &ct);

    
    // ==========================region0=========================
    if (true) {
        int64_t gather_idx[batch_size * 2] = {0};
        for(int64_t i = 0; i < batch_size; i++)
            gather_idx[2 * i] = i;
        vector<int64_t> gather_idx_shape = {batch_size, 2};
        vector<int64_t> gather_out_shape = {batch_size, hidden_size};
        vector<int64_t> gather_data_shape = {batch_size, seq_len, hidden_size};

        gather_nd_cxx_cuda(static_cast<T*>(x->data), g_x, gather_idx,
                            gather_out_shape, gather_data_shape, gather_idx_shape);

        h->data = g_h;
        c->data = g_c;
        w->data = g_w;
        u->data = g_u;

        f(x, h, c, w, u, ht, ct);

        int64_t scatter_idx[batch_size * 3] = {0};
        for(int64_t i = 0; i < batch_size; i++)
            scatter_idx[3 * i] = i;
        vector<int64_t> scatter_idx_shape = {batch_size, 3};
        vector<int64_t> scatter_out_shape = {batch_size, depth, seq_len, hidden_size};
        vector<int64_t> scatter_update_shape = {batch_size, hidden_size};

        scatter_nd_cxx_cuda(p_ysss, static_cast<T*>(ht->data), scatter_idx,
                    scatter_out_shape, scatter_update_shape, scatter_idx_shape);
        scatter_nd_cxx_cuda(p_csss, static_cast<T*>(ct->data), scatter_idx,
                    scatter_out_shape, scatter_update_shape, scatter_idx_shape);
    }

    if (seq_len > 1){
        int64_t gather_idx[batch_size * (seq_len - 1) * 2] = {0};
        for(int64_t i = 1; i < seq_len; i++)
            for(int64_t j = 0; j < batch_size; j++){
                auto loc = 2 * ((i - 1) * batch_size + j);
                gather_idx[loc] = j;
                gather_idx[loc + 1] = i;
            }
        
        vector<int64_t> gather_idx_shape = {seq_len - 1, batch_size, 2};
        vector<int64_t> gather_out_shape = {seq_len - 1, batch_size, hidden_size};
        vector<int64_t> gather_data_shape = {batch_size, seq_len, hidden_size};
        
        T* gather_out;
        cudaCheck(cudaMalloc(&gather_out, (g_x_len - x_len) * sizeof(T)));
        gather_nd_cxx_cuda(gather_out, g_x, gather_idx,
                        gather_out_shape, gather_data_shape, gather_idx_shape);
        
        T *h_list, *c_list;
        cudaCheck(cudaMalloc(&h_list, g_x_len * sizeof(T)));
        cudaCheck(cudaMalloc(&c_list, g_x_len * sizeof(T)));
        
        int block = 512;
        int64_t n = batch_size * hidden_size;
        int64_t grid = (n + block - 1) / block;

        copykernel<T><<<grid, block, 0>>>(g_h, h_list, batch_size * hidden_size);
        copykernel<T><<<grid, block, 0>>>(g_c, c_list, batch_size * hidden_size);

        for(int i = 1; i < seq_len - 1; i++){
            x->data = &(gather_out[(i - 1) * batch_size * hidden_size]);
            h->data = &(h_list[(i - 1) * batch_size * hidden_size]);
            c->data = &(c_list[(i - 1) * batch_size * hidden_size]);
            // w, u keep still
            f(x, h, c, w, u, ht, ct);

            copykernel<T><<<grid, block, 0>>>(
                    static_cast<T*>(ht->data),
                    h_list + i * batch_size * hidden_size,
                    batch_size * hidden_size);
            copykernel<T><<<grid, block, 0>>>(
                    static_cast<T*>(ct->data),
                    c_list + i * batch_size * hidden_size,
                    batch_size * hidden_size);
        }
        
        int64_t scatter_idx[batch_size * (seq_len - 1) * 3] = {0};
        for(int64_t i = 1; i < seq_len; i++)
            for(int64_t j = 0; j < batch_size; j++){
                auto loc = 3 * ((i - 1) * batch_size + j);
                scatter_idx[loc] = j;
                scatter_idx[loc + 2] = i;
            }
        vector<int64_t> scatter_idx_shape = {seq_len - 1, batch_size, 3};
        vector<int64_t> scatter_out_shape = {batch_size, depth, seq_len, hidden_size};
        vector<int64_t> scatter_update_shape = {batch_size * (seq_len - 1), hidden_size};
        
        // FIXME: don't know why this scatter failed
        // scatter_nd_cxx_cuda(p_ysss, h_list + batch_size * hidden_size, scatter_idx,
        //             scatter_out_shape, scatter_update_shape, scatter_idx_shape);
        // std::cout << "hi" << std::endl;
        
        // scatter_nd_cxx_cuda(p_ysss, c_list + batch_size * hidden_size, scatter_idx,
        //             scatter_out_shape, scatter_update_shape, scatter_idx_shape);
        
    
        cudaCheck(cudaFree(gather_out));
        cudaCheck(cudaFree(h_list));
        cudaCheck(cudaFree(c_list));
    }
    //============================end==========================
    // missing a gather-like function
    // output share the same shape with xss
    // gather: (bs, dep, seq, hs) -> (bs, seq, hs)
    
    cudaCheck(cudaMemcpy(p_out, p_ysss,
                       x_len * seq_len * depth * sizeof(T), cudaMemcpyDeviceToHost));
    
    // 'error' happens even if disable all the free
    x->data = nullptr;
    h->data = nullptr;
    c->data = nullptr;
    w->data = nullptr;
    u->data = nullptr;
    ht->data = nullptr;
    ct->data = nullptr;
    
    TVMArrayFree(x);
    TVMArrayFree(h);
    TVMArrayFree(c);
    TVMArrayFree(w);
    TVMArrayFree(u);
    TVMArrayFree(ht);
    TVMArrayFree(ct);
    
    cudaCheck(cudaFree(g_x));
    cudaCheck(cudaFree(g_h));
    cudaCheck(cudaFree(g_c));
    cudaCheck(cudaFree(g_w));
    cudaCheck(cudaFree(g_u));
    cudaCheck(cudaFree(p_ysss));
    cudaCheck(cudaFree(p_csss));

    return;
}