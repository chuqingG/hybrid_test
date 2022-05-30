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
#include <curand.h>
#include "utils.h"
#include "scatter.cu"
// #include "lstm.cu"

#pragma GCC diagnostic ignored "-Wformat="


void myprint(float* data, int x, int y, int z, int w){
    printf("[");
    for(int i=0;i < x; i++){
        printf("[");
        for(int j = 0; j < y; j++){
            printf("[");
            for(int k = 0; k< 10; k++){
                printf("[");
                for(int m = 0; m < 10; m++)
                    print("%.4f, ", data[i*(y*z*w) + j*(z*w) + k*w + m]);
                printf("],\n   ");
            }   
            printf("],\n  ");
        }
        printf("],\n ");
    }
    printf("]\n");
}

void RandomData(float* input, int length, float mean = 0, float stddev = 0.1) {
  curandGenerator_t prng;
  curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

  curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long)clock());
  curandGenerateNormal(prng, input, length, mean, stddev);
}

template <typename T>
void lstm_cxx(pybind11::array_t<T> out_pb, int64_t depth, int64_t seq_len, int64_t batch_size, int64_t hidden_size){
    
    pybind11::buffer_info out_bf = out_pb.request();

    std::string lib_path = "lib/cell_b" + std::to_string(batch_size) + 
                            "_h" + std::to_string(hidden_size) + ".so";
    std::cout << lib_path << std::endl;

    bool enabled = tvm::runtime::RuntimeEnabled("cuda");
    const tvm::runtime::PackedFunc* graph_executor_create =
                tvm::runtime::Registry::Get("tvm.graph_executor.create");
    
    auto lib = tvm::runtime::Module::LoadFromFile(lib_path);
    tvm::runtime::PackedFunc f = lib.GetFunction("cell");
    ICHECK(f != nullptr);

    T* p_out = reinterpret_cast<T*>(out_bf.ptr);

    // // copy all data to CUDA
    T *g_x, *g_h, *g_c, *g_w, *g_u; 
    T *p_ysss, *p_csss;

    int64_t g_x_len = batch_size * seq_len * hidden_size;
    int64_t g_h_len = depth * batch_size * hidden_size;
    int64_t g_w_len = depth * 4 * hidden_size * hidden_size;
    int64_t g_ysss_len = batch_size * depth * seq_len * hidden_size;

    cudaCheck(cudaMalloc(&g_x, g_x_len * sizeof(T)));
    cudaCheck(cudaMalloc(&g_h, g_h_len * sizeof(T)));
    cudaCheck(cudaMalloc(&g_c, g_h_len * sizeof(T)));
    cudaCheck(cudaMalloc(&g_w, g_w_len * sizeof(T)));
    cudaCheck(cudaMalloc(&g_u, g_w_len * sizeof(T)));
    cudaCheck(cudaMalloc(&p_ysss, g_ysss_len * sizeof(T)));
    cudaCheck(cudaMalloc(&p_csss, g_ysss_len * sizeof(T)));

    RandomData(g_x, g_x_len);
    RandomData(g_h, g_h_len);
    RandomData(g_c, g_h_len);
    RandomData(g_w, g_w_len);
    RandomData(g_u, g_w_len);

    // params for a lstmcell (TVM part)
    int64_t x_shape[2] = {batch_size, hidden_size};
    int64_t w_shape[2] = {4 * hidden_size, hidden_size};
  
    int64_t x_len = batch_size * hidden_size;
    // int64_t w_len = 4 * hidden_size * hidden_size;

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
        // std::cout << "hi1" << std::endl;
        // scatter_nd_cxx_cuda(p_csss, static_cast<T*>(ct->data), scatter_idx,
        //             scatter_out_shape, scatter_update_shape, scatter_idx_shape);
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
        scatter_nd_cxx_cuda(p_ysss, h_list + batch_size * hidden_size, scatter_idx,
                    scatter_out_shape, scatter_update_shape, scatter_idx_shape);

        // scatter_nd_cxx_cuda(p_csss, c_list + batch_size * hidden_size, scatter_idx1,
        //             scatter_out_shape1, scatter_update_shape1, scatter_idx_shape1);
        
    
        cudaCheck(cudaFree(gather_out));
        cudaCheck(cudaFree(h_list));
        cudaCheck(cudaFree(c_list));
    }

    if (depth > 1){

        int64_t gather_idx[batch_size * 2] = {0};
        for(int64_t i = 0; i < batch_size; i++)
            gather_idx[2 * i] = i;
        vector<int64_t> gather_idx_shape = {batch_size, 2};
        vector<int64_t> gather_out_shape = {batch_size, hidden_size};
        vector<int64_t> gather_data_shape = {batch_size, seq_len, hidden_size};
        
        T* gather_out;
        cudaCheck(cudaMalloc(&gather_out, (batch_size * hidden_size) * sizeof(T)));
        gather_nd_cxx_cuda(gather_out, g_x, gather_idx,
                        gather_out_shape, gather_data_shape, gather_idx_shape);
        
        T *h_list, *c_list;
        cudaCheck(cudaMalloc(&h_list, g_h_len * sizeof(T)));
        cudaCheck(cudaMalloc(&c_list, g_h_len * sizeof(T)));
        
        int block = 512;
        int64_t n = batch_size * hidden_size;
        int64_t grid = (n + block - 1) / block;

        // std::ofstream fs;
        // fs.open("scatter.txt",std::ios::out|std::ios::app);

        // std::string setting = "(bs, hs) = (" +
        //                         std::to_string(batch_size) + ", " +
        //                         std::to_string(hidden_size) + ")" + "  copy";

        // timeKeep(1000, 10, setting, fs, 
        //     call_copy<T>(grid, block, 
        //             g_h, h_list, batch_size * hidden_size));
        // fs.close();
        copykernel<T><<<grid, block, 0>>>(g_h, h_list, batch_size * hidden_size);
        // copykernel<T><<<grid, block, 0>>>(g_c, c_list, batch_size * hidden_size);
        

        for(int i = 1; i < depth; i++){
            x->data = &(h_list[(i - 1) * batch_size * hidden_size]);
            h->data = g_h + i * batch_size * hidden_size;
            c->data = g_c + i * batch_size * hidden_size;
            w->data = g_w + i * 4 * hidden_size * hidden_size;
            u->data = g_u + i * 4 * hidden_size * hidden_size;
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
        
        int64_t scatter_idx[batch_size * (depth - 1) * 3] = {0};
        for(int64_t i = 1; i < depth; i++)
            for(int64_t j = 0; j < batch_size; j++){
                auto loc = 3 * ((i - 1) * batch_size + j);
                scatter_idx[loc] = j;
                scatter_idx[loc + 1] = i;
            }
        vector<int64_t> scatter_idx_shape = {depth - 1, batch_size, 3};
        vector<int64_t> scatter_out_shape = {batch_size, depth, seq_len, hidden_size};
        vector<int64_t> scatter_update_shape = {batch_size * (depth - 1), hidden_size};
        
        // FIXME: don't know why the second scatter failed
        scatter_nd_cxx_cuda(p_ysss, h_list + batch_size * hidden_size, scatter_idx,
                    scatter_out_shape, scatter_update_shape, scatter_idx_shape);

        // scatter_nd_cxx_cuda(p_csss, c_list + batch_size * hidden_size, scatter_idx1,
        //             scatter_out_shape1, scatter_update_shape1, scatter_idx_shape1);
        
    
        cudaCheck(cudaFree(gather_out));
        cudaCheck(cudaFree(h_list));
        cudaCheck(cudaFree(c_list));
    }
    // FIXME(chuqing): if we stop here, we may find that something wrong with the scatter output
    //                 the last two lines are non-zero
    if ((depth > 1) && (seq_len > 1)) {
        //TODO(chuqing): mm -> bmm 

        int block = 512;
        int64_t n = batch_size * hidden_size;
        int64_t grid = (n + block - 1) / block;

        for(int64_t m = 2; m < seq_len + depth - 1; m++){

            int64_t low = (m - seq_len + 1 > 1) ? m - seq_len + 1 : 1;
            int64_t high = depth > m ? m : depth;

            int64_t scatter_idx[batch_size * (high - low) * 3] = {0};

            T *h_list, *c_list;
            cudaCheck(cudaMalloc(&h_list, (high - low) * batch_size * hidden_size * sizeof(T)));
            cudaCheck(cudaMalloc(&c_list, (high - low) * batch_size * hidden_size * sizeof(T)));

            for(int64_t p = low; p < high; p++){

                int64_t x_gather_idx[batch_size * 3] = {0};
                int64_t h_gather_idx[batch_size * 3] = {0};

                for(int64_t n = 0; n < batch_size; n++){
                    int64_t i = n;
                    int64_t j = p;
                    int64_t k = m - p;

                    x_gather_idx[3 * n] = i;
                    x_gather_idx[3 * n + 1] = j - 1;
                    x_gather_idx[3 * n + 2] = k;

                    h_gather_idx[3 * n] = i;
                    h_gather_idx[3 * n + 1] = j;
                    h_gather_idx[3 * n + 2] = k - 1;

                    int64_t loc = 3 * (p * batch_size + n);
                    scatter_idx[loc] = i;
                    scatter_idx[loc + 1] = j;
                    scatter_idx[loc + 2] = k;
                }

                vector<int64_t> gather_idx_shape = {batch_size, 3};
                vector<int64_t> gather_out_shape = {batch_size, hidden_size};
                vector<int64_t> gather_data_shape = {batch_size, depth, seq_len, hidden_size};
                
                T *x_gather_out, *h_gather_out, *c_gather_out;
                cudaCheck(cudaMalloc(&x_gather_out, (batch_size * hidden_size) * sizeof(T)));
                cudaCheck(cudaMalloc(&h_gather_out, (batch_size * hidden_size) * sizeof(T)));
                cudaCheck(cudaMalloc(&c_gather_out, (batch_size * hidden_size) * sizeof(T)));
                gather_nd_cxx_cuda(x_gather_out, p_ysss, x_gather_idx,
                        gather_out_shape, gather_data_shape, gather_idx_shape);
                gather_nd_cxx_cuda(h_gather_out, p_ysss, h_gather_idx,
                        gather_out_shape, gather_data_shape, gather_idx_shape);
                gather_nd_cxx_cuda(c_gather_out, p_csss, h_gather_idx,
                        gather_out_shape, gather_data_shape, gather_idx_shape);
                
                x->data = x_gather_out;
                h->data = h_gather_out;
                c->data = c_gather_out;
                w->data = g_w + p * 4 * hidden_size * hidden_size;
                u->data = g_u + p * 4 * hidden_size * hidden_size;

                copykernel<T><<<grid, block, 0>>>(
                    static_cast<T*>(ht->data),
                    h_list + p * batch_size * hidden_size,
                    batch_size * hidden_size);
                copykernel<T><<<grid, block, 0>>>(
                    static_cast<T*>(ct->data),
                    c_list + p * batch_size * hidden_size,
                    batch_size * hidden_size);

                cudaCheck(cudaFree(x_gather_out));
                cudaCheck(cudaFree(h_gather_out));
                cudaCheck(cudaFree(c_gather_out));
            }

            vector<int64_t> scatter_idx_shape = {high - low, batch_size, 3};
            vector<int64_t> scatter_out_shape = {batch_size, depth, seq_len, hidden_size};
            vector<int64_t> scatter_update_shape = {batch_size * (high - low), hidden_size};
            
            // FIXME: don't know why the second scatter failed
            scatter_nd_cxx_cuda(p_ysss, h_list + batch_size * hidden_size, scatter_idx,
                        scatter_out_shape, scatter_update_shape, scatter_idx_shape);

            // scatter_nd_cxx_cuda(p_csss, c_list + batch_size * hidden_size, scatter_idx,
            //             scatter_out_shape, scatter_update_shape, scatter_idx_shape);

            cudaCheck(cudaFree(h_list));
            cudaCheck(cudaFree(c_list));
        }
    }

    // //============================end==========================
    // // missing a gather-like function
    // // output share the same shape with xss
    // // gather: (bs, dep, seq, hs) -> (bs, seq, hs)
    
    cudaCheck(cudaMemcpy(p_out, p_ysss,
                       g_ysss_len * sizeof(T), cudaMemcpyDeviceToHost));
    
    cudaDeviceSynchronize();
    myprint(p_out, batch_size, depth, seq_len, hidden_size);
    // // 'error' happens even if disable all the free
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