#pragma once

#include <sstream>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
// #include <pybind11/eigen.h>
#include <cuda_runtime.h>
#include "utils.h"
#include "tensor.h"
#include "kernel.h"
#pragma GCC diagnostic ignored "-Wformat="


//===============================no tensor version=====================

template <typename T>
void scatter_nd(pybind11::array_t<T> output_pb, 
                pybind11::array_t<int64_t> indices_pb,
                pybind11::array_t<T> update_pb){
    pybind11::buffer_info output = output_pb.request();
    pybind11::buffer_info indices = indices_pb.request();
    pybind11::buffer_info update = update_pb.request();

    auto idx_shape = indices.shape;
    auto idx_shape_size = indices.ndim;

    auto output_shape = output.shape;
    auto output_shape_size = output.ndim;

    int64_t each_idx_len = idx_shape[idx_shape_size - 1];
    int64_t idx_number = 1;
    for(int i = 0; i< idx_shape_size - 1; i++)
        idx_number *= idx_shape[i];

    int64_t each_update_size = 1;
    for(int64_t i = each_idx_len; i < output_shape_size; i++)
        each_update_size *= output_shape[i];

    int block = 512;
    int64_t n = each_update_size * idx_number;
    int64_t grid = (n + block - 1) / block;

    // create tensor
    T* p_output = reinterpret_cast<T*>(output.ptr);
    T* p_update = reinterpret_cast<T*>(update.ptr);
    int64_t* p_idx = reinterpret_cast<int64_t*>(indices.ptr);

    int64_t output_len = reduceMul(output_shape);
    int64_t update_len = reduceMul(update.shape);
    int64_t indice_len = reduceMul(idx_shape);

    T* g_output;
    T* g_update;
    int64_t* g_indice;
    int64_t* g_output_shape;

    // init params on device
    cudaCheck(cudaMalloc(&g_output_shape, output_shape_size * sizeof(int64_t)));
    cudaCheck(cudaMemcpy(g_output_shape, vec2ptr(output_shape),
                output_shape_size * sizeof(int64_t),
                cudaMemcpyHostToDevice));

    
    cudaCheck(cudaMalloc(&g_output, output_len * sizeof(T)));
    cudaCheck(cudaMalloc(&g_update, update_len * sizeof(T)));
    cudaCheck(cudaMalloc(&g_indice, indice_len * sizeof(int64_t)));
    cudaCheck(cudaMemcpy(g_output, p_output,
                output_len * sizeof(T), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(g_update, p_update,
                update_len * sizeof(T), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(g_indice, p_idx,
                indice_len * sizeof(int64_t), cudaMemcpyHostToDevice));
    
    // open log_path
    std::ofstream fs;
    fs.open("scatter.txt",std::ios::out|std::ios::app);

    // bad scalability: {bs, dep, seq, hs}
    std::string setting = "(dep, seq, bs, hs) = (" +
                            std::to_string(output_shape[1]) + ", " +
                            std::to_string(output_shape[2]) + ", " +
                            std::to_string(output_shape[0]) + ", " +
                            std::to_string(output_shape[3]) + ")" + "  scatter";

    timeKeep(1000, 10, setting, fs, 
            call_scatter<T>(grid, block, 
                    g_update, g_indice, g_output,
                    g_output_shape, idx_number, each_update_size, each_idx_len));
    // ScatterNdCUDAKernel<float><<<grid, block, 0>>>(
    //         g_update, g_indice, g_output,
    //         g_output_shape, idx_number, each_update_size, each_idx_len);
    
    cudaDeviceSynchronize();

    cudaCheck(cudaMemcpy(p_output, g_output,
                       output_len * sizeof(T), cudaMemcpyDeviceToHost));

    fs.close();
    // cudaCheck(cudaFree(g_output));
}


template <typename T>
void gather_nd(pybind11::array_t<T> output_pb, 
                pybind11::array_t<int64_t> indices_pb,
                pybind11::array_t<T> data_pb){

    pybind11::buffer_info output = output_pb.request();
    pybind11::buffer_info indices = indices_pb.request();
    pybind11::buffer_info data = data_pb.request();

    auto idx_shape = indices.shape;
    auto idx_shape_size = indices.ndim;

    auto output_shape = output.shape;
    // auto output_shape_size = output.ndim;

    auto data_shape = data.shape;
    auto data_shape_size = data.ndim;

    int64_t each_idx_len = idx_shape[idx_shape_size - 1];
    int64_t idx_number = 1;
    for(int i = 0; i< idx_shape_size - 1; i++)
        idx_number *= idx_shape[i];

    int64_t each_size = 1;
    for(int64_t i = each_idx_len; i < data_shape_size; i++)
        each_size *= data_shape[i];

    int block = 512;
    int64_t n = each_size * idx_number;
    int64_t grid = (n + block - 1) / block;

    // create tensor
    T* p_output = reinterpret_cast<T*>(output.ptr);
    T* p_data = reinterpret_cast<T*>(data.ptr);
    int64_t* p_idx = reinterpret_cast<int64_t*>(indices.ptr);

    int64_t output_len = reduceMul(output_shape);
    int64_t data_len = reduceMul(data_shape);
    int64_t indice_len = reduceMul(idx_shape);

    T* g_output;
    T* g_data;
    int64_t* g_indice;
    int64_t* g_data_shape;

    // init params on device
    cudaCheck(cudaMalloc(&g_data_shape, data_shape_size * sizeof(int64_t)));
    cudaCheck(cudaMemcpy(g_data_shape, vec2ptr(data_shape),
                data_shape_size * sizeof(int64_t),
                cudaMemcpyHostToDevice));

    
    cudaCheck(cudaMalloc(&g_output, output_len * sizeof(T)));
    cudaCheck(cudaMalloc(&g_data, data_len * sizeof(T)));
    cudaCheck(cudaMalloc(&g_indice, indice_len * sizeof(int64_t)));
    cudaCheck(cudaMemcpy(g_output, p_output,
                output_len * sizeof(T), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(g_data, p_data,
                data_len * sizeof(T), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(g_indice, p_idx,
                indice_len * sizeof(int64_t), cudaMemcpyHostToDevice));
    
    // open log_path
    std::ofstream fs;
    fs.open("scatter.txt",std::ios::out|std::ios::app);

    // bad scalability: {bs, dep, seq, hs}
    std::string setting = "(seq, bs, hs) = (" +
                            std::to_string(data_shape[1]) + ", " +
                            std::to_string(data_shape[0]) + ", " +
                            std::to_string(data_shape[2]) + ")" + "  gather";

    timeKeep(1000, 10, setting, fs, 
            call_gather<T>(grid, block, 
                    g_data, g_indice, g_output,
                    g_data_shape, idx_number, each_size, each_idx_len));
    // GatherNdCUDAKernel<float><<<grid, block, 0>>>(
    //         g_data, g_data_shape, g_indice, g_output,
    //         idx_number, each_size, each_idx_len);
    
    cudaDeviceSynchronize();

    cudaCheck(cudaMemcpy(p_output, g_output,
                       output_len * sizeof(T), cudaMemcpyDeviceToHost));

    fs.close();
    // cudaCheck(cudaFree(g_output));
}


/** 
 * assume that at least p_output and p_data exist on CUDA 
 * even use template<T>, this function exactly only works for float32 
 * "return": g_output, still on cuda
 */
template <typename T>
void gather_nd_cxx_cuda(T* g_output, T* g_data, int64_t* p_idx,
                    vector<int64_t> &output_shape,
                    vector<int64_t> &data_shape,
                    vector<int64_t> &idx_shape){

    auto idx_shape_size = idx_shape.size();
    auto data_shape_size = data_shape.size();

    int64_t each_idx_len = idx_shape[idx_shape_size - 1];
    int64_t idx_number = 1;
    for(int i = 0; i< idx_shape_size - 1; i++)
        idx_number *= idx_shape[i];

    int64_t each_size = 1;
    for(int64_t i = each_idx_len; i < data_shape_size; i++)
        each_size *= data_shape[i];

    int block = 512;
    int64_t n = each_size * idx_number;
    int64_t grid = (n + block - 1) / block;

    int64_t indice_len = reduceMul(idx_shape);

    int64_t* g_indice;
    int64_t* g_data_shape;

    // init params on device
    cudaCheck(cudaMalloc(&g_data_shape, data_shape_size * sizeof(int64_t)));
    cudaCheck(cudaMemcpy(g_data_shape, vec2ptr(data_shape),
                data_shape_size * sizeof(int64_t),
                cudaMemcpyHostToDevice));

    cudaCheck(cudaMalloc(&g_indice, indice_len * sizeof(int64_t)));
    cudaCheck(cudaMemcpy(g_indice, p_idx,
                indice_len * sizeof(int64_t), cudaMemcpyHostToDevice));
    
    GatherNdCUDAKernel<float><<<grid, block, 0>>>(
            g_data, g_data_shape, g_indice, g_output,
            idx_number, each_size, each_idx_len);
    
    cudaDeviceSynchronize();
    // cudaCheck(cudaFree(g_output));
}


/** 
 * assume that at least p_output and p_update exist on CUDA 
 * even use template<T>, this function exactly only works for float32 
 * "return": g_output, still on cuda
 */
template <typename T>
void scatter_nd_cxx_cuda(T* g_output, T* g_update, int64_t* p_idx,
                    vector<int64_t> &output_shape,
                    vector<int64_t> &update_shape,
                    vector<int64_t> &idx_shape){
    
    auto idx_shape_size = idx_shape.size();
    auto output_shape_size = output_shape.size();

    int64_t each_idx_len = idx_shape[idx_shape_size - 1];
    int64_t idx_number = 1;
    for(int i = 0; i< idx_shape_size - 1; i++)
        idx_number *= idx_shape[i];

    int64_t each_update_size = 1;
    for(int64_t i = each_idx_len; i < output_shape_size; i++)
        each_update_size *= output_shape[i];

    int block = 512;
    int64_t n = each_update_size * idx_number;
    int64_t grid = (n + block - 1) / block;

    int64_t indice_len = reduceMul(idx_shape);

    int64_t* g_indice;
    int64_t* g_output_shape;

    // init params on device
    cudaCheck(cudaMalloc(&g_output_shape, output_shape_size * sizeof(int64_t)));
    cudaCheck(cudaMemcpy(g_output_shape, vec2ptr(output_shape),
                output_shape_size * sizeof(int64_t),
                cudaMemcpyHostToDevice));
    
    cudaCheck(cudaMalloc(&g_indice, indice_len * sizeof(int64_t)));
    cudaCheck(cudaMemcpy(g_indice, p_idx,
                indice_len * sizeof(int64_t), cudaMemcpyHostToDevice));
    
    ScatterNdCUDAKernel<float><<<grid, block, 0>>>(
            g_update, g_indice, g_output,
            g_output_shape, idx_number, each_update_size, each_idx_len);
    
    cudaDeviceSynchronize();
    cudaCheck(cudaFree(g_output_shape));
    cudaCheck(cudaFree(g_indice));
}