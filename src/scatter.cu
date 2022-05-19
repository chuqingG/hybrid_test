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
// #include "binding.cu"
#include "utils.h"
#include "tensor.h"
#include "kernel.h"
#pragma GCC diagnostic ignored "-Wformat="


// template <typename T>
// void scatter_nd(pybind11::array_t<T> output_pb, 
//                 pybind11::array_t<int64_t> indices_pb,
//                 pybind11::array_t<T> update_pb,
//                 pybind11::str log_path = "result.txt"){
//     pybind11::buffer_info output = output_pb.request();
//     pybind11::buffer_info indices = indices_pb.request();
//     pybind11::buffer_info update = update_pb.request();

//     auto idx_shape = indices.shape;
//     auto idx_shape_size = indices.ndim;

//     auto output_shape = output.shape;
//     auto output_shape_size = output.ndim;

//     int64_t each_idx_len = idx_shape[idx_shape_size - 1];
//     int64_t idx_number = 1;
//     for(int i = 0; i< idx_shape_size - 1; i++)
//         idx_number *= idx_shape[i];

//     int64_t each_update_size = 1;
//     for(int64_t i = each_idx_len; i < output_shape_size; i++)
//         each_update_size *= output_shape[i];

//     int block = 512;
//     int64_t n = each_update_size * idx_number;
//     int64_t grid = (n + block - 1) / block;

//     // init params on device
//     int64_t* g_output_shape;
//     cudaCheck(cudaMalloc(&g_output_shape, output_shape_size * sizeof(int64_t)));
//     cudaCheck(cudaMemcpy(g_output_shape, vec2ptr(output_shape),
//                 output_shape_size * sizeof(int64_t),
//                 cudaMemcpyHostToDevice));

//     // create tensor
//     T* p_output = reinterpret_cast<T*>(output.ptr);
//     T* p_update = reinterpret_cast<T*>(update.ptr);
//     int64_t* p_idx = (int64_t*)(indices.ptr);

//     testtest();

//     // std::cout << typeid(p_output).name() << std::endl;

//     Tensor<T> output_t(output_shape, p_output);
//     Tensor<T> update_t(update.shape, p_update);
//     Tensor<int64_t> idx_t(idx_shape, p_idx);

//     // open log_path
//     std::ofstream fs;
//     fs.open(log_path,std::ios::out|std::ios::app);

//     // bad scalability: {bs, dep, seq, hs}
//     std::string setting = "(dep, seq, bs, hs) = (" +
//                             std::to_string(output_shape[1]) + ", " +
//                             std::to_string(output_shape[2]) + ", " +
//                             std::to_string(output_shape[0]) + ", " +
//                             std::to_string(output_shape[3]) + ")";

//     output_t.print_data();

//     // timeKeep(1000, 10, setting, fs, 
//     //         call_scatter(grid, block, 
//     //                 update_t.data(), idx_t.data(), output_t.mutable_data(),
//     //                 g_output_shape, idx_number, each_update_size, each_idx_len));
//     ScatterNdCUDAKernel<float><<<grid, block, 0>>>(
//             update_t.data(), idx_t.data(), output_t.mutable_data(),
//             g_output_shape, idx_number, each_update_size, each_idx_len);
    
//     std::cout << "==" << std::endl;
//     // auto p = output_t.mutable_data();
    
//     // output_t.print_data();
//     // output_t.data_sync();
//     cudaMemcpy(p_output, output_t.mutable_data(),
//                        output_t.total_len() * sizeof(float), cudaMemcpyDeviceToHost);
//     std::cout << p_output[0] << std::endl;
//     fs.close();

// }

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

    // init params on device
    // int64_t* g_output_shape;
    // cudaCheck(cudaMalloc(&g_output_shape, output_shape_size * sizeof(int64_t)));
    // cudaCheck(cudaMemcpy(g_output_shape, vec2ptr(output_shape),
    //             output_shape_size * sizeof(int64_t),
    //             cudaMemcpyHostToDevice));

    // create tensor
    T* p_output = reinterpret_cast<T*>(output.ptr);
    T* p_update = reinterpret_cast<T*>(update.ptr);
    int64_t* p_idx = reinterpret_cast<int64_t*>(indices.ptr);

    int64_t output_len = reduceMul(output_shape);
    int64_t update_len = reduceMul(update.shape);
    int64_t indice_len = reduceMul(idx_shape);

    // std::cout << output_len << std::endl;

    T* g_output;
    T* g_update;
    int64_t* g_indice;

    int64_t* g_output_shape;
    cudaCheck(cudaMalloc(&g_output_shape, output_shape_size * sizeof(int64_t)));
    cudaCheck(cudaMemcpy(g_output_shape, vec2ptr(output_shape),
                output_shape_size * sizeof(int64_t),
                cudaMemcpyHostToDevice));

    std::cout << output_len << std::endl;
    cudaCheck(cudaMalloc(&g_output, output_len * sizeof(T)));
    // T* g_update;
    cudaCheck(cudaMalloc(&g_update, update_len * sizeof(T)));
    // int64_t* g_indice;
    cudaCheck(cudaMalloc(&g_indice, indice_len * sizeof(int64_t)));
    cudaCheck(cudaMemcpy(g_output, p_output,
                output_len * sizeof(T), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(g_update, p_update,
                update_len * sizeof(T), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(g_indice, p_idx,
                indice_len * sizeof(int64_t), cudaMemcpyHostToDevice));

    
    // open log_path
    // std::ofstream fs;
    // fs.open(log_path,std::ios::out|std::ios::app);

    // bad scalability: {bs, dep, seq, hs}
    // std::string setting = "(dep, seq, bs, hs) = (" +
    //                         std::to_string(output_shape[1]) + ", " +
    //                         std::to_string(output_shape[2]) + ", " +
    //                         std::to_string(output_shape[0]) + ", " +
    //                         std::to_string(output_shape[3]) + ")";

    // timeKeep(1000, 10, setting, fs, 
    //         call_scatter(grid, block, 
    //                 update_t.data(), idx_t.data(), output_t.mutable_data(),
    //                 g_output_shape, idx_number, each_update_size, each_idx_len));
    ScatterNdCUDAKernel<float><<<grid, block, 0>>>(
            g_update, g_indice, g_output,
            g_output_shape, idx_number, each_update_size, each_idx_len);
    
    cudaDeviceSynchronize();
    // output_t.print_data();
    // output_t.data_sync();
    cudaCheck(cudaMemcpy(p_output, g_output,
                       output_len * sizeof(T), cudaMemcpyDeviceToHost));
    std::cout << p_output[0] << std::endl;
    // fs.close();
    
    // cudaCheck(cudaFree(g_output));
    
}