// #include "../../tvm/src/runtime/cuda/cuda_device_api.cc"
// #include "../../tvm/src/runtime/cuda/cuda_module.cc"

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
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <memory>

#include "utils.h"

#pragma GCC diagnostic ignored "-Wformat="

void initArray(DLTensor* input, const size_t size) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(-1., 1.);

  for (size_t i = 0; i < size; ++i)
    static_cast<float*>(input->data)[i] = dist(gen);
}

constexpr int64_t seq_len = 100;
constexpr int64_t batch_size = 128;
constexpr int64_t input_size = 512;
constexpr int64_t hidden_size = 512;

// void lstm_cell() {
//   
//   This function works only if the .so built from te
//   
  // DLDevice dev{kDLCUDA, 0};
  // // DLDevice dev{kDLCPU, 0};
  // std::string lib_path = "lib/cell_b128_h128.so";

  // bool enabled = tvm::runtime::RuntimeEnabled("cuda");
  
  // const tvm::runtime::PackedFunc* graph_executor_create =
  //     tvm::runtime::Registry::Get("tvm.graph_executor.create");
  
  // auto lib = tvm::runtime::Module::LoadFromFile(lib_path);
  // tvm::runtime::PackedFunc f = lib.GetFunction("cell");
  // ICHECK(f != nullptr);

  // DLTensor* x;
  // DLTensor* h;
  // DLTensor* c;
  // DLTensor* w;
  // DLTensor* u;
  // DLTensor* ht;
  // DLTensor* ct;

  // int ndim = 2;
  // int dtype_code = kDLFloat;
  // int dtype_bits = 32;
  // int dtype_lanes = 1;
  // int device_type = kDLCPU;
  // int device_id = 0;

  // int64_t input_shape[2] = {batch_size, input_size};
  // DLTensor* x;
  // TVMArrayAlloc(input_shape, 2 /*ndim*/, dtype_code, dtype_bits, dtype_lanes,
  //               device_type, device_id, &x);
  // initArray(x, batch_size * input_size);

  // int64_t state_shape[2] = {batch_size, hidden_size};
  // DLTensor* h_prev;
  // TVMArrayAlloc(state_shape, 2 /*hdim*/, dtype_code, dtype_bits, dtype_lanes,
  //               device_type, device_id, &h_prev);
  // initArray(h_prev, batch_size * hidden_size);

  // DLTensor* c_prev;
  // TVMArrayAlloc(state_shape, 2 /*hdim*/, dtype_code, dtype_bits, dtype_lanes,
  //               device_type, device_id, &c_prev);
  // initArray(c_prev, batch_size * hidden_size);

  // int64_t output_shape[2] = {batch_size, hidden_size};
  // DLTensor* y;
  // TVMArrayAlloc(output_shape, 2 /*hdim*/, dtype_code, dtype_bits, dtype_lanes,
  //               device_type, device_id, &y);
  // initArray(y, batch_size * hidden_size);

  // set_input("input", x);
  // set_input("h_prev", h_prev);
  // set_input("c_prev", c_prev);

  // run();
  // get_output(0, y);

  // TVMArrayFree(x);
  // TVMArrayFree(h_prev);
  // TVMArrayFree(c_prev);
  // TVMArrayFree(y);
// }

// int main() {
//   lstm_cell();
//   return 0;
// }

template <typename T>
void lstm_cell(pybind11::array_t<T> x_pb,
                pybind11::array_t<T> h_pb,
                pybind11::array_t<T> c_pb,
                pybind11::array_t<T> w_pb,
                pybind11::array_t<T> u_pb,
                pybind11::array_t<T> ht_pb,
                pybind11::array_t<T> ct_pb){
  //This function only works for .so built from te

  //batchsize, hidden_size can be known from x.shape = [bs, hs]
  
  pybind11::buffer_info x_bf = x_pb.request();
  pybind11::buffer_info h_bf = h_pb.request();
  pybind11::buffer_info c_bf = c_pb.request();
  pybind11::buffer_info w_bf = w_pb.request();
  pybind11::buffer_info u_bf = u_pb.request();
  pybind11::buffer_info ht_bf = ht_pb.request();
  pybind11::buffer_info ct_bf = ct_pb.request();

  int64_t batch_size = x_bf.shape[0];
  int64_t hidden_size = x_bf.shape[1];

  DLDevice dev{kDLCUDA, 0};
  // DLDevice dev{kDLCPU, 0};
  // TODO: change to params here
  std::string lib_path = "lib/cell_b128_h128.so";

  bool enabled = tvm::runtime::RuntimeEnabled("cuda");
  
  const tvm::runtime::PackedFunc* graph_executor_create =
      tvm::runtime::Registry::Get("tvm.graph_executor.create");
  
  auto lib = tvm::runtime::Module::LoadFromFile(lib_path);
  tvm::runtime::PackedFunc f = lib.GetFunction("cell");
  ICHECK(f != nullptr);

  DLTensor* x;
  DLTensor* h;
  DLTensor* c;
  DLTensor* w;
  DLTensor* u;
  DLTensor* ht;
  DLTensor* ct;

  int ndim = 2;
  int dtype_code = kDLFloat;
  int dtype_bits = 32;
  int dtype_lanes = 1;
  int device_type = kDLCUDA;
  int device_id = 0;

  int64_t x_shape[2] = {batch_size, hidden_size};
  int64_t w_shape[2] = {4 * hidden_size, hidden_size};
  
  auto x_len = batch_size * hidden_size;
  auto w_len = 4 * hidden_size * hidden_size;

  std::cout << x_shape[0] << " " << x_shape[1] << std::endl;

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

  T* p_x = reinterpret_cast<T*>(x_bf.ptr);
  T* p_h = reinterpret_cast<T*>(h_bf.ptr);
  T* p_c = reinterpret_cast<T*>(c_bf.ptr);
  T* p_w = reinterpret_cast<T*>(w_bf.ptr);
  T* p_u = reinterpret_cast<T*>(c_bf.ptr);
  T* p_ht = reinterpret_cast<T*>(ht_bf.ptr);
  T* p_ct = reinterpret_cast<T*>(ct_bf.ptr);

  cudaMemcpy(static_cast<T*>(x->data), p_x, x_len * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(static_cast<T*>(h->data), p_h, x_len * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(static_cast<T*>(c->data), p_c, x_len * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(static_cast<T*>(w->data), p_w, w_len * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(static_cast<T*>(u->data), p_u, w_len * sizeof(T), cudaMemcpyHostToDevice);

  f(x, h, c, w, u, ht, ct);
  cudaMemcpy(p_ht, static_cast<T*>(ht->data), x_len * sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpy(p_ct, static_cast<T*>(ct->data), x_len * sizeof(T), cudaMemcpyDeviceToHost);
  
  TVMArrayFree(x);
  TVMArrayFree(h);
  TVMArrayFree(c);
  TVMArrayFree(w);
  TVMArrayFree(u);
  TVMArrayFree(ht);
  TVMArrayFree(ct);
  return;
}