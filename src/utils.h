#pragma once

#include <iomanip>
#include <memory>
#include <iostream>
#include <algorithm>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <string.h>
#include <cuda.h>
#include <cudnn.h>
#include <cuda_runtime_api.h>
#include "kernel.h"
#pragma GCC diagnostic ignored "-Wformat="

using namespace std;

#define print(str, var) printf(str, var)


#define RAND(low, high) \
    low + static_cast<float>(rand()) / static_cast<float>(RAND_MAX / (high - low))


#define cudaCheck(f) { \
  ::cudaError_t err = (f); \
  if (err != cudaSuccess) { \
    std::cout << #f ": " << cudaGetErrorString(err) << std::endl; \
    std::exit(1); \
  } \
}

#define arrLen(a) sizeof(a) / sizeof(a[0])

// it: iter_num
// w: warmup
// setting
// fs: fstream
// f: funcion to test
#define timeKeep(it, w, setting, fs, f) {\
  cudaEvent_t start, stop; \
  float timeForward; \
  for(size_t  i = 0; i < w; ++i) \
    f; \
  cudaCheck(cudaEventCreate(&start)); \
  cudaCheck(cudaEventCreate(&stop)); \
  cudaCheck(cudaEventRecord(start)); \
  for(size_t  i = 0; i < it; ++i) \
    f; \
  cudaCheck(cudaEventRecord(stop)); \
  cudaCheck(cudaEventSynchronize(stop)); \
  cudaCheck(cudaEventElapsedTime(&timeForward, start, stop)); \
  timeForward /= float(it);  \
  fs << "|" << setting << "|" << timeForward << "(ms) |||" << std::endl; \
}


void init_vector(vector<float> &vec, float low, float high){
    for(auto t: vec){
        t = RAND(low, high);
    }
    return;
}

template <typename T>
void print_vec(vector<T> &vec){
    for(auto t: vec)
        cout << t << ", ";
    cout << endl;
}

template <typename T>
vector<T> create_vec(std::vector<int64_t> sizes, 
                        float low=0.0, float high=1.0){
    vector<T> res;
    int total = 1;
    for(auto l: sizes)
        total *= l;
    for(int i = 0; i < total; i++)
        res.push_back(RAND(low, high));
    return res;
}

template <typename T>
T* ilist2list(std::initializer_list<T> &ilist){
    T* res = new T[ilist.size()];
    int i = 0;
    for(auto t: ilist)
        res[i++] = t;
    return res;
}


void call_scatter(int grid, int block, 
                const float* update, const int64_t* indices,
                float* output, const int64_t* output_dim,
                size_t remain, size_t slice_size, size_t end_size){
    ScatterNdCUDAKernel<float><<<grid, block, 0>>>(
        update, indices, output,
        output_dim, remain, slice_size, end_size);
}

template <typename T>
T* vec2ptr(std::vector<T> vec){
  int len = vec.size();
  T* p = new T(len);
  for(int i = 0; i < len; i++){
    p[i] = vec[i];
  }
  return p;
}

template <typename T>
T* vec2list(std::vector<T> vec){
  int len = vec.size();
  T p[len];
  for(int i = 0; i < len; i++){
    p[i] = vec[i];
  }
  return p;
}

void testtest(){
  std::cout << "hi" << std::endl;
}
