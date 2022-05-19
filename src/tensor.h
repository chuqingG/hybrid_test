#pragma once

#include <iostream>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include "utils.h"

template <typename T>
class Tensor{
  private:
    T* data_;
    T* cudadata_;
    int total_len_;
    std::vector<int64_t> shapes_;
    std::string dtype_;
    std::string device_;
    // std::shared_ptr<double> sptr_;
    
  public:
    Tensor(std::vector<int64_t> shapes,
            T* ptr = nullptr,
            const std::string& dtype = "float32",
            const std::string device = "cuda") {
        // data_ = ptr;
        dtype_ = dtype;
        device_ = device;
        shapes_ = shapes;
        
        total_len_ = 1;
        for(auto l : shapes_)
            total_len_ *= l;

        data_ = new T[total_len_];
        
        // std::cout << typeid(ptr).name() << std::endl;

        if (ptr == nullptr){
            vector<T> v = create_vec<T>(shapes);
            memcpy(data_, &v[0], sizeof(v[0]) * v.size());
        }
        else 
            data_ = ptr;

        if (device_ == "cuda"){
            cudaMalloc(&cudadata_, Tensor::total_len_ * sizeof(T));
            cudaMemcpy(cudadata_, data_, 
                        Tensor::total_len_ * sizeof(T),
                        cudaMemcpyHostToDevice);
        }

    }
    ~Tensor() {
        if (device_ == "cuda")
            cudaFree(cudadata_);
    }


    const T* data() const {
        if (device_ == "cuda")
            return cudadata_;
        else
            return data_;
    }

    T* mutable_data(){
        if (device_ == "cuda")
            return cudadata_;
        else
            return data_;
    }

    std::vector<int64_t> shape() {
        return shapes_;
    }

    int64_t total_len() {
        return total_len_;
    }

    std::string dtype() {
        return dtype_;
    }

    void data_sync(){
        if (device_ == "cuda"){
            cudaMemcpy(data_, cudadata_,
                       total_len_ * sizeof(T), cudaMemcpyDeviceToHost);

            // std::cout << data_[0] << std::endl;
        }
    }

    void print_data(){
        int64_t* shape_l = vec2ptr<int64_t>(shapes_);
        T* ptr;
        if (device_ == "cuda")
            ptr = cudadata_;
        else
            ptr = data_;
        if (shapes_.size() == 3){
            int x = shape_l[0];
            int y = shape_l[1];
            int z = shape_l[2];
            printf("[");
            for(int i=0;i < x; i++){
                printf("[");
                for(int j = 0; j < y; j++){
                    printf("[");
                    for(int k = 0; k< z; k++)
                        print("%.4f, ", (float)data_[i*(y*z) + j*z + k]);
                    printf("],\n  ");
                }
                printf("],\n ");
            }
            printf("]\n");
        }
        else if (shapes_.size() == 2){
            int x = shape_l[0];
            int y = shape_l[1];
            printf("[");
            for(int i=0;i < x; i++){
                printf("[");
                for(int j = 0; j < y; j++)
                    print("%.4f, ", (float)data_[i*y + j]);
                printf("],\n ");
            }
            printf("]\n");
        }
        if (shapes_.size() == 4){
            int x = shape_l[0];
            int y = shape_l[1];
            int z = shape_l[2];
            int w = shape_l[3];
            printf("[");
            for(int i=0;i < x; i++){
                printf("[");
                for(int j = 0; j < y; j++){
                    printf("[");
                    for(int k = 0; k< z; k++){
                        printf("[");
                        for(int m = 0; m < w; m++)
                            print("%.4f, ", (float)data_[i*(y*z*w) + j*(z*w) + k*w + m]);
                        printf("],\n   ");
                    }   
                    printf("],\n  ");
                }
                printf("],\n ");
            }
            printf("]\n");
        }
    }

};

// Tensor::~Tensor(void){
//     cout << "bye" << endl;
// }




// class Tensor{
//   private:
//     float* data_;
//     float* cudadata_;
//     int total_len_;
//     std::initializer_list<int64_t> shapes_;
//     std::string dtype_;
//     std::string device_;
//     // std::shared_ptr<double> sptr_;
    
//   public:
//     Tensor(std::initializer_list<int64_t> shapes,
//             float* ptr = nullptr,
//             const std::string& dtype = "float32",
//             const std::string device = "cuda") {
//         // data_ = ptr;
//         dtype_ = dtype;
//         device_ = device;
//         shapes_ = shapes;
        
//         total_len_ = 1;
//         for(auto l : shapes_)
//             total_len_ *= l;

//         data_ = new float[total_len_];

//         if (ptr == nullptr){
//             vector<float> v = create_vec(shapes);
//             memcpy(data_, &v[0], sizeof(v[0]) * v.size());
//         }

//         if (device_ == "cuda"){
//             cudaMalloc(&cudadata_, Tensor::total_len_ * sizeof(float));
//             cudaMemcpy(cudadata_, data_, 
//                         Tensor::total_len_ * sizeof(float),
//                         cudaMemcpyHostToDevice);
//         }

//     }
//     ~Tensor();


//     const float* data() const {
//         if (device_ == "cuda")
//             return cudadata_;
//         else
//             return data_;
//     }

//     float* mutable_data(){
//         if (device_ == "cuda")
//             return cudadata_;
//         else
//             return data_;
//     }

//     std::initializer_list<int64_t> shape() {
//         return shapes_;
//     }

//     std::string dtype() {
//         return dtype_;
//     }

//     void print_data(){
//         int64_t* shape_l = ilist2list<int64_t>(shapes_);
//         if (shapes_.size() == 3){
//             int x = shape_l[0];
//             int y = shape_l[1];
//             int z = shape_l[2];
//             printf("[");
//             for(int i=0;i < x; i++){
//                 printf("[");
//                 for(int j = 0; j < y; j++){
//                     printf("[");
//                     for(int k = 0; k< z; k++)
//                         print("%.4f, ", data_[i*(y*z) + j*z + k]);
//                     printf("],\n  ");
//                 }
//                 printf("],\n ");
//             }
//             printf("]\n");
//         }
//         else if (shapes_.size() == 2){
//             int x = shape_l[0];
//             int y = shape_l[1];
//             printf("[");
//             for(int i=0;i < x; i++){
//                 printf("[");
//                 for(int j = 0; j < y; j++)
//                     print("%.4f, ", data_[i*y + j]);
//                 printf("],\n ");
//             }
//             printf("]\n");
//         }
//         if (shapes_.size() == 4){
//             int x = shape_l[0];
//             int y = shape_l[1];
//             int z = shape_l[2];
//             int w = shape_l[3];
//             printf("[");
//             for(int i=0;i < x; i++){
//                 printf("[");
//                 for(int j = 0; j < y; j++){
//                     printf("[");
//                     for(int k = 0; k< z; k++){
//                         printf("[");
//                         for(int m = 0; m < w; m++)
//                             print("%.4f, ", data_[i*(y*z*w) + j*(z*w) + k*w + m]);
//                         printf("],\n   ");
//                     }   
//                     printf("],\n  ");
//                 }
//                 printf("],\n ");
//             }
//             printf("]\n");
//         }
//     }

// };

// Tensor::~Tensor(void){
//     cout << "bye" << endl;
// }