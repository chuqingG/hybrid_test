cmake_minimum_required(VERSION 3.6)
add_definitions(-w)
set(TVM_PATH "/home/v-chuqinggao/tvm")

include_directories(
    ${PYTHON_INCLUDE_DIRS}
    /home/v-chuqinggao/pybind11/include
    ${TVM_PATH}/include
    ${TVM_PATH}/3rdparty/dmlc-core/include
    ${TVM_PATH}/3rdparty/dlpack/include
    )

link_directories(
    /usr/local/cuda/lib64
    ${TVM_PATH}
    ${TVM_PATH}/build
    )

find_package(CUDA REQUIRED)
find_package(PythonInterp 3.6 REQUIRED)
find_package(PythonLibs 3.6 REQUIRED)
# find_package(pybind11 REQUIRED)

# set(CMAKE_CXX_FLAGS 
#     "${CMAKE_CXX_FLAGS} -std=c++11 -Wno-unused-value -Wno-extra")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -w")
set(CMAKE_NVCC_FLAGS "${CMAKE_C_FLAGS} -w")
# pybind11_add_module(fastop src/tensor.h)

cuda_add_library(fastop SHARED
  src/binding.cu)


target_link_libraries(fastop
  # ${PYTHON_LIBRARIES}
  # cudart
  tvm
  cuda
  )

set_target_properties(fastop PROPERTIES PREFIX "")