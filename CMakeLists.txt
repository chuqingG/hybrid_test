cmake_minimum_required(VERSION 3.6)

include_directories(
    ${PYTHON_INCLUDE_DIRS}
    /home/v-chuqinggao/pybind11/include
    )

link_directories(
    /usr/local/cuda/lib64
    )

find_package(CUDA REQUIRED)
find_package(PythonInterp 3.6 REQUIRED)
find_package(PythonLibs 3.6 REQUIRED)
# find_package(pybind11 REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# pybind11_add_module(fastop src/tensor.h)

cuda_add_library(fastop SHARED
  src/binding.cu)

# add_executable(
#   tensor
#   src/tensor.h
# )

target_link_libraries(fastop
  ${PYTHON_LIBRARIES}
  cudart)

set_target_properties(fastop PROPERTIES PREFIX "")