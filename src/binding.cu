#include <sstream>
#include <iostream>
#include <cuda_runtime.h>
// #include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>
#include "multiply.cu"
#include "scatter.cu"
#include "tensor.h"



void bind_multiply_vector(pybind11::module &m) {
    m.def("multiply_vector", map_array<double>, 
           "dot product of vector x and vector y (CUDA)"
           );
}


void bind_scatter_nd(pybind11::module &m) {
    m.def("scatter_nd", scatter_nd<float>, 
           "scatter_nd(output, indices, update)"
           );
}


template <typename T>
void bind_tensor(pybind11::module &m) {
    pybind11::class_<Tensor<T>>(m, "Tensor")
        .def(pybind11::init<std::vector<int64_t>, T*, const std::string&, const std::string>())
        .def("data", &Tensor<T>::data)
        .def("shape", &Tensor<T>::shape)
        .def("dtype", &Tensor<T>::dtype);
}


PYBIND11_MODULE(fastop, m) {
    m.doc() = R"pbdoc(
        module fastop
        -----------------------
        .. Operators in CUDA
        .. functions::
           :toctree: multiply_vector

    )pbdoc";

    bind_multiply_vector(m);
    bind_scatter_nd(m);
    // bind_tensor<float>(m);
    // bind_tensor<int64_t>(m);

}