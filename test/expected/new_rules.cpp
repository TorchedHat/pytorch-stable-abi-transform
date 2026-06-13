#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/csrc/stable/device.h>
#include <torch/headeronly/core/Dispatch.h>
#include <torch/headeronly/core/Dispatch_v2.h>
#include <torch/csrc/stable/library.h>

// Pattern 1: TORCH_LIBRARY → STABLE_TORCH_LIBRARY
STABLE_TORCH_LIBRARY(my_ops, m) {
    m.def("custom_op(Tensor input) -> Tensor");
}

// Pattern 2: TORCH_LIBRARY_IMPL → STABLE_TORCH_LIBRARY_IMPL
STABLE_TORCH_LIBRARY_IMPL(my_ops, CPU, m) {
    m.def("custom_op(Tensor input) -> Tensor");
}

// Pattern 3: AT_ERROR → STD_TORCH_CHECK(false, ...)
void check_error(const torch::stable::Tensor& t) {
    if (t.dim() != 2)
        STD_TORCH_CHECK(false, "expected 2D tensor, got ", t.dim(), "D");
}

// Pattern 4: .itemsize() → .element_size()
void use_itemsize(const torch::stable::Tensor& t) {
    auto bytes_per_elem = t.element_size();
}

// Pattern 5: .nbytes() → .numel() * .element_size()
void use_nbytes(const torch::stable::Tensor& t) {
    auto total_bytes = t.numel() * t.element_size();
}

// Pattern 6: AT_DISPATCH_FLOATING_TYPES → THO_DISPATCH_V2
void dispatch_floating(const torch::stable::Tensor& input) {
    THO_DISPATCH_V2(input.scalar_type(), "floating_kernel", [&] {
        auto* ptr = input.mutable_data_ptr<scalar_t>();
    }, AT_FLOATING_TYPES);
}
