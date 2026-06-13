#include <torch/all.h>
#include <ATen/Dispatch.h>
#include <torch/library.h>

// Pattern 1: TORCH_LIBRARY → STABLE_TORCH_LIBRARY
TORCH_LIBRARY(my_ops, m) {
    m.def("custom_op(Tensor input) -> Tensor");
}

// Pattern 2: TORCH_LIBRARY_IMPL → STABLE_TORCH_LIBRARY_IMPL
TORCH_LIBRARY_IMPL(my_ops, CPU, m) {
    m.def("custom_op(Tensor input) -> Tensor");
}

// Pattern 3: AT_ERROR → STD_TORCH_CHECK(false, ...)
void check_error(const at::Tensor& t) {
    if (t.dim() != 2)
        AT_ERROR("expected 2D tensor, got ", t.dim(), "D");
}

// Pattern 4: .itemsize() → .element_size()
void use_itemsize(const at::Tensor& t) {
    auto bytes_per_elem = t.itemsize();
}

// Pattern 5: .nbytes() → .numel() * .element_size()
void use_nbytes(const at::Tensor& t) {
    auto total_bytes = t.nbytes();
}

// Pattern 6: AT_DISPATCH_FLOATING_TYPES → THO_DISPATCH_V2
void dispatch_floating(const at::Tensor& input) {
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "floating_kernel", [&] {
        auto* ptr = input.data_ptr<scalar_t>();
    });
}
