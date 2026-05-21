#include <torch/all.h>

void kernel(torch::Tensor& out, const at::Tensor& input) {
    float* ptr = input.data_ptr<float>();
    auto result = input.clone();
}

void process_refs(c10::ArrayRef<int64_t> sizes, c10::IntArrayRef strides) {
    c10::string_view name = "test";
    (void)sizes;
    (void)strides;
    (void)name;
}
