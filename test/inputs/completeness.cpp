#include <torch/all.h>

void completeness(at::Tensor& t, const at::Tensor& ct) {
    // Types
    c10::Device dev(at::kCPU);
    at::ScalarType st = at::kFloat;

    // data_ptr (mutable + const)
    float* p = t.data_ptr<float>();
    const float* cp = ct.data_ptr<float>();

    // Method to free function
    auto cl = t.clone();
    auto cg = t.contiguous();

    // Method rename
    auto dt = t.dtype();

    // nbytes
    auto nb = t.nbytes();

    // sizes
    auto s = t.sizes()[0];

    // Free function
    auto e = at::empty({3, 3}, at::kFloat);

    // Macro
    TORCH_CHECK(t.dim() > 0, "bad");

    // nullopt + optional
    c10::optional<at::Tensor> opt = c10::nullopt;
}
