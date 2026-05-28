#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/csrc/stable/device.h>

void completeness(torch::stable::Tensor& t, const torch::stable::Tensor& ct) {
    // Types
    torch::stable::Device dev(torch::headeronly::DeviceType::CPU);
    torch::headeronly::ScalarType st = torch::headeronly::ScalarType::Float;

    // data_ptr (mutable + const)
    float* p = t.mutable_data_ptr<float>();
    const float* cp = ct.const_data_ptr<float>();

    // Method to free function
    auto cl = torch::stable::clone(t);
    auto cg = torch::stable::contiguous(t);

    // Method rename
    auto dt = t.scalar_type();

    // nbytes
    auto nb = t.numel() * t.element_size();

    // sizes
    auto s = t.size(0);

    // Free function
    auto e = torch::stable::empty({3, 3}, torch::headeronly::ScalarType::Float);

    // Macro
    STD_TORCH_CHECK(t.dim() > 0, "bad");

    // nullopt + optional
    std::optional<torch::stable::Tensor> opt = std::nullopt;
}
