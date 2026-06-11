#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/csrc/stable/device.h>
#include <cuda_runtime.h>

void kernel(torch::stable::Tensor& out, const torch::stable::Tensor& input) {
    STD_TORCH_CHECK(input.is_contiguous(), "must be contiguous");
    const torch::stable::accelerator::DeviceGuard device_guard(input.get_device_index());
    StreamHandle stream_handle = nullptr;
    aoti_torch_get_current_stream(input.get_device_index(), &stream_handle);
    const cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
    float* ptr = input.mutable_data_ptr<float>();
    auto result = torch::stable::clone(input);
}
