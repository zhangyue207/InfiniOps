#include "aclrtlaunch_AddRmsNorm.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_kernel_helper.h"

namespace ascend::detail {

// Values mirror `infini::ops::DataType` for AscendC kernel dispatch.
constexpr int64_t kInfiniDataTypeFloat16 = 8;
constexpr int64_t kInfiniDataTypeFloat32 = 10;

std::vector<at::Tensor> AddRmsNorm(const at::Tensor& input,
                                   const at::Tensor& residual,
                                   const at::Tensor& weight, double eps) {
  // Input validation.
  TORCH_CHECK(input.dim() > 0,
              "`AddRmsNorm`: `input` must have at least 1 dimension");
  TORCH_CHECK(input.sizes() == residual.sizes(),
              "`AddRmsNorm`: `input` and `residual` must have the same shape");
  TORCH_CHECK(input.scalar_type() == residual.scalar_type(),
              "`AddRmsNorm`: `input` and `residual` must have the same dtype");
  TORCH_CHECK(
      input.scalar_type() == at::kHalf || input.scalar_type() == at::kFloat,
      "`AddRmsNorm`: only `float16` and `float32` are supported; got ",
      input.scalar_type());
  TORCH_CHECK(weight.dim() == 1,
              "`AddRmsNorm`: `weight` must be 1-dimensional");
  TORCH_CHECK(weight.size(0) == input.size(-1), "`AddRmsNorm`: `weight` size (",
              weight.size(0), ") must match input last dim (", input.size(-1),
              ")");

  int64_t dim_length = input.size(-1);
  int64_t total_rows = input.numel() / dim_length;

  if (total_rows == 0 || dim_length == 0) {
    return {at::empty_like(input), at::empty_like(input)};
  }

  at::Tensor input_contiguous = input.contiguous();
  at::Tensor residual_contiguous = residual.contiguous();
  int64_t dtype_size = input_contiguous.element_size();

  // Hardware parameters.
  auto ascendc_platform =
      platform_ascendc::PlatformAscendCManager::GetInstance();
  int64_t core_num = static_cast<int64_t>(ascendc_platform->GetCoreNumAiv());
  uint64_t ub_size;
  ascendc_platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
  int64_t ub_size_limit = static_cast<int64_t>(ub_size);

  // Alignment (32-byte boundary).
  int64_t align_elements = 32 / dtype_size;
  int64_t dim_length_align =
      ((dim_length + align_elements - 1) / align_elements) * align_elements;

  // UB capacity check.
  //
  // - `fp16`: `inQ_x1` (*2*2) + `inQ_x2` (*2*2) + `outQ_y` (*2*2) +
  //   `outQ_xout` (*2*2) + `fp32Buf1` (*4) + `fp32Buf2` (*4) +
  //   `weight` (*4) = 16 + 12 = 28
  // - `fp32`: `inQ_x1` (*2*4) + `inQ_x2` (*2*4) + `outQ_y` (*2*4) +
  //   `outQ_xout` (*2*4) + `weight` (*4) = 32 + 4 = 36
  int64_t buffer_coefficient = (dtype_size == 2) ? 28 : 36;
  int64_t max_dim_length = (ub_size_limit - 1024) / buffer_coefficient;
  int64_t fp_align_elements = 32 / 4;
  max_dim_length = (max_dim_length / fp_align_elements) * fp_align_elements;
  TORCH_CHECK(dim_length_align <= max_dim_length, "`AddRmsNorm`: `dim_length` ",
              dim_length, " (aligned ", dim_length_align,
              ") exceeds UB capacity (max ", max_dim_length, ")");

  // Padding.
  at::Tensor kernel_input;
  at::Tensor kernel_residual;

  if (dim_length != dim_length_align) {
    kernel_input = input_contiguous.reshape({total_rows, dim_length});
    kernel_input = at::constant_pad_nd(kernel_input,
                                       {0, dim_length_align - dim_length}, 0.0);
    kernel_input = kernel_input.contiguous();

    kernel_residual = residual_contiguous.reshape({total_rows, dim_length});
    kernel_residual = at::constant_pad_nd(
        kernel_residual, {0, dim_length_align - dim_length}, 0.0);
    kernel_residual = kernel_residual.contiguous();
  } else {
    kernel_input =
        input_contiguous.reshape({total_rows, dim_length_align}).contiguous();
    kernel_residual =
        residual_contiguous.reshape({total_rows, dim_length_align})
            .contiguous();
  }

  at::Tensor kernel_out = at::empty_like(kernel_input);
  at::Tensor kernel_residual_out = at::empty_like(kernel_input);

  // Always pass `weight` as `fp32`, padded to `dim_length_align`.
  at::Tensor weight_float = weight.contiguous().to(at::kFloat);

  if (dim_length != dim_length_align) {
    weight_float = at::constant_pad_nd(weight_float,
                                       {0, dim_length_align - dim_length}, 0.0);
  }

  weight_float = weight_float.contiguous();

  // Block-level tiling (distribute rows across cores).
  int64_t used_core_num = std::min(total_rows, core_num);
  int64_t former_length = (total_rows + used_core_num - 1) / used_core_num;
  int64_t tail_length = former_length - 1;
  int64_t former_num = total_rows - tail_length * used_core_num;
  uint32_t block_dim = static_cast<uint32_t>(used_core_num);

  // All `EXEC_KERNEL_CMD` args must be lvalues.
  float eps_float = static_cast<float>(eps);
  int64_t dtype_code = input.scalar_type() == at::kHalf
                           ? kInfiniDataTypeFloat16
                           : kInfiniDataTypeFloat32;

  // The first arg `AddRmsNorm` is the AscendC kernel entry-point name — it
  // must match the `__global__ __aicore__ void AddRmsNorm(...)` definition
  // in `op_kernel/` and the generated `aclrtlaunch_AddRmsNorm.h` header.
  EXEC_KERNEL_CMD(AddRmsNorm, block_dim, kernel_input, kernel_residual,
                  weight_float, total_rows, dim_length, dim_length_align,
                  former_num, former_length, tail_length, eps_float, dtype_code,
                  kernel_out, kernel_residual_out);

  // Remove padding and reshape back to original shape.
  at::Tensor out = kernel_out;
  at::Tensor residual_out = kernel_residual_out;

  if (dim_length != dim_length_align) {
    out = out.narrow(-1, 0, dim_length).contiguous();
    residual_out = residual_out.narrow(-1, 0, dim_length).contiguous();
  }

  out = out.reshape(input.sizes());
  residual_out = residual_out.reshape(input.sizes());

  return {out, residual_out};
}

}  // namespace ascend::detail
