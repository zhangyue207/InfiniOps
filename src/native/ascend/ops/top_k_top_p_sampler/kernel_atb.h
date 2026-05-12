#ifndef INFINI_OPS_ASCEND_TOP_K_TOP_P_SAMPLER_KERNEL_ATB_H_
#define INFINI_OPS_ASCEND_TOP_K_TOP_P_SAMPLER_KERNEL_ATB_H_

#ifdef INFINI_HAS_ATB

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <optional>

#include "acl/acl.h"
#include "native/ascend/ops/scaled_softmax/kernel.h"
#include "native/ascend/ops/topk_topp_sampling/kernel_atb.h"
#include "native/ascend/workspace_pool_.h"
#include "base/top_k_top_p_sampler.h"
#include "data_type.h"
#include "operator.h"
#include "tensor.h"

namespace infini::ops {

template <>
class Operator<TopKTopPSampler, Device::Type::kAscend, 1>
    : public TopKTopPSampler {
 public:
  Operator(const Tensor logits, std::optional<Tensor> k,
           std::optional<Tensor> p, Tensor out)
      : TopKTopPSampler(logits, k, p, out) {
    assert((dtype_ == DataType::kFloat16 || dtype_ == DataType::kBFloat16) &&
           "`TopKTopPSampler` Ascend ATB path requires float16 or bfloat16 "
           "logits.");
    assert(logits.IsContiguous() &&
           "`TopKTopPSampler` Ascend ATB path requires contiguous logits.");
    assert(out.IsContiguous() &&
           "`TopKTopPSampler` Ascend ATB path requires contiguous output.");
    ValidateHostTensor(k);
    ValidateHostTensor(p);
  }

  void operator()(const Tensor logits, std::optional<Tensor> k,
                  std::optional<Tensor> p, Tensor out) const override {
    assert(logits.IsContiguous() &&
           "`TopKTopPSampler` Ascend ATB path requires contiguous logits.");
    assert(out.IsContiguous() &&
           "`TopKTopPSampler` Ascend ATB path requires contiguous output.");

    auto stream = static_cast<aclrtStream>(stream_);
    auto elem_size = kDataTypeToSize.at(dtype_);
    auto probs_size = vocab_size_ * elem_size;
    auto& probs_arena = ascend::GetWorkspacePool().Ensure(stream, probs_size,
                                                          "top_k_top_p_probs");
    auto* logits_base =
        static_cast<std::uint8_t*>(const_cast<void*>(logits.data()));
    auto* out_base = static_cast<std::uint8_t*>(out.data());
    Config sub_config;

    for (Tensor::Size row = 0; row < batch_size_; ++row) {
      auto logits_offset =
          static_cast<std::size_t>(row * logits.stride(0)) * elem_size;
      auto out_offset = static_cast<std::size_t>(row * out.stride(0)) *
                        kDataTypeToSize.at(DataType::kInt32);

      Tensor row_logits{logits_base + logits_offset,
                        Tensor::Shape{1, vocab_size_}, dtype_, logits.device()};
      Tensor probs{probs_arena.buf, Tensor::Shape{1, vocab_size_}, dtype_,
                   logits.device()};
      Tensor row_out{out_base + out_offset, Tensor::Shape{1}, DataType::kInt32,
                     out.device()};

      ScaledSoftmax::Call(handle_, sub_config, row_logits, 1.0, probs);
      TopkToppSampling::Call(handle_, sub_config, probs, GetK(k, row),
                             GetP(p, row), row_out);
    }
  }

 private:
  void ValidateHostTensor(std::optional<Tensor> tensor) const {
    if (!tensor.has_value()) return;

    assert(tensor->device().type() == Device::Type::kCpu &&
           "`TopKTopPSampler` Ascend ATB path currently requires host-side "
           "`k`/`p` tensors.");
    assert(tensor->IsContiguous() &&
           "`TopKTopPSampler` Ascend ATB path requires contiguous `k`/`p` "
           "tensors.");
  }

  int64_t GetK(std::optional<Tensor> k, Tensor::Size row) const {
    if (!k.has_value()) return static_cast<int64_t>(vocab_size_);

    const auto offset = k->size(0) == 1 ? 0 : row;
    int64_t value = 0;
    if (k->dtype() == DataType::kInt32) {
      value = static_cast<const int32_t*>(k->data())[offset];
    } else {
      value = static_cast<const int64_t*>(k->data())[offset];
    }

    if (value <= 0) return static_cast<int64_t>(vocab_size_);
    return std::min<int64_t>(value, static_cast<int64_t>(vocab_size_));
  }

  double GetP(std::optional<Tensor> p, Tensor::Size row) const {
    if (!p.has_value()) return 1.0;

    const auto offset = p->size(0) == 1 ? 0 : row;
    double value = 1.0;
    switch (p->dtype()) {
      case DataType::kFloat16:
        value = static_cast<const Float16*>(p->data())[offset].ToFloat();
        break;
      case DataType::kBFloat16:
        value = static_cast<const BFloat16*>(p->data())[offset].ToFloat();
        break;
      case DataType::kFloat32:
        value = static_cast<const float*>(p->data())[offset];
        break;
      case DataType::kFloat64:
        value = static_cast<const double*>(p->data())[offset];
        break;
      default:
        assert(false && "`TopKTopPSampler` has unsupported `p` dtype.");
    }

    if (value <= 0.0 || value > 1.0) return 1.0;
    return value;
  }
};

}  // namespace infini::ops

#endif  // INFINI_HAS_ATB

#endif  // INFINI_OPS_ASCEND_TOP_K_TOP_P_SAMPLER_KERNEL_ATB_H_
