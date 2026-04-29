#ifndef INFINI_OPS_CPU_SCALED_SOFTMAX_H_
#define INFINI_OPS_CPU_SCALED_SOFTMAX_H_

#include <cmath>
#include <limits>

#include "base/scaled_softmax.h"
#include "common/generic_utils.h"
#include "cpu/caster_.h"
#include "data_type.h"
#include "tensor.h"

namespace infini::ops {

template <>
class Operator<ScaledSoftmax, Device::Type::kCpu> : public ScaledSoftmax,
                                                    Caster<Device::Type::kCpu> {
 public:
  Operator(const Tensor input, double scale, Tensor out)
      : ScaledSoftmax(input, scale, out) {}

  void operator()(const Tensor input, double scale, Tensor out) const override {
    assert(scale == scale_ &&
           "`ScaledSoftmax` scale changed after descriptor creation.");
    DispatchFunc<Device::Type::kCpu, AllFloatTypes>(
        out.dtype(),
        [&](auto tag) {
          using T = typename decltype(tag)::type;
          Compute<T>(input, out);
        },
        "`Operator<ScaledSoftmax, Device::Type::kCpu>::operator()`");
  }

 private:
  template <typename T>
  void Compute(const Tensor input, Tensor out) const {
    const auto* input_ptr = static_cast<const T*>(input.data());
    auto* out_ptr = static_cast<T*>(out.data());

    for (Tensor::Size i = 0; i < batch_size_; ++i) {
      const auto* input_row = input_ptr + i * input_strides_[0];
      auto* out_row = out_ptr + i * out_strides_[0];

      float max_val = -std::numeric_limits<float>::infinity();
      for (Tensor::Size j = 0; j < vocab_size_; ++j) {
        float v = Cast<float>(input_row[j * input_strides_[1]]) *
                  static_cast<float>(scale_);
        if (v > max_val) {
          max_val = v;
        }
      }

      float sum = 0.0f;
      for (Tensor::Size j = 0; j < vocab_size_; ++j) {
        float v = std::exp(Cast<float>(input_row[j * input_strides_[1]]) *
                               static_cast<float>(scale_) -
                           max_val);
        out_row[j * out_strides_[1]] = Cast<T>(v);
        sum += v;
      }

      for (Tensor::Size j = 0; j < vocab_size_; ++j) {
        auto out_offset = j * out_strides_[1];
        out_row[out_offset] = Cast<T>(Cast<float>(out_row[out_offset]) / sum);
      }
    }
  }
};

}  // namespace infini::ops

#endif  // INFINI_OPS_CPU_SCALED_SOFTMAX_H_
