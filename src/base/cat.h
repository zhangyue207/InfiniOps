#ifndef INFINI_OPS_BASE_CAT_H_
#define INFINI_OPS_BASE_CAT_H_

#include <vector>

#include "operator.h"

namespace infini::ops {

// Concatenates materialized tensors with the PyTorch `torch.cat` shape
// contract.  PyTorch's special 1D empty-tensor cases are outside InfiniOps v1.
class Cat : public Operator<Cat> {
 public:
  Cat(const Tensor first_input, std::vector<Tensor> rest_inputs, int64_t dim,
      Tensor out)
      : input_count_{1 + rest_inputs.size()} {
    assert(input_count_ >= 2 && "`Cat` requires at least 2 input tensors");

    auto ndim = static_cast<int64_t>(out.ndim());
    // Normalize negative dim (e.g. -1 means last dimension).
    dim_ = dim < 0 ? dim + ndim : dim;
    assert(dim_ >= 0 && dim_ < ndim && "`Cat` dim out of range");
    assert(first_input.ndim() == out.ndim() &&
           "`Cat` requires inputs and output to have same ndim");
    assert(first_input.dtype() == out.dtype() &&
           "`Cat` requires inputs and output to have same dtype");
    assert(first_input.device() == out.device() &&
           "`Cat` requires inputs and output to be on same device");

    Tensor::Shape expected_shape = first_input.shape();
    for (const auto& input : rest_inputs) {
      assert(input.ndim() == out.ndim() &&
             "`Cat` requires all inputs to have same ndim");
      assert(input.dtype() == out.dtype() &&
             "`Cat` requires all inputs and output to have same dtype");
      assert(input.device() == out.device() &&
             "`Cat` requires all inputs and output to be on same device");

      for (std::size_t i = 0; i < expected_shape.size(); ++i) {
        if (static_cast<int64_t>(i) == dim_) continue;
        assert(input.size(i) == expected_shape[i] &&
               "`Cat` requires non-concatenated dimensions to match");
      }

      expected_shape[dim_] += input.size(dim_);
    }

    assert(out.shape() == expected_shape &&
           "`Cat` requires output shape to equal concatenated input shape");
  }

  virtual void operator()(const Tensor first_input,
                          std::vector<Tensor> rest_inputs, int64_t dim,
                          Tensor out) const = 0;

 protected:
  int64_t dim_;

  size_t input_count_;
};

}  // namespace infini::ops

#endif
