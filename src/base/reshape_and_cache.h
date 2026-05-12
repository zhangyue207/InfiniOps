#ifndef INFINI_OPS_BASE_RESHAPE_AND_CACHE_H_
#define INFINI_OPS_BASE_RESHAPE_AND_CACHE_H_

#include <cstddef>
#include <string>
#include <vector>

#include "data_type.h"
#include "operator.h"

namespace infini::ops {

class ReshapeAndCache : public Operator<ReshapeAndCache> {
 public:
  // Matches vLLM `_C_cache_ops.reshape_and_cache`.
  ReshapeAndCache(const Tensor key, const Tensor value, const Tensor key_cache,
                  const Tensor value_cache, const Tensor slot_mapping,
                  const std::string kv_cache_dtype, const Tensor k_scale,
                  const Tensor v_scale)
      : num_tokens_{key.size(0)},
        num_kv_heads_{key.size(1)},
        head_size_{key.size(2)},
        block_size_{key_cache.size(1)},
        key_shape_{key.shape()},
        value_shape_{value.shape()},
        key_cache_shape_{key_cache.shape()},
        value_cache_shape_{value_cache.shape()},
        slot_mapping_shape_{slot_mapping.shape()},
        key_strides_{key.strides()},
        value_strides_{value.strides()},
        key_cache_strides_{key_cache.strides()},
        value_cache_strides_{value_cache.strides()},
        slot_mapping_strides_{slot_mapping.strides()},
        kv_cache_dtype_{kv_cache_dtype} {
    (void)k_scale;
    (void)v_scale;

    assert(key.shape() == value.shape() &&
           "`ReshapeAndCache` requires `key` and `value` same shape");
    assert(key_cache.ndim() == 4 &&
           "`ReshapeAndCache` requires `key_cache` to be 4D "
           "`[num_blocks, block_size, num_kv_heads, head_size]`");
    assert(value_cache.ndim() == 4 &&
           "`ReshapeAndCache` requires `value_cache` to be 4D "
           "`[num_blocks, block_size, num_kv_heads, head_size]`");
    assert(key_cache.shape() == value_cache.shape() &&
           "`ReshapeAndCache` requires `key_cache` and `value_cache` same "
           "shape");
    assert(key.dtype() == value.dtype() && key.dtype() == key_cache.dtype() &&
           key.dtype() == value_cache.dtype() &&
           "`ReshapeAndCache` requires `key`, `value`, `key_cache`, and "
           "`value_cache` same dtype");
    assert(key_cache.size(1) == block_size_ &&
           "`ReshapeAndCache` requires `block_size` to match cache shape");
    assert(key_cache.size(2) == num_kv_heads_ &&
           "`ReshapeAndCache` requires `num_kv_heads` to match cache shape");
    assert(key_cache.size(3) == head_size_ &&
           "`ReshapeAndCache` requires `head_size` to match cache shape");
    assert(key.stride(-1) == 1 && value.stride(-1) == 1 &&
           key_cache.stride(-1) == 1 && value_cache.stride(-1) == 1 &&
           "`ReshapeAndCache` requires contiguous last dimension");
    assert(slot_mapping.ndim() == 1 &&
           "`ReshapeAndCache` requires `slot_mapping` to be 1D");
    assert(slot_mapping.size(0) == num_tokens_ &&
           "`ReshapeAndCache` requires `slot_mapping` length to match `key`");
    assert((slot_mapping.dtype() == DataType::kInt32 ||
            slot_mapping.dtype() == DataType::kInt64) &&
           "`ReshapeAndCache` requires `slot_mapping` to be `int32` or "
           "`int64`");
    assert(kv_cache_dtype == "auto" &&
           "`ReshapeAndCache` currently supports only `kv_cache_dtype=auto`");
  }

  virtual void operator()(const Tensor key, const Tensor value,
                          const Tensor key_cache, const Tensor value_cache,
                          const Tensor slot_mapping,
                          const std::string kv_cache_dtype,
                          const Tensor k_scale, const Tensor v_scale) const = 0;

 protected:
  Tensor::Size num_tokens_{0};

  Tensor::Size num_kv_heads_{0};

  Tensor::Size head_size_{0};

  Tensor::Size block_size_{0};

  Tensor::Shape key_shape_;

  Tensor::Shape value_shape_;

  Tensor::Shape key_cache_shape_;

  Tensor::Shape value_cache_shape_;

  Tensor::Shape slot_mapping_shape_;

  Tensor::Strides key_strides_;

  Tensor::Strides value_strides_;

  Tensor::Strides key_cache_strides_;

  Tensor::Strides value_cache_strides_;

  Tensor::Strides slot_mapping_strides_;

  std::string kv_cache_dtype_;
};

}  // namespace infini::ops

#endif
