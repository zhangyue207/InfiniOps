#ifndef INFINI_OPS_BASE_PAGED_ATTENTION_H_
#define INFINI_OPS_BASE_PAGED_ATTENTION_H_

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

#include "data_type.h"
#include "operator.h"

namespace infini::ops {

// Paged decode attention operator.
//
// Matches vLLM `_C.paged_attention_v1`.
class PagedAttention : public Operator<PagedAttention> {
 public:
  PagedAttention(const Tensor output, const Tensor query,
                 const Tensor key_cache, const Tensor value_cache,
                 int64_t num_kv_heads, float scale, const Tensor block_tables,
                 const Tensor seq_lens, int64_t block_size, int64_t max_seq_len,
                 std::optional<Tensor> alibi_slopes,
                 const std::string kv_cache_dtype, const Tensor k_scale,
                 const Tensor v_scale, int64_t tp_rank = 0,
                 int64_t blocksparse_local_blocks = 0,
                 int64_t blocksparse_vert_stride = 0,
                 int64_t blocksparse_block_size = 64,
                 int64_t blocksparse_head_sliding_step = 0)
      : batch_size_{query.size(0)},
        num_heads_{query.size(1)},
        num_kv_heads_{num_kv_heads},
        head_size_{query.size(2)},
        scale_{scale},
        block_size_{block_size},
        max_seq_len_{max_seq_len},
        kv_cache_dtype_{kv_cache_dtype},
        dtype_{query.dtype()},
        output_shape_{output.shape()},
        query_shape_{query.shape()},
        key_cache_shape_{key_cache.shape()},
        value_cache_shape_{value_cache.shape()},
        block_tables_shape_{block_tables.shape()},
        seq_lens_shape_{seq_lens.shape()},
        has_alibi_slopes_{alibi_slopes.has_value()} {
    (void)k_scale;
    (void)v_scale;

    assert(num_heads_ % num_kv_heads_ == 0 &&
           "`PagedAttention` requires `num_heads` divisible by `num_kv_heads`");
    assert(query.ndim() == 3 &&
           "`PagedAttention` requires `query` to be 3D "
           "`[batch, num_heads, head_size]`");
    assert(key_cache.ndim() == 4 &&
           "`PagedAttention` requires `key_cache` to be 4D "
           "`[num_blocks, block_size, num_kv_heads, head_size]`");
    assert(value_cache.ndim() == 4 &&
           "`PagedAttention` requires `value_cache` to be 4D "
           "`[num_blocks, block_size, num_kv_heads, head_size]`");
    assert(key_cache.shape() == value_cache.shape() &&
           "`PagedAttention` requires `key_cache` and `value_cache` same "
           "shape");
    assert(query.dtype() == key_cache.dtype() &&
           query.dtype() == value_cache.dtype() &&
           query.dtype() == output.dtype() &&
           "`PagedAttention` requires `query`, caches, and `output` same "
           "dtype");
    assert(output.shape() == query.shape() &&
           "`PagedAttention` requires `output` to match `query` shape");
    assert(query.stride(-1) == 1 && key_cache.stride(-1) == 1 &&
           value_cache.stride(-1) == 1 && output.stride(-1) == 1 &&
           "`PagedAttention` requires contiguous last dimension");
    assert(key_cache.size(1) == static_cast<Tensor::Size>(block_size) &&
           "`PagedAttention` requires `block_size` to match cache shape");
    assert(key_cache.size(2) == static_cast<Tensor::Size>(num_kv_heads) &&
           "`PagedAttention` requires `num_kv_heads` to match cache shape");
    assert(key_cache.size(3) == head_size_ &&
           "`PagedAttention` requires `head_size` to match cache shape");
    assert(block_tables.ndim() == 2 &&
           "`PagedAttention` requires `block_tables` to be 2D "
           "`[batch, max_num_blocks]`");
    assert(block_tables.size(0) == batch_size_ &&
           "`PagedAttention` requires `block_tables` batch to match `query`");
    assert(block_tables.dtype() == DataType::kInt32 &&
           "`PagedAttention` requires `block_tables` to be `int32`");
    assert(seq_lens.ndim() == 1 &&
           "`PagedAttention` requires `seq_lens` to be 1D `[batch]`");
    assert(seq_lens.size(0) == batch_size_ &&
           "`PagedAttention` requires `seq_lens` batch to match `query`");
    assert(seq_lens.dtype() == DataType::kInt32 &&
           "`PagedAttention` requires `seq_lens` to be `int32`");
    assert(max_seq_len >= 0 &&
           "`PagedAttention` requires non-negative `max_seq_len`");
    assert(!alibi_slopes.has_value() &&
           "`PagedAttention` does not support `alibi_slopes` on Ascend yet");
    assert(kv_cache_dtype == "auto" &&
           "`PagedAttention` currently supports only `kv_cache_dtype=auto`");
    assert(tp_rank == 0 && "`PagedAttention` requires `tp_rank == 0`");
    assert(blocksparse_local_blocks == 0 &&
           "`PagedAttention` does not support block-sparse attention");
    assert(blocksparse_vert_stride == 0 &&
           "`PagedAttention` does not support block-sparse attention");
    assert(blocksparse_block_size == 64 &&
           "`PagedAttention` requires default `blocksparse_block_size`");
    assert(blocksparse_head_sliding_step == 0 &&
           "`PagedAttention` does not support block-sparse attention");
  }

  virtual void operator()(
      const Tensor output, const Tensor query, const Tensor key_cache,
      const Tensor value_cache, int64_t num_kv_heads, float scale,
      const Tensor block_tables, const Tensor seq_lens, int64_t block_size,
      int64_t max_seq_len, std::optional<Tensor> alibi_slopes,
      const std::string kv_cache_dtype, const Tensor k_scale,
      const Tensor v_scale, int64_t tp_rank = 0,
      int64_t blocksparse_local_blocks = 0, int64_t blocksparse_vert_stride = 0,
      int64_t blocksparse_block_size = 64,
      int64_t blocksparse_head_sliding_step = 0) const = 0;

 protected:
  Tensor::Size batch_size_{0};

  Tensor::Size num_heads_{0};

  int64_t num_kv_heads_{0};

  Tensor::Size head_size_{0};

  float scale_{0.0f};

  int64_t block_size_{0};

  int64_t max_seq_len_{0};

  std::string kv_cache_dtype_;

  const DataType dtype_;

  Tensor::Shape output_shape_;

  Tensor::Shape query_shape_;

  Tensor::Shape key_cache_shape_;

  Tensor::Shape value_cache_shape_;

  Tensor::Shape block_tables_shape_;

  Tensor::Shape seq_lens_shape_;

  bool has_alibi_slopes_{false};
};

}  // namespace infini::ops

#endif
