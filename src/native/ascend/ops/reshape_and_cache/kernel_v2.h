#ifndef INFINI_OPS_ASCEND_RESHAPE_AND_CACHE_KERNEL_V2_H_
#define INFINI_OPS_ASCEND_RESHAPE_AND_CACHE_KERNEL_V2_H_

// WARNING: This implementation is experimental and has strict hardware limits.
//
// Limitations:
//   1. Requires CANN 8.5.1+ (`aclnnScatterPaKvCache` API).
//   2. Only supported on Atlas A5 hardware (SoC 260).  NOT supported on
//      A2 (Ascend 910B, SoC 220-225) or A3 (SoC 250-255).
//   3. Not yet validated in production workloads.
//
// On unsupported hardware this file compiles to nothing (guarded by
// `__has_include`).  Use `implementation_index=0` (the default
// `aclnnInplaceIndexCopy` path) for general-purpose deployment.

#if __has_include("aclnnop/aclnn_scatter_pa_kv_cache.h")

#include <cassert>
#include <cstddef>
#include <string>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_scatter_pa_kv_cache.h"
#include "base/reshape_and_cache.h"
#include "native/ascend/common.h"
#include "native/ascend/workspace_pool_.h"
#include "operator.h"

namespace infini::ops {

// Fused KV cache scatter via `aclnnScatterPaKvCache` (implementation index 1).
//
// Handles both K and V scatter in a single CANN kernel launch, replacing two
// separate `aclnnInplaceIndexCopy` calls (index 0).  The fused API is
// purpose-built for paged KV cache and avoids the internal decomposition to
// `ScatterElementsV2`.
//
// Requirements:
//   - CANN 8.5.1+ (`aclnnop/aclnn_scatter_pa_kv_cache.h`).
//   - Atlas A5 hardware (SoC 260).  The API is NOT supported on A2 (910B,
//     SoC 220-225) or A3 (SoC 250-255).
//
// Select via `implementation_index=1` in Python:
//   infini.ops.reshape_and_cache(..., implementation_index=1, stream=s)
template <>
class Operator<ReshapeAndCache, Device::Type::kAscend, 1>
    : public ReshapeAndCache {
 public:
  Operator(const Tensor key, const Tensor value, const Tensor key_cache,
           const Tensor value_cache, const Tensor slot_mapping,
           const std::string kv_cache_dtype, const Tensor k_scale,
           const Tensor v_scale)
      : ReshapeAndCache(key, value, key_cache, value_cache, slot_mapping,
                        kv_cache_dtype, k_scale, v_scale),
        key_cache_(key),
        value_cache_(value),
        slot_cache_(slot_mapping) {
    auto num_blocks = static_cast<int64_t>(key_cache.size(0));
    auto bs = static_cast<int64_t>(block_size_);
    int64_t nkv = static_cast<int64_t>(num_kv_heads_);
    int64_t hs = static_cast<int64_t>(head_size_);

    aclDataType acl_dt = ascend::ToAclDtype(key.dtype());

    // 4D K cache view: [num_blocks, block_size, num_kv_heads, head_size].
    kv_k_cache_ = ascend::AclTensorCache({num_blocks, bs, nkv, hs}, acl_dt,
                                         const_cast<void*>(key_cache.data()));

    kv_v_cache_ = ascend::AclTensorCache({num_blocks, bs, nkv, hs}, acl_dt,
                                         const_cast<void*>(value_cache.data()));
  }

  void operator()(const Tensor key, const Tensor value, const Tensor key_cache,
                  const Tensor value_cache, const Tensor slot_mapping,
                  const std::string kv_cache_dtype, const Tensor k_scale,
                  const Tensor v_scale) const override {
    (void)kv_cache_dtype;
    (void)k_scale;
    (void)v_scale;

    auto stream = static_cast<aclrtStream>(stream_);

    void* kv_k_data = const_cast<void*>(key_cache.data());
    void* kv_v_data = const_cast<void*>(value_cache.data());

    auto t_key = key_cache_.get(const_cast<void*>(key.data()));
    auto t_value = value_cache_.get(const_cast<void*>(value.data()));
    auto t_slot = slot_cache_.get(const_cast<void*>(slot_mapping.data()));
    auto t_kv_k = kv_k_cache_.get(kv_k_data);
    auto t_kv_v = kv_v_cache_.get(kv_v_data);

    // Single fused scatter for both K and V caches.
    uint64_t ws = 0;
    aclOpExecutor* exec = nullptr;
    aclnnScatterPaKvCacheGetWorkspaceSize(
        t_key, t_kv_k, t_slot, t_value, t_kv_v,
        /*compressLensOptional=*/nullptr,
        /*compressSeqOffsetOptional=*/nullptr,
        /*seqLensOptional=*/nullptr,
        /*cacheModeOptional=*/nullptr,
        /*scatterModeOptional=*/nullptr,
        /*stridesOptional=*/nullptr,
        /*offsetsOptional=*/nullptr, &ws, &exec);
    auto& arena = ascend::GetWorkspacePool().Ensure(stream, ws);
    aclnnScatterPaKvCache(arena.buf, ws, exec, stream);
  }

 private:
  mutable ascend::AclTensorCache kv_k_cache_;

  mutable ascend::AclTensorCache kv_v_cache_;

  mutable ascend::AclTensorCache key_cache_;

  mutable ascend::AclTensorCache value_cache_;

  mutable ascend::AclTensorCache slot_cache_;
};

}  // namespace infini::ops

#endif  // __has_include("aclnnop/aclnn_scatter_pa_kv_cache.h")

#endif  // INFINI_OPS_ASCEND_RESHAPE_AND_CACHE_KERNEL_V2_H_
