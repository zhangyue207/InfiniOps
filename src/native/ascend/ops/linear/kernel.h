#ifndef INFINI_OPS_ASCEND_LINEAR_KERNEL_H_
#define INFINI_OPS_ASCEND_LINEAR_KERNEL_H_

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_add.h"
#include "aclnnop/aclnn_addmm.h"
#include "aclnnop/aclnn_baddbmm.h"
#include "aclnnop/aclnn_matmul.h"
#include "base/linear.h"
#include "native/ascend/common.h"
#include "native/ascend/workspace_pool_.h"
#include "operator.h"

namespace infini::ops {

template <>
class Operator<Linear, Device::Type::kAscend> : public Linear {
 public:
  Operator(const Tensor a, const Tensor b, std::optional<Tensor> bias,
           bool trans_a, bool trans_b, Tensor out)
      : Linear(a, b, bias, trans_a, trans_b, out),
        batched_{out.ndim() > 2},
        flatten_batched_bias_{has_bias_ && batched_ && b.ndim() == 2 &&
                              !trans_a},
        a_cache_(a, trans_a),
        b_cache_(b, trans_b),
        out_cache_(out) {
    if (has_bias_) {
      bias_cache_ = ascend::AclTensorCache(*bias);
      alpha_scalar_ = aclCreateScalar(&alpha_storage_, ACL_FLOAT);
      beta_scalar_ = aclCreateScalar(&beta_storage_, ACL_FLOAT);
    }

    if (flatten_batched_bias_) {
      assert(a.IsContiguous() &&
             "`Linear`: Ascend batched input with 2D weight and bias requires "
             "contiguous input.");
      assert(out.IsContiguous() &&
             "`Linear`: Ascend batched input with 2D weight and bias requires "
             "contiguous output.");

      auto rows = static_cast<int64_t>(out.numel() / out.size(-1));
      auto in_features = static_cast<int64_t>(a.size(-1));
      auto out_features = static_cast<int64_t>(out.size(-1));

      flat_a_cache_ = ascend::AclTensorCache(
          {rows, in_features}, ascend::ToAclDtype(a.dtype()), nullptr);
      flat_out_cache_ = ascend::AclTensorCache(
          {rows, out_features}, ascend::ToAclDtype(out.dtype()), nullptr);
    }
  }

  // vLLM-aligned overload — `weight [out, in]`, `out = input @ weight^T`.
  Operator(const Tensor input, const Tensor weight, std::optional<Tensor> bias,
           Tensor out)
      : Operator(input, weight, bias, /*trans_a=*/false, /*trans_b=*/true,
                 out) {}

  ~Operator() {
    if (!ascend::IsAclRuntimeAlive()) return;

    // Null cached descriptors — see `AclTensorCache::release()`.
    bias_cache_.release();
    a_cache_.release();
    b_cache_.release();
    out_cache_.release();
    flat_a_cache_.release();
    flat_out_cache_.release();

    if (alpha_scalar_) aclDestroyScalar(alpha_scalar_);
    if (beta_scalar_) aclDestroyScalar(beta_scalar_);
  }

  void operator()(const Tensor a, const Tensor b, std::optional<Tensor> bias,
                  bool trans_a, bool trans_b, Tensor out) const override {
    auto stream = static_cast<aclrtStream>(stream_);

    if (has_bias_ && flatten_batched_bias_) {
      auto t_bias = bias_cache_.get(const_cast<void*>(bias->data()));
      auto t_a = flat_a_cache_.get(const_cast<void*>(a.data()));
      auto t_b = b_cache_.get(const_cast<void*>(b.data()));
      auto t_out = flat_out_cache_.get(out.data());

      if (!flat_matmul_exec_) {
        int8_t cube_math_type = 1;
        auto ret =
            aclnnMatmulGetWorkspaceSize(t_a, t_b, t_out, cube_math_type,
                                        &flat_matmul_ws_, &flat_matmul_exec_);
        assert(ret == ACL_SUCCESS && "`aclnnMatmulGetWorkspaceSize` failed");
        aclSetAclOpExecutorRepeatable(flat_matmul_exec_);
      } else {
        aclSetInputTensorAddr(flat_matmul_exec_, 0, t_a,
                              const_cast<void*>(a.data()));
        aclSetInputTensorAddr(flat_matmul_exec_, 1, t_b,
                              const_cast<void*>(b.data()));
        aclSetOutputTensorAddr(flat_matmul_exec_, 0, t_out, out.data());
      }

      auto& matmul_arena =
          ascend::GetWorkspacePool().Ensure(stream, flat_matmul_ws_);
      auto ret = aclnnMatmul(matmul_arena.buf, flat_matmul_ws_,
                             flat_matmul_exec_, stream);
      assert(ret == ACL_SUCCESS && "`aclnnMatmul` failed");

      if (!flat_add_exec_) {
        ret = aclnnAddGetWorkspaceSize(t_out, t_bias, alpha_scalar_, t_out,
                                       &flat_add_ws_, &flat_add_exec_);
        assert(ret == ACL_SUCCESS && "`aclnnAddGetWorkspaceSize` failed");
        aclSetAclOpExecutorRepeatable(flat_add_exec_);
      } else {
        aclSetInputTensorAddr(flat_add_exec_, 0, t_out, out.data());
        aclSetInputTensorAddr(flat_add_exec_, 1, t_bias,
                              const_cast<void*>(bias->data()));
        aclSetOutputTensorAddr(flat_add_exec_, 0, t_out, out.data());
      }

      auto& add_arena = ascend::GetWorkspacePool().Ensure(stream, flat_add_ws_);
      ret = aclnnAdd(add_arena.buf, flat_add_ws_, flat_add_exec_, stream);
      assert(ret == ACL_SUCCESS && "`aclnnAdd` failed");
      return;
    }

    auto t_a = a_cache_.get(const_cast<void*>(a.data()));
    auto t_b = b_cache_.get(const_cast<void*>(b.data()));
    auto t_out = out_cache_.get(out.data());

    if (has_bias_) {
      auto t_bias = bias_cache_.get(const_cast<void*>(bias->data()));
      if (!executor_) {
        if (batched_) {
          aclnnBaddbmmGetWorkspaceSize(t_bias, t_a, t_b, beta_scalar_,
                                       alpha_scalar_, t_out, 0, &ws_size_,
                                       &executor_);
        } else {
          aclnnAddmmGetWorkspaceSize(t_bias, t_a, t_b, beta_scalar_,
                                     alpha_scalar_, t_out, 0, &ws_size_,
                                     &executor_);
        }
        aclSetAclOpExecutorRepeatable(executor_);
      } else {
        aclSetInputTensorAddr(executor_, 0, t_bias,
                              const_cast<void*>(bias->data()));
        aclSetInputTensorAddr(executor_, 1, t_a, const_cast<void*>(a.data()));
        aclSetInputTensorAddr(executor_, 2, t_b, const_cast<void*>(b.data()));
        aclSetOutputTensorAddr(executor_, 0, t_out, out.data());
      }

      auto& arena = ascend::GetWorkspacePool().Ensure(stream, ws_size_);

      if (batched_) {
        aclnnBaddbmm(arena.buf, ws_size_, executor_, stream);
      } else {
        aclnnAddmm(arena.buf, ws_size_, executor_, stream);
      }
    } else {
      if (!executor_) {
        int8_t cube_math_type = 1;
        aclnnMatmulGetWorkspaceSize(t_a, t_b, t_out, cube_math_type, &ws_size_,
                                    &executor_);
        aclSetAclOpExecutorRepeatable(executor_);
      } else {
        aclSetInputTensorAddr(executor_, 0, t_a, const_cast<void*>(a.data()));
        aclSetInputTensorAddr(executor_, 1, t_b, const_cast<void*>(b.data()));
        aclSetOutputTensorAddr(executor_, 0, t_out, out.data());
      }

      auto& arena = ascend::GetWorkspacePool().Ensure(stream, ws_size_);
      aclnnMatmul(arena.buf, ws_size_, executor_, stream);
    }
  }

 private:
  bool batched_;

  bool flatten_batched_bias_;

  mutable ascend::AclTensorCache bias_cache_;

  mutable ascend::AclTensorCache a_cache_;

  mutable ascend::AclTensorCache b_cache_;

  mutable ascend::AclTensorCache out_cache_;

  mutable ascend::AclTensorCache flat_a_cache_;

  mutable ascend::AclTensorCache flat_out_cache_;

  float alpha_storage_ = 1.0f;

  float beta_storage_ = 1.0f;

  aclScalar* alpha_scalar_ = nullptr;

  aclScalar* beta_scalar_ = nullptr;

  mutable aclOpExecutor* executor_ = nullptr;

  mutable uint64_t ws_size_ = 0;

  mutable aclOpExecutor* flat_matmul_exec_ = nullptr;

  mutable uint64_t flat_matmul_ws_ = 0;

  mutable aclOpExecutor* flat_add_exec_ = nullptr;

  mutable uint64_t flat_add_ws_ = 0;
};

}  // namespace infini::ops

#endif
