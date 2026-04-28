import infini.ops
import pytest
import torch

from tests.utils import Payload, get_stream, randn_strided


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "seq_lens, num_heads, num_kv_heads, head_size",
    (
        ((16,), 8, 8, 64),
        ((7, 11, 5), 8, 2, 64),
        ((9, 3), 16, 4, 128),
    ),
)
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float16, 1e-3, 1e-3),
        (torch.bfloat16, 1e-2, 5e-3),
    ),
)
def test_mha_varlen_fwd(
    seq_lens,
    num_heads,
    num_kv_heads,
    head_size,
    implementation_index,
    dtype,
    device,
    rtol,
    atol,
):
    scale = 1.0 / head_size**0.5
    total_tokens = sum(seq_lens)
    q = randn_strided(
        (total_tokens, num_heads, head_size), None, dtype=dtype, device=device
    )
    k = randn_strided(
        (total_tokens, num_kv_heads, head_size), None, dtype=dtype, device=device
    )
    v = randn_strided(
        (total_tokens, num_kv_heads, head_size), None, dtype=dtype, device=device
    )
    out = torch.empty_like(q)
    cu = torch.tensor(
        [0] + [sum(seq_lens[: i + 1]) for i in range(len(seq_lens))],
        dtype=torch.int32,
        device=device,
    )

    return Payload(
        lambda *args, **kwargs: _mha_varlen_fwd(
            *args, **kwargs, implementation_index=implementation_index
        ),
        _ref_mha_varlen_fwd,
        (q, k, v, out, cu, cu),
        {
            "seq_lens": seq_lens,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "head_size": head_size,
            "scale": scale,
            "causal": True,
        },
        rtol=rtol,
        atol=atol,
    )


def _mha_varlen_fwd(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    cu_seqlens_k,
    *,
    seq_lens,
    num_heads,
    num_kv_heads,
    head_size,
    scale,
    causal,
    implementation_index=0,
):
    infini.ops.mha_varlen_fwd(
        q,
        k,
        v,
        out,
        cu_seqlens_q,
        cu_seqlens_k,
        None,
        None,
        None,
        None,
        max(seq_lens),
        max(seq_lens),
        0.0,
        scale,
        False,
        causal,
        -1,
        0,
        0.0,
        False,
        None,
        0,
        implementation_index=implementation_index,
        stream=get_stream(q.device),
    )

    return out


def _ref_mha_varlen_fwd(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    cu_seqlens_k,
    *,
    seq_lens,
    num_heads,
    num_kv_heads,
    head_size,
    scale,
    causal,
):
    del out, cu_seqlens_q, cu_seqlens_k, head_size

    outputs = []
    offset = 0

    for seq_len in seq_lens:
        q_i = q[offset : offset + seq_len].cpu().float().transpose(0, 1)
        k_i = k[offset : offset + seq_len].cpu().float().transpose(0, 1)
        v_i = v[offset : offset + seq_len].cpu().float().transpose(0, 1)

        if num_kv_heads < num_heads:
            repeat = num_heads // num_kv_heads
            k_i = k_i.repeat_interleave(repeat, dim=0)
            v_i = v_i.repeat_interleave(repeat, dim=0)

        ref = torch.nn.functional.scaled_dot_product_attention(
            q_i.unsqueeze(0),
            k_i.unsqueeze(0),
            v_i.unsqueeze(0),
            scale=scale,
            is_causal=causal,
        )
        outputs.append(ref.squeeze(0).transpose(0, 1).to(q.dtype))
        offset += seq_len

    return torch.cat(outputs, dim=0).to(q.device)
