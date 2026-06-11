"""KDA MTP decode — GEMM/tensor-core KVBuffer chunkwise verify (Triton tl.dot).

The paper's actual kernel form (arXiv 2605.19049 implements its kernels in Triton):
O = Q S0 + ((Q K^T) (.) M) V with every reduction as a tensor-core tl.dot, so the
per-token compute cost is negligible and verify latency stays ~flat in T (Fig. 4).
This is what the SIMT shuffle-reduce kernels (vk/ws/tp-kvbuffer) cannot achieve:
their shuffle instruction count grows ~80*T per warp and FFMA work grows ~T^2.

Schedule per program = one (n, hv) head, tokens padded to BT=16 for tl.dot:
  gates/l2norm elementwise -> log-decay lb = cumsum(log g) -> kdec/kinv/qdec
  A = kdec kinv^T (strict lower),  P = qdec kinv^T (lower incl diag)
  (I + tril(beta*A))^{-1} = (I+L)(I+L^2)(I+L^4)(I+L^8), L = -tril(beta*A)  [log-depth]
  loop V blocks: Skdec = kdec S0^T; u = inv @ (beta*(v - Skdec));
                 o = qdec S0^T + P u;  S_T = b_last*(S0 + u^T kinv)
Same public signature / u-kinv-b buffer contract as the other kvbuffer verify ops;
flush reuses kda_flush_kvbuffer unchanged.
"""

import logging

import torch
import triton
import triton.language as tl

from cula.ops.kda_decode import (
    _normalize_A_log,
    _normalize_dt_bias,
    _normalize_state_indices,
    _normalize_state_source,
    _prepare_output_tensor,
)
from cula.ops.kda_decode_mtp import _normalize_mtp_a

logger = logging.getLogger(__name__)


@triton.jit
def _kda_mtp_kvbuffer_gemm_kernel(
    q_ptr, k_ptr, v_ptr, a_ptr, b_ptr, o_ptr,
    A_log_ptr, dt_bias_ptr,
    h0_ptr, idx_ptr,
    u_ptr, kinv_ptr, bb_ptr,
    scale,
    T: tl.constexpr, H: tl.constexpr, HV: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BV: tl.constexpr,
    USE_L2NORM: tl.constexpr, DSU: tl.constexpr,
    EMIT: tl.constexpr, WRITE_UBUF: tl.constexpr,
    SP_BETA: tl.constexpr, SP_THR: tl.constexpr,
):
    pid = tl.program_id(0)
    i_n = pid // HV
    i_hv = pid % HV
    i_h = i_hv // (HV // H)

    t = tl.arange(0, BT)
    kk = tl.arange(0, K)
    tmask = t < T
    r = t[:, None]
    c = t[None, :]

    # ---- token-parallel elementwise: loads, l2norm, gating ----
    q = tl.load(q_ptr + ((i_n * T + t[:, None]) * H + i_h) * K + kk[None, :],
                mask=tmask[:, None], other=0.0).to(tl.float32)
    k = tl.load(k_ptr + ((i_n * T + t[:, None]) * H + i_h) * K + kk[None, :],
                mask=tmask[:, None], other=0.0).to(tl.float32)
    a = tl.load(a_ptr + ((i_n * T + t[:, None]) * HV + i_hv) * K + kk[None, :],
                mask=tmask[:, None], other=0.0).to(tl.float32)
    bgate = tl.load(b_ptr + (i_n * T + t) * HV + i_hv, mask=tmask, other=0.0).to(tl.float32)
    dtb = tl.load(dt_bias_ptr + i_hv * K + kk).to(tl.float32)
    eA = tl.exp(tl.load(A_log_ptr + i_hv).to(tl.float32))

    if USE_L2NORM:
        q = q / tl.sqrt(tl.sum(q * q, 1)[:, None] + 1e-6)
        k = k / tl.sqrt(tl.sum(k * k, 1)[:, None] + 1e-6)
    q = q * scale

    x = a + dtb[None, :]
    bx = SP_BETA * x
    sp = tl.where(bx <= SP_THR, tl.log(1.0 + tl.exp(bx)) / SP_BETA, x)
    logg = -eA * sp  # log g_t[k]
    logg = tl.where(tmask[:, None], logg, 0.0)
    lb = tl.cumsum(logg, axis=0)  # log cumulative decay
    bcum = tl.exp(lb)
    kdec = tl.where(tmask[:, None], k * bcum, 0.0)
    kinv = tl.where(tmask[:, None], k * tl.exp(-lb), 0.0)
    qdec = tl.where(tmask[:, None], q * bcum, 0.0)
    beta = tl.where(tmask, tl.sigmoid(bgate), 0.0)
    blast = tl.sum(tl.where(t[:, None] == T - 1, bcum, 0.0), 0)  # b_{T-1}[k]

    # ---- T x T intra-chunk matrices + log-depth triangular inverse ----
    Amat = tl.dot(kdec, tl.trans(kinv))
    Pmat = tl.dot(qdec, tl.trans(kinv))
    Pmat = tl.where(r >= c, Pmat, 0.0)
    eye = tl.where(r == c, 1.0, 0.0)
    L = tl.where(r > c, -beta[:, None] * Amat, 0.0)
    inv = eye + L  # (I - L)^{-1} = (I+L)(I+L^2)(I+L^4)(I+L^8), L nilpotent (BT=16)
    Lp = tl.dot(L, L)
    inv = inv + tl.dot(inv, Lp)
    Lp = tl.dot(Lp, Lp)
    inv = inv + tl.dot(inv, Lp)
    Lp = tl.dot(Lp, Lp)
    inv = inv + tl.dot(inv, Lp)

    if WRITE_UBUF:
        tl.store(kinv_ptr + ((i_n * T + t[:, None]) * HV + i_hv) * K + kk[None, :],
                 kinv, mask=tmask[:, None])
        tl.store(bb_ptr + ((i_n * T + t[:, None]) * HV + i_hv) * K + kk[None, :],
                 bcum, mask=tmask[:, None])

    # ---- V blocks: all-GEMM verify + one-shot state update ----
    cidx = tl.load(idx_ptr + i_n)
    has_state = cidx >= 0
    cbase = (tl.where(has_state, cidx, 0) * HV + i_hv).to(tl.int64) * V * K
    for iv in range(0, V, BV):
        vr = iv + tl.arange(0, BV)
        S0 = tl.load(h0_ptr + cbase + vr[:, None] * K + kk[None, :],
                     mask=has_state, other=0.0)  # [BV, K] fp32
        S0t = tl.trans(S0)
        vt = tl.load(v_ptr + ((i_n * T + t[:, None]) * HV + i_hv) * V + vr[None, :],
                     mask=tmask[:, None], other=0.0).to(tl.float32)
        Skdec = tl.dot(kdec, S0t)  # [BT, BV]
        u = tl.dot(inv, beta[:, None] * (vt - Skdec))
        if EMIT:
            ot = tl.dot(qdec, S0t) + tl.dot(Pmat, u)
            tl.store(o_ptr + ((i_n * T + t[:, None]) * HV + i_hv) * V + vr[None, :],
                     ot.to(o_ptr.dtype.element_ty), mask=tmask[:, None])
        if WRITE_UBUF:
            tl.store(u_ptr + ((i_n * T + t[:, None]) * HV + i_hv) * V + vr[None, :],
                     u, mask=tmask[:, None])
        if not DSU:
            Snew = blast[None, :] * (S0 + tl.dot(tl.trans(u), kinv))
            tl.store(h0_ptr + cbase + vr[:, None] * K + kk[None, :], Snew, mask=has_state)


def kda_decode_mtp_gemm_kvbuffer(
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    scale: float | None = None,
    use_qk_l2norm_in_kernel: bool = True,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
    out: torch.Tensor | None = None,
    disable_state_update: bool = True,
    emit_output: bool = True,
    u_buffer: torch.Tensor | None = None,
    kinv_buffer: torch.Tensor | None = None,
    b_buffer: torch.Tensor | None = None,
    bv: int = 64,
    num_warps: int = 4,
    num_stages: int = 2,
) -> torch.Tensor:
    """GEMM/tensor-core kvbuffer VERIFY (Triton). Drop-in for the other kvbuffer verify ops."""
    N, T, H, K = q.shape
    HV = v.shape[2]
    V = v.shape[3]
    write_ubuf = u_buffer is not None
    assert T <= 16, f"gemm-kvbuffer pads tokens to BT=16, needs T<=16, got {T}"
    assert V % bv == 0, f"V must be divisible by bv, got V={V}, bv={bv}"

    if scale is None:
        scale = K**-0.5

    h0_source, pool_size, _ = _normalize_state_source(
        initial_state_source, N=N, HV=HV, K=K, V=V, device=q.device, state_layout="vk",
    )
    a = _normalize_mtp_a(a, N=N, T=T, HV=HV, K=K)
    o = _prepare_output_tensor(q, out, (N, T, HV, V))
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    a = a.contiguous()
    b = b.contiguous()
    A_log = _normalize_A_log(A_log, HV)
    dt_bias = _normalize_dt_bias(dt_bias, HV, K)
    initial_state_indices = _normalize_state_indices(
        initial_state_indices, N=N, pool_size=pool_size, device=q.device
    )

    if write_ubuf:
        u_buf, kinv_buf, b_buf = u_buffer, kinv_buffer, b_buffer
    else:  # dummy 1-elem tensors; kernel never touches them when WRITE_UBUF=False
        u_buf = kinv_buf = b_buf = torch.empty(1, dtype=torch.float32, device=q.device)

    h0_flat = h0_source.view(pool_size * HV, V, K)
    grid = (N * HV,)
    _kda_mtp_kvbuffer_gemm_kernel[grid](
        q, k, v, a, b, o,
        A_log, dt_bias,
        h0_flat, initial_state_indices,
        u_buf, kinv_buf, b_buf,
        scale,
        T=T, H=H, HV=HV, K=K, V=V, BT=16, BV=bv,
        USE_L2NORM=use_qk_l2norm_in_kernel,
        DSU=disable_state_update,
        EMIT=emit_output, WRITE_UBUF=write_ubuf,
        SP_BETA=softplus_beta, SP_THR=softplus_threshold,
        num_warps=num_warps, num_stages=num_stages,
    )
    return o
