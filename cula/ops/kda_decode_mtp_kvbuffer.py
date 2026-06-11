"""CuTe DSL KDA MTP decode — KVBuffer / chunkwise parallel-verification variant.

KVBuffer paper's chunkwise verify form (https://arxiv.org/abs/2605.19049) as a new
operator vs the recurrent vk/kv ops in ``kda_decode_mtp.py``. The T draft tokens
are treated as ONE chunk: per-token outputs come from the FIXED input state S0 plus a
small T×T intra-chunk correction, and the state is updated once at the end — the
S0-matvecs are independent across tokens (no length-T serial chain), the latency angle
at small batch. Infra (grid N*HV*(V//BV), 1 warp/CTA, lane=K, float4 loads, butterfly
reduce-over-K) mirrors the production vk kernel for apples-to-apples comparison.

Chunkwise math (state S0[v,k], decay-first; matches the recurrent op):
    g_t[k]  = exp(-exp(A_log) * softplus(a_t[k] + dt_bias[k]))    # per channel
    b_t[k]  = prod_{i<=t} g_i[k]                                  # cumulative decay
    kdec_t  = k_norm_t * b_t ; kinv_t = k_norm_t / b_t ; qdec_t = q_scaled_t * b_t
    A[t,i]  = <kdec_t, kinv_i> (i<t)       P[t,i] = <qdec_t, kinv_i> (i<=t)
    u_t[v]  = beta_t * (v_t[v] - (S0 @ kdec_t)[v] - sum_{i<t} A[t,i] u_i[v])
    o_t[v]  = (S0 @ qdec_t)[v] + sum_{i<=t} P[t,i] u_i[v]
    S_T[v,k]= b_{T-1}[k] * (S0[v,k] + sum_i u_i[v] kinv_i[k])     # full accept
"""

import logging

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

from cula.ops.kda_decode import (
    TILE_K,
    _get_cached_stream,
    _normalize_A_log,
    _normalize_dt_bias,
    _normalize_state_indices,
    _normalize_state_source,
    _prepare_output_tensor,
)
from cula.ops.kda_decode_mtp import (
    VEC_SIZE,
    _normalize_mtp_a,
    _select_vk_bv,
)

logger = logging.getLogger(__name__)


@cute.kernel
def kda_mtp_kvbuffer_vk_kernel(
    h0_source: cute.Tensor,  # [pool*HV, V, K] fp32 (vk)
    A_log: cute.Tensor,
    a: cute.Tensor,
    dt_bias: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    b: cute.Tensor,
    o: cute.Tensor,
    h0_indices: cute.Tensor,
    u_buf: cute.Tensor,     # [N, T, HV, V] fp32  pseudo-value u_t[v] (written when write_ubuf)
    kinv_buf: cute.Tensor,  # [N, T, HV, K] fp32  kinv_t[k] = k_norm_t / b_t
    b_buf: cute.Tensor,     # [N, T, HV, K] fp32  cumulative decay b_t[k]
    vec_size: cutlass.Constexpr[int],
    num_v_tiles: cutlass.Constexpr[int],
    BV: cutlass.Constexpr[int],
    softplus_beta: cutlass.Constexpr[float],
    softplus_threshold: cutlass.Constexpr[float],
    scale: cutlass.Constexpr[float],
    HV: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    use_qk_l2norm: cutlass.Constexpr[bool],
    disable_state_update: cutlass.Constexpr[bool],
    emit_output: cutlass.Constexpr[bool],
    write_ubuf: cutlass.Constexpr[bool],  # write compact u/kinv/b to GMEM for flush rank-m rebuild
    fast_math: cutlass.Constexpr[bool],
):
    tidx, _, _ = cute.arch.thread_idx()
    lane = tidx  # 1 warp = 32 lanes

    bidx, _, _ = cute.arch.block_idx()
    i_v = bidx % num_v_tiles
    tmp = bidx // num_v_tiles
    i_hv = tmp % HV
    i_n = tmp // HV
    i_h = i_hv // (HV // H)

    cache_idx = h0_indices[i_n]
    r_exp_A = cute.exp(cutlass.Float32(A_log[i_hv]), fastmath=fast_math)

    # lane owns vec_size contiguous K channels (K[4*lane:4*lane+4]) across all BV V-cols;
    # r_h[vv*vec_size+c] = S0[i_v*BV+vv, vec_size*lane+c] (held fixed, overwritten to S_T at the end).
    r_h = cute.make_rmem_tensor(cute.make_layout((BV * vec_size,), stride=(1,)), cutlass.Float32)
    r_h4 = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    r_dtb = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    # lane-private decayed feature maps over all T tokens: kdec=k_norm*b, kinv=k_norm/b, qdec=q_scaled*b.
    kdec = cute.make_rmem_tensor(cute.make_layout((T * vec_size,), stride=(1,)), cutlass.Float32)
    kinv = cute.make_rmem_tensor(cute.make_layout((T * vec_size,), stride=(1,)), cutlass.Float32)
    qdec = cute.make_rmem_tensor(cute.make_layout((T * vec_size,), stride=(1,)), cutlass.Float32)
    r_beta = cute.make_rmem_tensor(cute.make_layout((T,), stride=(1,)), cutlass.Float32)
    r_u = cute.make_rmem_tensor(cute.make_layout((T * BV,), stride=(1,)), cutlass.Float32)
    b_run = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    r_qf = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    r_kf = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    # 2-stage pipeline: current (no suffix) + next (_n) double buffers prefetch t+1 q/k/a over token-t compute.
    r_qbf = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16)
    r_kbf = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16)
    r_abf = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16)
    r_qbf_n = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16)
    r_kbf_n = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16)
    r_abf_n = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16)
    r_redB = cute.make_rmem_tensor(cute.make_layout((T + BV,), stride=(1,)), cutlass.Float32)

    # Load state S0 (contiguous float4, coalesced + vectorized).
    if cache_idx >= 0:
        flat_state_idx = cache_idx * HV + i_hv
        for vv in cutlass.range_constexpr(BV):
            v_global = i_v * BV + vv
            h_tile = cute.local_tile(h0_source, (1, 1, vec_size), (flat_state_idx, v_global, lane))
            cute.autovec_copy(h_tile, r_h4)
            for c in cutlass.range_constexpr(vec_size):
                r_h[vv * vec_size + c] = r_h4[c]
    else:
        for j in cutlass.range_constexpr(BV * vec_size):
            r_h[j] = cutlass.Float32(0.0)

    for c in cutlass.range_constexpr(vec_size):  # dt_bias loaded once outside the loop
        r_dtb[c] = cutlass.Float32(dt_bias[i_hv, vec_size * lane + c])

    # Phase A: per token compute g_t -> cumulative b_t, l2norm/scale, kdec/kinv/qdec, beta_t.
    # Pipeline: preload token 0, issue t+1 q/k/a loads early each iter (overlap compute), swap at the end.
    for c in cutlass.range_constexpr(vec_size):
        b_run[c] = cutlass.Float32(1.0)
    q_tile0 = cute.local_tile(q, (1, 1, 1, vec_size), (i_n, 0, i_h, lane))
    k_tile0 = cute.local_tile(k, (1, 1, 1, vec_size), (i_n, 0, i_h, lane))
    a_tile0 = cute.local_tile(a, (1, 1, 1, vec_size), (i_n, 0, i_hv, lane))
    cute.autovec_copy(q_tile0, r_qbf)
    cute.autovec_copy(k_tile0, r_kbf)
    cute.autovec_copy(a_tile0, r_abf)
    for i_t in cutlass.range_constexpr(T):
        # prefetch t+1 q/k/a into the _n buffers (issued before compute -> load latency hidden)
        if cutlass.const_expr(i_t + 1 < T):
            q_tile_n = cute.local_tile(q, (1, 1, 1, vec_size), (i_n, i_t + 1, i_h, lane))
            k_tile_n = cute.local_tile(k, (1, 1, 1, vec_size), (i_n, i_t + 1, i_h, lane))
            a_tile_n = cute.local_tile(a, (1, 1, 1, vec_size), (i_n, i_t + 1, i_hv, lane))
            cute.autovec_copy(q_tile_n, r_qbf_n)
            cute.autovec_copy(k_tile_n, r_kbf_n)
            cute.autovec_copy(a_tile_n, r_abf_n)
        for c in cutlass.range_constexpr(vec_size):
            r_qf[c] = cutlass.Float32(r_qbf[c])
            r_kf[c] = cutlass.Float32(r_kbf[c])

        if cutlass.const_expr(use_qk_l2norm):
            sum_q = cutlass.Float32(0.0)
            sum_k = cutlass.Float32(0.0)
            for c in cutlass.range_constexpr(vec_size):
                sum_q += r_qf[c] * r_qf[c]
                sum_k += r_kf[c] * r_kf[c]
            for off in [16, 8, 4, 2, 1]:
                sum_q += cute.arch.shuffle_sync_bfly(sum_q, offset=off, mask=-1, mask_and_clamp=31)
                sum_k += cute.arch.shuffle_sync_bfly(sum_k, offset=off, mask=-1, mask_and_clamp=31)
            inv_q = cute.rsqrt(sum_q + 1e-6, fastmath=fast_math) * scale
            inv_k = cute.rsqrt(sum_k + 1e-6, fastmath=fast_math)
            for c in cutlass.range_constexpr(vec_size):
                r_qf[c] = r_qf[c] * inv_q
                r_kf[c] = r_kf[c] * inv_k
        else:
            for c in cutlass.range_constexpr(vec_size):
                r_qf[c] = r_qf[c] * scale

        # per-channel gate g_t -> accumulate b_run *= g_t (a read from the prefetched r_abf)
        for c in cutlass.range_constexpr(vec_size):
            x = cutlass.Float32(r_abf[c]) + r_dtb[c]
            beta_x = softplus_beta * x
            exp_bx = cute.exp(beta_x, fastmath=fast_math)
            sp_val = (cutlass.Float32(1.0) / softplus_beta) * cute.log(
                cutlass.Float32(1.0) + exp_bx, fastmath=fast_math
            )
            use_sp = (
                cutlass.Float32(1.0)
                if beta_x <= softplus_threshold
                else cutlass.Float32(0.0)
            )
            sp_x = use_sp * sp_val + (cutlass.Float32(1.0) - use_sp) * x
            g_c = cute.exp(-r_exp_A * sp_x, fastmath=fast_math)
            b_run[c] = b_run[c] * g_c
            kdec[i_t * vec_size + c] = r_kf[c] * b_run[c]
            kinv[i_t * vec_size + c] = r_kf[c] / b_run[c]
            qdec[i_t * vec_size + c] = r_qf[c] * b_run[c]

        r_beta[i_t] = cutlass.Float32(1.0) / (
            cutlass.Float32(1.0) + cute.exp(-cutlass.Float32(b[i_n, i_t, i_hv]), fastmath=fast_math)
        )

        # u-buffer: kinv_t / b_t are v-tile-independent -> only the i_v==0 CTA writes them (float4).
        if cutlass.const_expr(write_ubuf):
            if i_v == 0:
                for c in cutlass.range_constexpr(vec_size):
                    r_h4[c] = kinv[i_t * vec_size + c]
                kinv_out = cute.local_tile(kinv_buf, (1, 1, 1, vec_size), (i_n, i_t, i_hv, lane))
                cute.autovec_copy(r_h4, kinv_out)
                for c in cutlass.range_constexpr(vec_size):
                    r_h4[c] = b_run[c]
                b_out = cute.local_tile(b_buf, (1, 1, 1, vec_size), (i_n, i_t, i_hv, lane))
                cute.autovec_copy(r_h4, b_out)

        # swap next -> current (register moves; the prefetched t+1 becomes next iter's current)
        if cutlass.const_expr(i_t + 1 < T):
            for c in cutlass.range_constexpr(vec_size):
                r_qbf[c] = r_qbf_n[c]
                r_kbf[c] = r_kbf_n[c]
                r_abf[c] = r_abf_n[c]
    # after the loop b_run[c] = b_{T-1}[c] (used by the final state update).

    # Phase B: forward-subst u_t = beta_t*(v_t - S0@kdec_t - sum_{i<t} A[t,i] u_i).
    for i_t in cutlass.range_constexpr(T):
        for i_i in cutlass.range_constexpr(i_t):
            s = cutlass.Float32(0.0)
            for c in cutlass.range_constexpr(vec_size):
                s += kdec[i_t * vec_size + c] * kinv[i_i * vec_size + c]
            r_redB[i_i] = s
        for vv in cutlass.range_constexpr(BV):
            s = cutlass.Float32(0.0)
            for c in cutlass.range_constexpr(vec_size):
                s += r_h[vv * vec_size + c] * kdec[i_t * vec_size + c]
            r_redB[i_t + vv] = s
        for off in [16, 8, 4, 2, 1]:
            for j in cutlass.range_constexpr(i_t + BV):
                r_redB[j] = r_redB[j] + cute.arch.shuffle_sync_bfly(r_redB[j], offset=off, mask=-1, mask_and_clamp=31)
        for vv in cutlass.range_constexpr(BV):
            acc = cutlass.Float32(v[i_n, i_t, i_hv, i_v * BV + vv]) - r_redB[i_t + vv]
            for i_i in cutlass.range_constexpr(i_t):
                acc -= r_redB[i_i] * r_u[i_i * BV + vv]
            r_u[i_t * BV + vv] = r_beta[i_t] * acc

    # u-buffer: u_t[v] is this CTA's v-column slice; lane vv writes column vv (coalesced).
    if cutlass.const_expr(write_ubuf):
        if lane < BV:
            for i_t in cutlass.range_constexpr(T):
                u_buf[i_n, i_t, i_hv, i_v * BV + lane] = r_u[i_t * BV + lane]

    # Phase C: output o_t = S0@qdec_t + sum_{i<=t} P[t,i] u_i.
    if cutlass.const_expr(emit_output):
        for i_t in cutlass.range_constexpr(T):
            for i_i in cutlass.range_constexpr(i_t + 1):
                s = cutlass.Float32(0.0)
                for c in cutlass.range_constexpr(vec_size):
                    s += qdec[i_t * vec_size + c] * kinv[i_i * vec_size + c]
                r_redB[i_i] = s
            for vv in cutlass.range_constexpr(BV):
                s = cutlass.Float32(0.0)
                for c in cutlass.range_constexpr(vec_size):
                    s += r_h[vv * vec_size + c] * qdec[i_t * vec_size + c]
                r_redB[(i_t + 1) + vv] = s
            for off in [16, 8, 4, 2, 1]:
                for j in cutlass.range_constexpr((i_t + 1) + BV):
                    r_redB[j] = r_redB[j] + cute.arch.shuffle_sync_bfly(r_redB[j], offset=off, mask=-1, mask_and_clamp=31)
            for vv in cutlass.range_constexpr(BV):
                ov = r_redB[(i_t + 1) + vv]
                for i_i in cutlass.range_constexpr(i_t + 1):
                    ov += r_redB[i_i] * r_u[i_i * BV + vv]
                o[(i_n, i_t, i_hv, i_v * BV + vv)] = cutlass.BFloat16(ov)  # all-reduced -> idempotent

    # Phase D: one-shot final state S_T[v,k] = b_{T-1}[k] * (S0[v,k] + sum_t u_t[v] * kinv_t[k]).
    if cache_idx >= 0:
        if cutlass.const_expr(not disable_state_update):
            flat_state_idx = cache_idx * HV + i_hv
            for vv in cutlass.range_constexpr(BV):
                v_global = i_v * BV + vv
                for c in cutlass.range_constexpr(vec_size):
                    acc = r_h[vv * vec_size + c]
                    for i_t in cutlass.range_constexpr(T):
                        acc += r_u[i_t * BV + vv] * kinv[i_t * vec_size + c]
                    r_h4[c] = b_run[c] * acc
                h_out = cute.local_tile(h0_source, (1, 1, vec_size), (flat_state_idx, v_global, lane))
                cute.autovec_copy(r_h4, h_out)


@cute.jit
def run_kda_mtp_kvbuffer_vk_kernel(
    h0_source: cute.Tensor,
    A_log: cute.Tensor,
    a: cute.Tensor,
    dt_bias: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    b: cute.Tensor,
    o: cute.Tensor,
    h0_indices: cute.Tensor,
    u_buf: cute.Tensor,
    kinv_buf: cute.Tensor,
    b_buf: cute.Tensor,
    vec_size: cutlass.Constexpr[int],
    BV: cutlass.Constexpr[int],
    softplus_beta: cutlass.Constexpr[float],
    softplus_threshold: cutlass.Constexpr[float],
    scale: cutlass.Constexpr[float],
    HV: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    use_qk_l2norm: cutlass.Constexpr[bool],
    disable_state_update: cutlass.Constexpr[bool],
    emit_output: cutlass.Constexpr[bool],
    write_ubuf: cutlass.Constexpr[bool],
    fast_math: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    """vk-kvbuffer launcher: grid = N*HV*(V//BV), block = 32 (1 warp), no SMEM."""
    n_indices = h0_indices.layout.shape[0]
    num_v_tiles = cute.ceil_div(V, BV)
    grid_size = n_indices * HV * num_v_tiles

    kda_mtp_kvbuffer_vk_kernel(
        h0_source,
        A_log,
        a,
        dt_bias,
        q,
        k,
        v,
        b,
        o,
        h0_indices,
        u_buf,
        kinv_buf,
        b_buf,
        vec_size,
        num_v_tiles,
        BV,
        softplus_beta,
        softplus_threshold,
        scale,
        HV,
        T,
        H,
        K,
        V,
        use_qk_l2norm,
        disable_state_update,
        emit_output,
        write_ubuf,
        fast_math,
    ).launch(
        grid=(grid_size, 1, 1),
        block=[32, 1, 1],
        smem=0,
        stream=stream,
    )


_compiled_mtp_kvbuffer_kernels: dict[tuple, object] = {}


def _get_compiled_mtp_kvbuffer_kernel(
    N,
    T,
    H,
    HV,
    K,
    V,
    pool_size,
    BV,
    scale,
    use_qk_l2norm,
    disable_state_update,
    emit_output,
    write_ubuf,
    softplus_beta,
    softplus_threshold,
    opt_level=3,
    fast_math=True,
):
    key = (
        N,
        T,
        H,
        HV,
        K,
        V,
        pool_size,
        BV,
        scale,
        use_qk_l2norm,
        disable_state_update,
        emit_output,
        write_ubuf,
        softplus_beta,
        softplus_threshold,
        opt_level,
        fast_math,
    )
    if key in _compiled_mtp_kvbuffer_kernels:
        return _compiled_mtp_kvbuffer_kernels[key]

    q = torch.zeros(N, T, H, K, dtype=torch.bfloat16, device="cuda")
    k = torch.zeros(N, T, H, K, dtype=torch.bfloat16, device="cuda")
    v = torch.zeros(N, T, HV, V, dtype=torch.bfloat16, device="cuda")
    a = torch.zeros(N, T, HV, K, dtype=torch.bfloat16, device="cuda")
    b = torch.zeros(N, T, HV, dtype=torch.bfloat16, device="cuda")
    o = torch.zeros(N, T, HV, V, dtype=torch.bfloat16, device="cuda")
    A_log = torch.zeros(HV, dtype=torch.float32, device="cuda")
    dt_bias = torch.zeros(HV, K, dtype=torch.float32, device="cuda")
    h0_source = torch.zeros(pool_size * HV, V, K, dtype=torch.float32, device="cuda")
    h0_indices = torch.zeros(N, dtype=torch.int32, device="cuda")
    u_buf = torch.zeros(N, T, HV, V, dtype=torch.float32, device="cuda")
    kinv_buf = torch.zeros(N, T, HV, K, dtype=torch.float32, device="cuda")
    b_buf = torch.zeros(N, T, HV, K, dtype=torch.float32, device="cuda")

    q_t = from_dlpack(q, assumed_align=16)
    k_t = from_dlpack(k, assumed_align=16)
    v_t = from_dlpack(v, assumed_align=16)
    a_t = from_dlpack(a, assumed_align=16)
    b_t = from_dlpack(b, assumed_align=16)
    o_t = from_dlpack(o, assumed_align=16)
    A_log_t = from_dlpack(A_log, assumed_align=16)
    dt_bias_t = from_dlpack(dt_bias, assumed_align=16)
    h0_source_t = from_dlpack(h0_source, assumed_align=16)
    h0_indices_t = from_dlpack(h0_indices, assumed_align=16)
    u_buf_t = from_dlpack(u_buf, assumed_align=16)
    kinv_buf_t = from_dlpack(kinv_buf, assumed_align=16)
    b_buf_t = from_dlpack(b_buf, assumed_align=16)

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    compiled_kernel = cute.compile(
        run_kda_mtp_kvbuffer_vk_kernel,
        h0_source_t,
        A_log_t,
        a_t,
        dt_bias_t,
        q_t,
        k_t,
        v_t,
        b_t,
        o_t,
        h0_indices_t,
        u_buf_t,
        kinv_buf_t,
        b_buf_t,
        vec_size=VEC_SIZE,
        BV=BV,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        scale=scale,
        HV=HV,
        T=T,
        H=H,
        K=K,
        V=V,
        use_qk_l2norm=use_qk_l2norm,
        disable_state_update=disable_state_update,
        emit_output=emit_output,
        write_ubuf=write_ubuf,
        fast_math=fast_math,
        stream=stream,
        options=f"--enable-tvm-ffi --opt-level {opt_level}",
    )

    _compiled_mtp_kvbuffer_kernels[key] = compiled_kernel
    logger.info(
        "CuTe DSL KDA MTP KVBuffer(lane=K, chunkwise) kernel compiled: "
        f"N={N}, T={T}, H={H}, HV={HV}, K={K}, V={V}, pool_size={pool_size}, BV={BV}, "
        f"opt_level={opt_level}, fast_math={fast_math}"
    )
    return compiled_kernel


def kda_decode_mtp_kvbuffer(
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
    bv: int = -1,
    opt_level: int = 3,
    fast_math: bool = True,
) -> torch.Tensor:
    N, T, H, K = q.shape
    HV = v.shape[2]
    V = v.shape[3]
    write_ubuf = u_buffer is not None

    if scale is None:
        scale = K**-0.5
    else:
        assert scale > 0, f"scale must be positive, got {scale}"

    assert K == TILE_K, f"KDA MTP (kvbuffer) requires K={TILE_K}, got {K}"
    assert K % VEC_SIZE == 0 and K // VEC_SIZE == 32, (
        f"kvbuffer assumes K//vec_size==32 (one warp), got K={K}, vec_size={VEC_SIZE}"
    )

    if bv <= 0:  # auto: reuse vk's BV heuristic (smaller BV at small batch to fill the grid)
        num_sms = torch.cuda.get_device_properties(q.device).multi_processor_count
        bv = _select_vk_bv(N * HV, V, num_sms)
    assert bv in (8, 16, 32), f"kvbuffer bv must be 8/16/32 or <=0 (auto), got {bv}"
    assert V % bv == 0, f"kvbuffer requires V % bv == 0, got V={V}, bv={bv}"

    h0_source, pool_size, _ = _normalize_state_source(
        initial_state_source, N=N, HV=HV, K=K, V=V, device=q.device, state_layout="vk",
    )

    a = _normalize_mtp_a(a, N=N, T=T, HV=HV, K=K)
    if b.dim() != 3 or tuple(b.shape) != (N, T, HV):
        raise ValueError(f"Unexpected b shape for MTP dense: {tuple(b.shape)}; expected {(N, T, HV)}")

    o = _prepare_output_tensor(q, out, (N, T, HV, V))

    q = q if q.is_contiguous() else q.contiguous()
    k = k if k.is_contiguous() else k.contiguous()
    v = v if v.is_contiguous() else v.contiguous()
    a = a if a.is_contiguous() else a.contiguous()
    b = b if b.is_contiguous() else b.contiguous()

    A_log = _normalize_A_log(A_log, HV)
    dt_bias = _normalize_dt_bias(dt_bias, HV, K)
    initial_state_indices = _normalize_state_indices(
        initial_state_indices, N=N, pool_size=pool_size, device=q.device
    )

    if write_ubuf:
        if tuple(u_buffer.shape) != (N, T, HV, V):
            raise ValueError(f"u_buffer shape must be {(N, T, HV, V)}, got {tuple(u_buffer.shape)}")
        if tuple(kinv_buffer.shape) != (N, T, HV, K) or tuple(b_buffer.shape) != (N, T, HV, K):
            raise ValueError(f"kinv_buffer/b_buffer shape must be {(N, T, HV, K)}")
        u_buf, kinv_buf, b_buf = u_buffer, kinv_buffer, b_buffer
    else:  # placeholders (kernel has write_ubuf=False, never touches them)
        u_buf = torch.zeros(N, T, HV, V, dtype=torch.float32, device=q.device)
        kinv_buf = torch.zeros(N, T, HV, K, dtype=torch.float32, device=q.device)
        b_buf = torch.zeros(N, T, HV, K, dtype=torch.float32, device=q.device)

    stream = _get_cached_stream(q.device)

    h0_source_flat = h0_source.view(pool_size * HV, V, K)  # vk
    compiled_kernel = _get_compiled_mtp_kvbuffer_kernel(
        N, T, H, HV, K, V, pool_size, bv,
        scale=scale, use_qk_l2norm=use_qk_l2norm_in_kernel,
        disable_state_update=disable_state_update, emit_output=emit_output,
        write_ubuf=write_ubuf,
        softplus_beta=softplus_beta, softplus_threshold=softplus_threshold,
        opt_level=opt_level, fast_math=fast_math,
    )

    compiled_kernel(
        h0_source_flat,
        A_log,
        a,
        dt_bias,
        q,
        k,
        v,
        b,
        o,
        initial_state_indices,
        u_buf,
        kinv_buf,
        b_buf,
        stream,
    )

    return o


@cute.kernel
def kda_mtp_ws_kvbuffer_kernel(
    h0_source: cute.Tensor,  # [pool*HV, V, K] fp32 (vk)
    A_log: cute.Tensor,
    a: cute.Tensor,
    dt_bias: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    b: cute.Tensor,
    o: cute.Tensor,
    h0_indices: cute.Tensor,
    u_buf: cute.Tensor,     # [N, T, HV, V] fp32
    kinv_buf: cute.Tensor,  # [N, T, HV, K] fp32
    b_buf: cute.Tensor,     # [N, T, HV, K] fp32
    vec_size: cutlass.Constexpr[int],
    num_v_tiles: cutlass.Constexpr[int],
    tile_v: cutlass.Constexpr[int],
    ilp_rows: cutlass.Constexpr[int],
    softplus_beta: cutlass.Constexpr[float],
    softplus_threshold: cutlass.Constexpr[float],
    scale: cutlass.Constexpr[float],
    HV: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    use_qk_l2norm: cutlass.Constexpr[bool],
    disable_state_update: cutlass.Constexpr[bool],
    emit_output: cutlass.Constexpr[bool],
    write_ubuf: cutlass.Constexpr[bool],
    use_smem_v: cutlass.Constexpr[bool],
    fast_math: cutlass.Constexpr[bool],
):
    tidx, _, _ = cute.arch.thread_idx()
    lane_id = tidx % 32
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)

    threads_per_group: cutlass.Constexpr[int] = K // vec_size  # 32 = full warp
    num_groups: cutlass.Constexpr[int] = 4
    lane_in_group = lane_id  # threads_per_group==32 → lane_in_group==lane
    group_idx = warp_idx

    bidx, _, _ = cute.arch.block_idx()
    i_v = bidx % num_v_tiles
    tmp = bidx // num_v_tiles
    i_hv = tmp % HV
    i_n = tmp // HV
    i_h = i_hv // (HV // H)

    cache_idx = h0_indices[i_n]
    r_exp_A = cute.exp(cutlass.Float32(A_log[i_hv]), fastmath=fast_math)

    # SMEM: warp0 producer broadcasts to all consumer warps.
    smem = cutlass.utils.SmemAllocator()
    sKdec = smem.allocate_tensor(cutlass.Float32, cute.make_layout((T, K), stride=(K + 8, 1)), 16)
    sKinv = smem.allocate_tensor(cutlass.Float32, cute.make_layout((T, K), stride=(K + 8, 1)), 16)
    sQdec = smem.allocate_tensor(cutlass.Float32, cute.make_layout((T, K), stride=(K + 8, 1)), 16)
    sBeta = smem.allocate_tensor(cutlass.Float32, cute.make_layout((T,)), 16)
    sBlast = smem.allocate_tensor(cutlass.Float32, cute.make_layout((K,)), 16)  # b_{T-1}[k]
    sA = smem.allocate_tensor(cutlass.Float32, cute.make_layout((T, T), stride=(T, 1)), 16)  # A[t,i] i<t
    sP = smem.allocate_tensor(cutlass.Float32, cute.make_layout((T, T), stride=(T, 1)), 16)  # P[t,i] i<=t
    if cutlass.const_expr(use_smem_v):
        sVdata = smem.allocate_tensor(cutlass.Float32, cute.make_layout((T, tile_v), stride=(tile_v, 1)), 16)

    # Registers (declared at the top for all warps).
    r_qbf = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16)
    r_kbf = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16)
    r_qf = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    r_kf = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    r_dtb = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    b_run = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    r_tmp = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    # consumer: hold ilp_rows rows of S0 K-channels together (batched butterfly); r_part = per-row partials.
    r_h = cute.make_rmem_tensor(cute.make_layout((ilp_rows, vec_size), stride=(vec_size, 1)), cutlass.Float32)
    r_u = cute.make_rmem_tensor(cute.make_layout((ilp_rows, T), stride=(T, 1)), cutlass.Float32)
    r_part = cute.make_rmem_tensor(cute.make_layout((ilp_rows,), stride=(1,)), cutlass.Float32)

    if cache_idx >= 0:
        k_start = lane_in_group * vec_size
        rows_per_group: cutlass.Constexpr[int] = tile_v // num_groups
        flat_state_idx = cache_idx * HV + i_hv

        # warp0 producer
        if warp_idx == 0:
            for c in cutlass.range_constexpr(vec_size):
                r_dtb[c] = cutlass.Float32(dt_bias[i_hv, k_start + c])
                b_run[c] = cutlass.Float32(1.0)
            for i_t in cutlass.range_constexpr(T):
                q_tile = cute.local_tile(q, (1, 1, 1, vec_size), (i_n, i_t, i_h, lane_in_group))
                k_tile = cute.local_tile(k, (1, 1, 1, vec_size), (i_n, i_t, i_h, lane_in_group))
                cute.autovec_copy(q_tile, r_qbf)
                cute.autovec_copy(k_tile, r_kbf)
                for c in cutlass.range_constexpr(vec_size):
                    r_qf[c] = cutlass.Float32(r_qbf[c])
                    r_kf[c] = cutlass.Float32(r_kbf[c])

                if cutlass.const_expr(use_qk_l2norm):
                    sum_q = cutlass.Float32(0.0)
                    sum_k = cutlass.Float32(0.0)
                    for c in cutlass.range_constexpr(vec_size):
                        sum_q += r_qf[c] * r_qf[c]
                        sum_k += r_kf[c] * r_kf[c]
                    for off in [16, 8, 4, 2, 1]:
                        sum_q += cute.arch.shuffle_sync_bfly(sum_q, offset=off, mask=-1, mask_and_clamp=31)
                        sum_k += cute.arch.shuffle_sync_bfly(sum_k, offset=off, mask=-1, mask_and_clamp=31)
                    inv_q = cute.rsqrt(sum_q + 1e-6, fastmath=fast_math) * scale
                    inv_k = cute.rsqrt(sum_k + 1e-6, fastmath=fast_math)
                    for c in cutlass.range_constexpr(vec_size):
                        r_qf[c] = r_qf[c] * inv_q
                        r_kf[c] = r_kf[c] * inv_k
                else:
                    for c in cutlass.range_constexpr(vec_size):
                        r_qf[c] = r_qf[c] * scale

                for c in cutlass.range_constexpr(vec_size):
                    x = cutlass.Float32(a[i_n, i_t, i_hv, k_start + c]) + r_dtb[c]
                    beta_x = softplus_beta * x
                    exp_bx = cute.exp(beta_x, fastmath=fast_math)
                    sp_val = (cutlass.Float32(1.0) / softplus_beta) * cute.log(
                        cutlass.Float32(1.0) + exp_bx, fastmath=fast_math
                    )
                    use_sp = (
                        cutlass.Float32(1.0)
                        if beta_x <= softplus_threshold
                        else cutlass.Float32(0.0)
                    )
                    sp_x = use_sp * sp_val + (cutlass.Float32(1.0) - use_sp) * x
                    g_c = cute.exp(-r_exp_A * sp_x, fastmath=fast_math)
                    b_run[c] = b_run[c] * g_c
                    sKdec[i_t, k_start + c] = r_kf[c] * b_run[c]
                    sKinv[i_t, k_start + c] = r_kf[c] / b_run[c]
                    sQdec[i_t, k_start + c] = r_qf[c] * b_run[c]

                sBeta[i_t] = cutlass.Float32(1.0) / (
                    cutlass.Float32(1.0) + cute.exp(-cutlass.Float32(b[i_n, i_t, i_hv]), fastmath=fast_math)
                )

                if cutlass.const_expr(write_ubuf):
                    for c in cutlass.range_constexpr(vec_size):
                        r_tmp[c] = sKinv[i_t, k_start + c]
                    kinv_out = cute.local_tile(kinv_buf, (1, 1, 1, vec_size), (i_n, i_t, i_hv, lane_in_group))
                    cute.autovec_copy(r_tmp, kinv_out)
                    for c in cutlass.range_constexpr(vec_size):
                        r_tmp[c] = b_run[c]
                    b_out = cute.local_tile(b_buf, (1, 1, 1, vec_size), (i_n, i_t, i_hv, lane_in_group))
                    cute.autovec_copy(r_tmp, b_out)

            for c in cutlass.range_constexpr(vec_size):
                sBlast[k_start + c] = b_run[c]

            # A[t,i]=<kdec_t,kinv_i> (i<t), P[t,i]=<qdec_t,kinv_i> (i<=t) from SMEM, butterfly reduce.
            for i_t in cutlass.range_constexpr(T):
                for i_i in cutlass.range_constexpr(i_t):
                    aij = cutlass.Float32(0.0)
                    for c in cutlass.range_constexpr(vec_size):
                        aij += sKdec[i_t, k_start + c] * sKinv[i_i, k_start + c]
                    for off in [16, 8, 4, 2, 1]:
                        aij += cute.arch.shuffle_sync_bfly(aij, offset=off, mask=-1, mask_and_clamp=31)
                    sA[i_t, i_i] = aij
                for i_i in cutlass.range_constexpr(i_t + 1):
                    pij = cutlass.Float32(0.0)
                    for c in cutlass.range_constexpr(vec_size):
                        pij += sQdec[i_t, k_start + c] * sKinv[i_i, k_start + c]
                    for off in [16, 8, 4, 2, 1]:
                        pij += cute.arch.shuffle_sync_bfly(pij, offset=off, mask=-1, mask_and_clamp=31)
                    sP[i_t, i_i] = pij

        if cutlass.const_expr(use_smem_v):
            for i_t in cutlass.range_constexpr(T):
                if tidx < tile_v:
                    sVdata[i_t, tidx] = cutlass.Float32(v[i_n, i_t, i_hv, i_v * tile_v + tidx])

        # publish warp0's SMEM to all warps
        cute.arch.barrier()

        n_row_groups: cutlass.Constexpr[int] = rows_per_group // ilp_rows
        for rg in cutlass.range_constexpr(n_row_groups):
            v_base = i_v * tile_v + group_idx * rows_per_group + rg * ilp_rows
            for r in cutlass.range_constexpr(ilp_rows):
                h_tile = cute.local_tile(
                    h0_source, (1, 1, vec_size), (flat_state_idx, v_base + r, lane_in_group)
                )
                cute.autovec_copy(h_tile, cute.slice_(r_h, (r, None)))
            # forward-subst: per token batch-compute Skdec_t=S0@kdec_t, then per-row fwd-subst u_t.
            for i_t in cutlass.range_constexpr(T):
                for r in cutlass.range_constexpr(ilp_rows):
                    sk = cutlass.Float32(0.0)
                    for c in cutlass.range_constexpr(vec_size):
                        sk += r_h[r, c] * sKdec[i_t, k_start + c]
                    r_part[r] = sk
                for off in [16, 8, 4, 2, 1]:
                    for r in cutlass.range_constexpr(ilp_rows):
                        r_part[r] += cute.arch.shuffle_sync_bfly(r_part[r], offset=off, mask=-1, mask_and_clamp=31)
                v_local_base = group_idx * rows_per_group + rg * ilp_rows  # tile-local col of v_base
                for r in cutlass.range_constexpr(ilp_rows):
                    if cutlass.const_expr(use_smem_v):
                        r_vt = sVdata[i_t, v_local_base + r]
                    else:
                        r_vt = cutlass.Float32(v[i_n, i_t, i_hv, v_base + r])
                    acc = r_vt - r_part[r]
                    for i_i in cutlass.range_constexpr(i_t):
                        acc -= sA[i_t, i_i] * r_u[r, i_i]
                    r_u[r, i_t] = sBeta[i_t] * acc
            # u-buffer: u_t replicated on all lanes, lane0 writes once.
            if cutlass.const_expr(write_ubuf):
                if lane_in_group == 0:
                    for r in cutlass.range_constexpr(ilp_rows):
                        for i_t in cutlass.range_constexpr(T):
                            u_buf[i_n, i_t, i_hv, v_base + r] = r_u[r, i_t]
            # output o_t = Sqdec_t + sum_{i<=t} P[t,i] u_i (Sqdec also batched butterfly).
            if cutlass.const_expr(emit_output):
                for i_t in cutlass.range_constexpr(T):
                    for r in cutlass.range_constexpr(ilp_rows):
                        sq = cutlass.Float32(0.0)
                        for c in cutlass.range_constexpr(vec_size):
                            sq += r_h[r, c] * sQdec[i_t, k_start + c]
                        r_part[r] = sq
                    for off in [16, 8, 4, 2, 1]:
                        for r in cutlass.range_constexpr(ilp_rows):
                            r_part[r] += cute.arch.shuffle_sync_bfly(r_part[r], offset=off, mask=-1, mask_and_clamp=31)
                    for r in cutlass.range_constexpr(ilp_rows):
                        ov = r_part[r]
                        for i_i in cutlass.range_constexpr(i_t + 1):
                            ov += sP[i_t, i_i] * r_u[r, i_i]
                        if lane_in_group == 0:
                            o[(i_n, i_t, i_hv, v_base + r)] = cutlass.BFloat16(ov)
            # final state S_T[v,k] = b_{T-1}[k]*(S0[v,k] + sum_t u_t kinv_t[k]), written per row.
            if cutlass.const_expr(not disable_state_update):
                for r in cutlass.range_constexpr(ilp_rows):
                    for c in cutlass.range_constexpr(vec_size):
                        acc = r_h[r, c]
                        for i_t in cutlass.range_constexpr(T):
                            acc += r_u[r, i_t] * sKinv[i_t, k_start + c]
                        r_tmp[c] = sBlast[k_start + c] * acc
                    h_out = cute.local_tile(
                        h0_source, (1, 1, vec_size), (flat_state_idx, v_base + r, lane_in_group)
                    )
                    cute.autovec_copy(r_tmp, h_out)


@cute.jit
def run_kda_mtp_ws_kvbuffer_kernel(
    h0_source: cute.Tensor,
    A_log: cute.Tensor,
    a: cute.Tensor,
    dt_bias: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    b: cute.Tensor,
    o: cute.Tensor,
    h0_indices: cute.Tensor,
    u_buf: cute.Tensor,
    kinv_buf: cute.Tensor,
    b_buf: cute.Tensor,
    vec_size: cutlass.Constexpr[int],
    tile_v: cutlass.Constexpr[int],
    ilp_rows: cutlass.Constexpr[int],
    softplus_beta: cutlass.Constexpr[float],
    softplus_threshold: cutlass.Constexpr[float],
    scale: cutlass.Constexpr[float],
    HV: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    use_qk_l2norm: cutlass.Constexpr[bool],
    disable_state_update: cutlass.Constexpr[bool],
    emit_output: cutlass.Constexpr[bool],
    write_ubuf: cutlass.Constexpr[bool],
    use_smem_v: cutlass.Constexpr[bool],
    fast_math: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    """ws-kvbuffer launcher:grid = N*HV*(V//tile_v),block = 128(4 warp)。"""
    n_indices = h0_indices.layout.shape[0]
    num_v_tiles = cute.ceil_div(V, tile_v)
    grid_size = n_indices * HV * num_v_tiles
    smem_bytes = (
        3 * 4 * T * (K + 8)  # sKdec/sKinv/sQdec
        + 4 * T  # sBeta
        + 4 * K  # sBlast
        + 2 * 4 * T * T  # sA/sP
        + (4 * T * tile_v if use_smem_v else 0)  # sVdata
        + 256  # alignment slack
    )
    kda_mtp_ws_kvbuffer_kernel(
        h0_source, A_log, a, dt_bias, q, k, v, b, o, h0_indices,
        u_buf, kinv_buf, b_buf,
        vec_size, num_v_tiles, tile_v, ilp_rows,
        softplus_beta, softplus_threshold, scale,
        HV, T, H, K, V,
        use_qk_l2norm, disable_state_update, emit_output, write_ubuf, use_smem_v, fast_math,
    ).launch(grid=(grid_size, 1, 1), block=[128, 1, 1], smem=smem_bytes, stream=stream)


_compiled_mtp_ws_kvbuffer_kernels: dict[tuple, object] = {}


def _get_compiled_mtp_ws_kvbuffer_kernel(
    N, T, H, HV, K, V, pool_size, tile_v, ilp_rows, scale, use_qk_l2norm,
    disable_state_update, emit_output, write_ubuf, use_smem_v,
    softplus_beta, softplus_threshold, opt_level=3, fast_math=True,
):
    key = (
        N, T, H, HV, K, V, pool_size, tile_v, ilp_rows, scale, use_qk_l2norm,
        disable_state_update, emit_output, write_ubuf, use_smem_v,
        softplus_beta, softplus_threshold, opt_level, fast_math,
    )
    if key in _compiled_mtp_ws_kvbuffer_kernels:
        return _compiled_mtp_ws_kvbuffer_kernels[key]

    q = torch.zeros(N, T, H, K, dtype=torch.bfloat16, device="cuda")
    k = torch.zeros(N, T, H, K, dtype=torch.bfloat16, device="cuda")
    v = torch.zeros(N, T, HV, V, dtype=torch.bfloat16, device="cuda")
    a = torch.zeros(N, T, HV, K, dtype=torch.bfloat16, device="cuda")
    b = torch.zeros(N, T, HV, dtype=torch.bfloat16, device="cuda")
    o = torch.zeros(N, T, HV, V, dtype=torch.bfloat16, device="cuda")
    A_log = torch.zeros(HV, dtype=torch.float32, device="cuda")
    dt_bias = torch.zeros(HV, K, dtype=torch.float32, device="cuda")
    h0_source = torch.zeros(pool_size * HV, V, K, dtype=torch.float32, device="cuda")
    h0_indices = torch.zeros(N, dtype=torch.int32, device="cuda")
    u_buf = torch.zeros(N, T, HV, V, dtype=torch.float32, device="cuda")
    kinv_buf = torch.zeros(N, T, HV, K, dtype=torch.float32, device="cuda")
    b_buf = torch.zeros(N, T, HV, K, dtype=torch.float32, device="cuda")

    compiled_kernel = cute.compile(
        run_kda_mtp_ws_kvbuffer_kernel,
        from_dlpack(h0_source, assumed_align=16),
        from_dlpack(A_log, assumed_align=16),
        from_dlpack(a, assumed_align=16),
        from_dlpack(dt_bias, assumed_align=16),
        from_dlpack(q, assumed_align=16),
        from_dlpack(k, assumed_align=16),
        from_dlpack(v, assumed_align=16),
        from_dlpack(b, assumed_align=16),
        from_dlpack(o, assumed_align=16),
        from_dlpack(h0_indices, assumed_align=16),
        from_dlpack(u_buf, assumed_align=16),
        from_dlpack(kinv_buf, assumed_align=16),
        from_dlpack(b_buf, assumed_align=16),
        vec_size=VEC_SIZE,
        tile_v=tile_v,
        ilp_rows=ilp_rows,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        scale=scale,
        HV=HV, T=T, H=H, K=K, V=V,
        use_qk_l2norm=use_qk_l2norm,
        disable_state_update=disable_state_update,
        emit_output=emit_output,
        write_ubuf=write_ubuf,
        use_smem_v=use_smem_v,
        fast_math=fast_math,
        stream=cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        options=f"--enable-tvm-ffi --opt-level {opt_level}",
    )
    _compiled_mtp_ws_kvbuffer_kernels[key] = compiled_kernel
    logger.info(
        "CuTe DSL KDA MTP ws-KVBuffer kernel compiled: "
        f"N={N}, T={T}, HV={HV}, K={K}, V={V}, tile_v={tile_v}, ilp_rows={ilp_rows}, "
        f"use_smem_v={use_smem_v}, opt_level={opt_level}, fast_math={fast_math}"
    )
    return compiled_kernel


def _select_ws_kvb_tile_v(V, N):
    """N-dependent tile_v (H200 sweep, H=HV=32 K=V=128 verify-chain spd_ws): N<=4 -> 32 (few blocks,
    favor occupancy; N=4 spd 1.03->1.16), N>=8 -> 64 (enough blocks, favor reuse / amortize producer).
    Returns the first candidate that divides V."""
    order = (32, 64, 16, 8) if N <= 4 else (64, 32, 16, 8)
    for tv in order:
        if V % tv == 0:
            return tv
    return 8


def _select_ws_kvb_ilp_rows(tile_v):
    """Largest ilp_rows in {4,2,1} dividing rows_per_group (=tile_v/4). Larger ilp_rows interleaves
    more K-reduce shuffle chains -> stronger ILP."""
    rows_per_group = tile_v // 4
    for r in (4, 2, 1):
        if rows_per_group % r == 0:
            return r
    return 1


def kda_decode_mtp_ws_kvbuffer(
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
    tile_v: int = -1,
    ilp_rows: int = -1,
    use_smem_v: int = -1,
    opt_level: int = 3,
    fast_math: bool = True,
) -> torch.Tensor:
    """KDA MTP decode — ws-KVBuffer / warp-spec chunkwise (single chunk = T). VERIFY stage.

    Same role as kda_decode_mtp_kvbuffer (verify->flush, flush reuses kda_flush_kvbuffer), but a 4-warp
    warp-spec chunkwise impl for the large-batch regime (vs production ws). Signature mirrors vk-kvbuffer
    plus tile_v / ilp_rows / use_smem_v tuning knobs.
    """
    N, T, H, K = q.shape
    HV = v.shape[2]
    V = v.shape[3]
    write_ubuf = u_buffer is not None

    if scale is None:
        scale = K**-0.5
    else:
        assert scale > 0, f"scale must be positive, got {scale}"

    assert K == TILE_K, f"ws-kvbuffer requires K={TILE_K}, got {K}"
    assert K % VEC_SIZE == 0 and K // VEC_SIZE == 32, (
        f"ws-kvbuffer assumes K//vec_size==32 (one warp), got K={K}, vec_size={VEC_SIZE}"
    )

    if tile_v <= 0:
        tile_v = _select_ws_kvb_tile_v(V, N)
    assert V % tile_v == 0, f"ws-kvbuffer requires V % tile_v == 0, got V={V}, tile_v={tile_v}"
    assert tile_v % 4 == 0, f"ws-kvbuffer requires tile_v % 4 == 0 (4 warps), got {tile_v}"
    rows_per_group = tile_v // 4
    if ilp_rows <= 0:
        ilp_rows = _select_ws_kvb_ilp_rows(tile_v)
    assert rows_per_group % ilp_rows == 0, (
        f"ws-kvbuffer requires (tile_v/4) % ilp_rows == 0, got tile_v={tile_v}, ilp_rows={ilp_rows}"
    )
    use_smem_v_b = (N <= 16 and tile_v <= 128) if use_smem_v < 0 else bool(use_smem_v)

    h0_source, pool_size, _ = _normalize_state_source(
        initial_state_source, N=N, HV=HV, K=K, V=V, device=q.device, state_layout="vk",
    )

    a = _normalize_mtp_a(a, N=N, T=T, HV=HV, K=K)
    if b.dim() != 3 or tuple(b.shape) != (N, T, HV):
        raise ValueError(f"Unexpected b shape for MTP dense: {tuple(b.shape)}; expected {(N, T, HV)}")

    o = _prepare_output_tensor(q, out, (N, T, HV, V))

    q = q if q.is_contiguous() else q.contiguous()
    k = k if k.is_contiguous() else k.contiguous()
    v = v if v.is_contiguous() else v.contiguous()
    a = a if a.is_contiguous() else a.contiguous()
    b = b if b.is_contiguous() else b.contiguous()

    A_log = _normalize_A_log(A_log, HV)
    dt_bias = _normalize_dt_bias(dt_bias, HV, K)
    initial_state_indices = _normalize_state_indices(
        initial_state_indices, N=N, pool_size=pool_size, device=q.device
    )

    if write_ubuf:
        if tuple(u_buffer.shape) != (N, T, HV, V):
            raise ValueError(f"u_buffer shape must be {(N, T, HV, V)}, got {tuple(u_buffer.shape)}")
        if tuple(kinv_buffer.shape) != (N, T, HV, K) or tuple(b_buffer.shape) != (N, T, HV, K):
            raise ValueError(f"kinv_buffer/b_buffer shape must be {(N, T, HV, K)}")
        u_buf, kinv_buf, b_buf = u_buffer, kinv_buffer, b_buffer
    else:
        u_buf = torch.zeros(N, T, HV, V, dtype=torch.float32, device=q.device)
        kinv_buf = torch.zeros(N, T, HV, K, dtype=torch.float32, device=q.device)
        b_buf = torch.zeros(N, T, HV, K, dtype=torch.float32, device=q.device)

    stream = _get_cached_stream(q.device)

    h0_source_flat = h0_source.view(pool_size * HV, V, K)
    compiled_kernel = _get_compiled_mtp_ws_kvbuffer_kernel(
        N, T, H, HV, K, V, pool_size, tile_v, ilp_rows,
        scale=scale, use_qk_l2norm=use_qk_l2norm_in_kernel,
        disable_state_update=disable_state_update, emit_output=emit_output,
        write_ubuf=write_ubuf, use_smem_v=use_smem_v_b,
        softplus_beta=softplus_beta, softplus_threshold=softplus_threshold,
        opt_level=opt_level, fast_math=fast_math,
    )
    compiled_kernel(
        h0_source_flat, A_log, a, dt_bias, q, k, v, b, o,
        initial_state_indices, u_buf, kinv_buf, b_buf, stream,
    )
    return o


# flush kernel: read the compact u-buffer from verify, rank-m update over the first m accepted tokens:
#   S_m[v,k] = b_m[k] * (S0[v,k] + sum_{i<m} u_i[v] * kinv_i[k])
# Pure Phase-D (no gating/l2norm/reduce/solve). lane=K + vk, grid/layout match verify; m is constexpr.
@cute.kernel
def kda_flush_kvbuffer_vk_kernel(
    h0_source: cute.Tensor,  # [pool*HV, V, K] fp32
    u_buf: cute.Tensor,      # [N, T, HV, V] fp32
    kinv_buf: cute.Tensor,   # [N, T, HV, K] fp32
    b_buf: cute.Tensor,      # [N, T, HV, K] fp32
    h0_indices: cute.Tensor,
    vec_size: cutlass.Constexpr[int],
    num_v_tiles: cutlass.Constexpr[int],
    BV: cutlass.Constexpr[int],
    m: cutlass.Constexpr[int],  # accept length (first m tokens)
    HV: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
):
    tidx, _, _ = cute.arch.thread_idx()
    lane = tidx

    bidx, _, _ = cute.arch.block_idx()
    i_v = bidx % num_v_tiles
    tmp = bidx // num_v_tiles
    i_hv = tmp % HV
    i_n = tmp // HV

    cache_idx = h0_indices[i_n]
    if cache_idx >= 0:
        flat_state_idx = cache_idx * HV + i_hv

        r_h = cute.make_rmem_tensor(cute.make_layout((BV * vec_size,), stride=(1,)), cutlass.Float32)
        r_h4 = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
        r_bm = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
        r_kinv = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)

        # load S0 (this lane's vec_size K channels x BV v-cols)
        for vv in cutlass.range_constexpr(BV):
            v_global = i_v * BV + vv
            h_tile = cute.local_tile(h0_source, (1, 1, vec_size), (flat_state_idx, v_global, lane))
            cute.autovec_copy(h_tile, r_h4)
            for c in cutlass.range_constexpr(vec_size):
                r_h[vv * vec_size + c] = r_h4[c]

        # b_m: cumulative decay at token m-1 (this lane's channels)
        bm_tile = cute.local_tile(b_buf, (1, 1, 1, vec_size), (i_n, m - 1, i_hv, lane))
        cute.autovec_copy(bm_tile, r_bm)

        # accumulate sum_{i<m} u_i[v] * kinv_i[k]
        for i_i in cutlass.range_constexpr(m):
            kinv_tile = cute.local_tile(kinv_buf, (1, 1, 1, vec_size), (i_n, i_i, i_hv, lane))
            cute.autovec_copy(kinv_tile, r_kinv)
            for vv in cutlass.range_constexpr(BV):
                uval = cutlass.Float32(u_buf[i_n, i_i, i_hv, i_v * BV + vv])
                for c in cutlass.range_constexpr(vec_size):
                    r_h[vv * vec_size + c] += uval * r_kinv[c]

        # S_m = b_m * (S0 + sum ...), write back (contiguous float4)
        for vv in cutlass.range_constexpr(BV):
            v_global = i_v * BV + vv
            for c in cutlass.range_constexpr(vec_size):
                r_h4[c] = r_bm[c] * r_h[vv * vec_size + c]
            h_out = cute.local_tile(h0_source, (1, 1, vec_size), (flat_state_idx, v_global, lane))
            cute.autovec_copy(r_h4, h_out)


@cute.jit
def run_kda_flush_kvbuffer_vk_kernel(
    h0_source: cute.Tensor,
    u_buf: cute.Tensor,
    kinv_buf: cute.Tensor,
    b_buf: cute.Tensor,
    h0_indices: cute.Tensor,
    vec_size: cutlass.Constexpr[int],
    BV: cutlass.Constexpr[int],
    m: cutlass.Constexpr[int],
    HV: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    stream: cuda.CUstream,
):
    n_indices = h0_indices.layout.shape[0]
    num_v_tiles = cute.ceil_div(V, BV)
    grid_size = n_indices * HV * num_v_tiles
    kda_flush_kvbuffer_vk_kernel(
        h0_source, u_buf, kinv_buf, b_buf, h0_indices,
        vec_size, num_v_tiles, BV, m, HV, T, K, V,
    ).launch(grid=(grid_size, 1, 1), block=[32, 1, 1], smem=0, stream=stream)


_compiled_flush_kvbuffer_kernels: dict[tuple, object] = {}


def _get_compiled_flush_kvbuffer_kernel(N, T, HV, K, V, pool_size, BV, m, opt_level=3):
    key = (N, T, HV, K, V, pool_size, BV, m, opt_level)
    if key in _compiled_flush_kvbuffer_kernels:
        return _compiled_flush_kvbuffer_kernels[key]

    h0_source = torch.zeros(pool_size * HV, V, K, dtype=torch.float32, device="cuda")
    u_buf = torch.zeros(N, T, HV, V, dtype=torch.float32, device="cuda")
    kinv_buf = torch.zeros(N, T, HV, K, dtype=torch.float32, device="cuda")
    b_buf = torch.zeros(N, T, HV, K, dtype=torch.float32, device="cuda")
    h0_indices = torch.zeros(N, dtype=torch.int32, device="cuda")

    compiled = cute.compile(
        run_kda_flush_kvbuffer_vk_kernel,
        from_dlpack(h0_source, assumed_align=16),
        from_dlpack(u_buf, assumed_align=16),
        from_dlpack(kinv_buf, assumed_align=16),
        from_dlpack(b_buf, assumed_align=16),
        from_dlpack(h0_indices, assumed_align=16),
        vec_size=VEC_SIZE,
        BV=BV,
        m=m,
        HV=HV,
        T=T,
        K=K,
        V=V,
        stream=cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        options=f"--enable-tvm-ffi --opt-level {opt_level}",
    )
    _compiled_flush_kvbuffer_kernels[key] = compiled
    logger.info(
        f"CuTe DSL KDA flush KVBuffer kernel compiled: N={N}, T={T}, HV={HV}, "
        f"K={K}, V={V}, BV={BV}, m={m}"
    )
    return compiled


def kda_flush_kvbuffer(
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    u_buffer: torch.Tensor,
    kinv_buffer: torch.Tensor,
    b_buffer: torch.Tensor,
    accept_len: int,
    bv: int = -1,
    opt_level: int = 3,
) -> torch.Tensor:
    N, T, HV, V = u_buffer.shape
    K = kinv_buffer.shape[3]
    m = int(accept_len)
    assert 1 <= m <= T, f"accept_len must be in [1,{T}], got {m}"

    if bv <= 0:
        num_sms = torch.cuda.get_device_properties(initial_state_source.device).multi_processor_count
        bv = _select_vk_bv(N * HV, V, num_sms)
    assert bv in (8, 16, 32) and V % bv == 0, f"flush bv must be 8/16/32 and divide V, got bv={bv}, V={V}"

    h0_source, pool_size, _ = _normalize_state_source(
        initial_state_source, N=N, HV=HV, K=K, V=V, device=initial_state_source.device, state_layout="vk",
    )
    initial_state_indices = _normalize_state_indices(
        initial_state_indices, N=N, pool_size=pool_size, device=initial_state_source.device
    )
    stream = _get_cached_stream(initial_state_source.device)

    h0_source_flat = h0_source.view(pool_size * HV, V, K)
    compiled = _get_compiled_flush_kvbuffer_kernel(N, T, HV, K, V, pool_size, bv, m, opt_level=opt_level)
    compiled(h0_source_flat, u_buffer, kinv_buffer, b_buffer, initial_state_indices, stream)
    return initial_state_source
