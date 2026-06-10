"""CuTe DSL KDA MTP decode — KVBuffer / chunkwise parallel-verification variant.

Follow-up to issue 17. The production recurrent operators live in
``kda_decode_mtp.py`` (vk / kv). This file implements the **KVBuffer** paper's
*chunkwise parallel-verification* form (``KVBuffer: IO-aware Serving for Linear
Attention``, §3.3) as a NEW operator, to be benchmarked against the recurrent
ops. Design notes + math derivation: ``kvbuffer_dev/KVBUFFER_DESIGN.md`` (the
single-chunk gated-delta-rule equivalence is verified to machine precision in
``kvbuffer_dev/verify_chunkwise_kda.py``).

Instead of evolving the d×d state token-by-token (a length-T serial dependency
chain), this kernel treats the T draft tokens as ONE chunk: it computes per-token
outputs from the FIXED input state S0 plus a tiny T×T intra-chunk correction, and
updates the state once at the end. The expensive S0-matvecs are independent
across the T tokens (no serial chain), which is the latency angle vs the
recurrent op at small batch.

Layout / infra are mirrored from the production vk kernel
(``kda_mtp_small_batch_vk_kernel``) so the comparison is apples-to-apples:
grid = N*HV*(V//BV), 1 warp/CTA, lane=K (each lane owns vec_size=4 contiguous K
channels across BV V-cols), float4 coalesced state load/store, butterfly
shuffle reduce-over-K. Every reduce-over-K is a full-warp all-reduce, so A/P/
Skdec/Sqdec/u/o are computed identically (replicated) on all 32 lanes; only the
register state r_h and this lane's kdec/kinv/qdec channels are lane-private.

Chunkwise math per chunk (state S0[v,k], decay-first; matches the recurrent op):
    g_t[k]  = exp(-exp(A_log) * softplus(a_t[k] + dt_bias[k]))   # per channel
    b_t[k]  = prod_{i<=t} g_i[k]                                 # cumulative decay
    kdec_t  = k_norm_t * b_t ; kinv_t = k_norm_t / b_t ; qdec_t = q_scaled_t * b_t
    A[t,i]  = <kdec_t, kinv_i>  (i<t)      P[t,i] = <qdec_t, kinv_i>  (i<=t)
    u_t[v]  = beta_t * (v_t[v] - (S0 @ kdec_t)[v] - sum_{i<t} A[t,i] u_i[v])
    o_t[v]  = (S0 @ qdec_t)[v] + sum_{i<=t} P[t,i] u_i[v]
    S_T[v,k]= b_{T-1}[k] * (S0[v,k] + sum_i u_i[v] kinv_i[k])     # full accept

Scope (v1): single chunk (T tokens), vk layout, full accept. Partial-accept
rollback and a kv-layout variant are future work (see design doc).
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


# ============================================================================
# kvbuffer:lane=K + chunkwise(单 chunk = T 个 draft token)。布局/载入/回写与
# 生产 vk kernel 完全一致;递推体替换为 chunkwise gated-delta-rule:
#   - S0(r_h)全程不变,T 个 token 的输出都对 FIXED S0 做 matvec(无长度-T 串行链);
#   - intra-chunk 用 T×T 的 A/P 修正(内联重算,不占寄存器)+ 前代求 u;
#   - 末尾一次性把 state 更新为 S_T。
# 数学与 recurrent vk/kv 完全一致(verify_chunkwise_kda.py 机器精度对齐)。
# ============================================================================
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
    u_buf: cute.Tensor,     # [N, T, HV, V] fp32  伪 value u_t[v]（write_ubuf 时写）
    kinv_buf: cute.Tensor,  # [N, T, HV, K] fp32  kinv_t[k]=k_norm_t/b_t
    b_buf: cute.Tensor,     # [N, T, HV, K] fp32  累积衰减 b_t[k]
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
    write_ubuf: cutlass.Constexpr[bool],  # 写紧凑 u/kinv/b 到 GMEM(供 flush rank-m 重建任意 S_m)
    fast_math: cutlass.Constexpr[bool],
):
    tidx, _, _ = cute.arch.thread_idx()
    lane = tidx  # 1 warp = 32 lane

    bidx, _, _ = cute.arch.block_idx()
    i_v = bidx % num_v_tiles
    tmp = bidx // num_v_tiles
    i_hv = tmp % HV
    i_n = tmp // HV
    i_h = i_hv // (HV // H)

    cache_idx = h0_indices[i_n]
    r_exp_A = cute.exp(cutlass.Float32(A_log[i_hv]), fastmath=fast_math)

    # lane t 沿 K 连续块持 vec_size 个 K(K[4t:4t+4])× 全 BV 个 V 列;
    # r_h[vv*vec_size+c] = S0[i_v*BV+vv, vec_size*lane+c](全程不变,末尾才覆盖为 S_T)。
    r_h = cute.make_rmem_tensor(cute.make_layout((BV * vec_size,), stride=(1,)), cutlass.Float32)
    r_h4 = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)  # float4 临时缓冲
    r_dtb = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)  # dt_bias 循环外载一次

    # 本 lane 的 4 个 K 通道、所有 T 个 token 的 decayed 特征图(lane-private):
    #   kdec[t*vec+c]=k_norm_t*b_t ; kinv[t*vec+c]=k_norm_t/b_t ; qdec[t*vec+c]=q_scaled_t*b_t
    kdec = cute.make_rmem_tensor(cute.make_layout((T * vec_size,), stride=(1,)), cutlass.Float32)
    kinv = cute.make_rmem_tensor(cute.make_layout((T * vec_size,), stride=(1,)), cutlass.Float32)
    qdec = cute.make_rmem_tensor(cute.make_layout((T * vec_size,), stride=(1,)), cutlass.Float32)
    r_beta = cute.make_rmem_tensor(cute.make_layout((T,), stride=(1,)), cutlass.Float32)  # 每 token sigmoid 门(复制)
    r_u = cute.make_rmem_tensor(cute.make_layout((T * BV,), stride=(1,)), cutlass.Float32)  # 伪 value u_t[vv](复制)
    r_bv = cute.make_rmem_tensor(cute.make_layout((BV,), stride=(1,)), cutlass.Float32)  # 当前 token 的 v 列
    b_run = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)  # 累积衰减 b_t(本 lane 通道)
    r_qf = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    r_kf = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    r_qbf = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16)
    r_kbf = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16)
    a_row = cute.make_rmem_tensor(cute.make_layout((T,), stride=(1,)), cutlass.Float32)  # A[t,i]/P[t,i] 行临时(复制)

    # ===== state S0 载入(连续块 + float4,coalesced+向量化) =====
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

    for c in cutlass.range_constexpr(vec_size):  # dt_bias 循环外载一次(连续块 K[4t:4t+4])
        r_dtb[c] = cutlass.Float32(dt_bias[i_hv, vec_size * lane + c])

    # ===== Phase A:逐 token 算 g_t→累积 b_t、l2norm/scale、kdec/kinv/qdec、beta_t =====
    for c in cutlass.range_constexpr(vec_size):
        b_run[c] = cutlass.Float32(1.0)
    for i_t in cutlass.range_constexpr(T):
        q_tile = cute.local_tile(q, (1, 1, 1, vec_size), (i_n, i_t, i_h, lane))
        k_tile = cute.local_tile(k, (1, 1, 1, vec_size), (i_n, i_t, i_h, lane))
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

        # g_t[c](per K 通道)→ 累积 b_run[c] *= g_t
        for c in cutlass.range_constexpr(vec_size):
            x = cutlass.Float32(a[i_n, i_t, i_hv, vec_size * lane + c]) + r_dtb[c]
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

        # u-buffer:kinv_t / b_t 与 v-tile 无关 → 仅 i_v==0 的 CTA 写一次(float4 连续块)。
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
    # 循环后 b_run[c] = b_{T-1}[c]（末尾 state 更新用）。

    # ===== Phase B:前代求 u_t[vv]。u_t = beta_t*(v_t - S0@kdec_t - Σ_{i<t} A[t,i] u_i) =====
    for i_t in cutlass.range_constexpr(T):
        # A[t,i] = <kdec_t, kinv_i>（i<t；reduce-over-K 全 warp all-reduce，每 lane 同值）
        for i_i in cutlass.range_constexpr(i_t):
            aij = cutlass.Float32(0.0)
            for c in cutlass.range_constexpr(vec_size):
                aij += kdec[i_t * vec_size + c] * kinv[i_i * vec_size + c]
            for off in [16, 8, 4, 2, 1]:
                aij += cute.arch.shuffle_sync_bfly(aij, offset=off, mask=-1, mask_and_clamp=31)
            a_row[i_i] = aij
        # 当前 token 的 v 列
        for vv in cutlass.range_constexpr(BV):
            r_bv[vv] = cutlass.Float32(v[i_n, i_t, i_hv, i_v * BV + vv])
        # 每个 v 列:Skdec=S0@kdec_t（reduce-over-K）→ u
        for vv in cutlass.range_constexpr(BV):
            skd = cutlass.Float32(0.0)
            for c in cutlass.range_constexpr(vec_size):
                skd += r_h[vv * vec_size + c] * kdec[i_t * vec_size + c]
            for off in [16, 8, 4, 2, 1]:
                skd += cute.arch.shuffle_sync_bfly(skd, offset=off, mask=-1, mask_and_clamp=31)
            acc = r_bv[vv] - skd
            for i_i in cutlass.range_constexpr(i_t):
                acc -= a_row[i_i] * r_u[i_i * BV + vv]
            r_u[i_t * BV + vv] = r_beta[i_t] * acc

    # u-buffer:u_t[v] 是本 CTA 的 v 列(每 CTA 写自己那段);lane vv 写第 vv 列(连续 coalesced)。
    if cutlass.const_expr(write_ubuf):
        if lane < BV:
            for i_t in cutlass.range_constexpr(T):
                u_buf[i_n, i_t, i_hv, i_v * BV + lane] = r_u[i_t * BV + lane]

    # ===== Phase C:输出 o_t[vv] = S0@qdec_t + Σ_{i<=t} P[t,i] u_i =====
    # flush 模式(emit_output=False)只更新 state,跳过整段输出计算(P/o 都不需要)。
    if cutlass.const_expr(emit_output):
        for i_t in cutlass.range_constexpr(T):
            # P[t,i] = <qdec_t, kinv_i>（i<=t）
            for i_i in cutlass.range_constexpr(i_t + 1):
                pij = cutlass.Float32(0.0)
                for c in cutlass.range_constexpr(vec_size):
                    pij += qdec[i_t * vec_size + c] * kinv[i_i * vec_size + c]
                for off in [16, 8, 4, 2, 1]:
                    pij += cute.arch.shuffle_sync_bfly(pij, offset=off, mask=-1, mask_and_clamp=31)
                a_row[i_i] = pij
            for vv in cutlass.range_constexpr(BV):
                sqd = cutlass.Float32(0.0)
                for c in cutlass.range_constexpr(vec_size):
                    sqd += r_h[vv * vec_size + c] * qdec[i_t * vec_size + c]
                for off in [16, 8, 4, 2, 1]:
                    sqd += cute.arch.shuffle_sync_bfly(sqd, offset=off, mask=-1, mask_and_clamp=31)
                ov = sqd
                for i_i in cutlass.range_constexpr(i_t + 1):
                    ov += a_row[i_i] * r_u[i_i * BV + vv]
                # all-reduce 后每 lane 同值 → 32 lane 同址幂等写
                o[(i_n, i_t, i_hv, i_v * BV + vv)] = cutlass.BFloat16(ov)

    # ===== Phase D / epilogue:一次性更新 state 并回写 =====
    # S_T[v,k] = b_{T-1}[k] * (S0[v,k] + Σ_t u_t[v] * kinv_t[k])
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
    """lane=K kvbuffer launcher:grid = N*HV*(V//BV),block = 32(1 warp)。无 SMEM。"""
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
    """KDA MTP decode — KVBuffer / chunkwise parallel-verification(单 chunk = T)。**verify 阶段**。

    spec-decode 统一走 verify→flush 两 kernel(m 在 verify 后才知道,见
    SGLANG_VERIFY_ROLLBACK_FLOW.md):
    - **verify**(本函数,默认): emit_output=True, disable_state_update=True，**不提交 state**;
      传入 u_buffer/kinv_buffer/b_buffer 则把紧凑回滚数据 (u_t[v], kinv_t[k], b_t[k]) 写进
      GMEM(2Td+Td,比 recurrent 的 T·d² 中间态小 ~43×),供 ``kda_flush_kvbuffer`` 任意 m 重建。
    - **flush**: 见 ``kda_flush_kvbuffer`` —— 读 u_buffer 对前 m 个 token 做 rank-m 更新 → S_m。
    - full-accept 快路径(可选): disable_state_update=False 时本 kernel 也能直接提交 S_T(末尾
      Phase D),省掉 flush;但 m<T 必须走 flush。

    q/k [N,T,H,K], v/a [N,T,HV,V/K], b [N,T,HV]。state pool = vk [pool,HV,V,K]。
    u_buffer [N,T,HV,V] / kinv_buffer,b_buffer [N,T,HV,K] fp32(传 None = 不写)。
    数学与 recurrent vk/kv 完全一致(verify_chunkwise_kda.py 机器精度对齐)。
    """
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
        f"kvbuffer 假定 K//vec_size==32(一个 warp),got K={K}, vec_size={VEC_SIZE}"
    )

    if bv <= 0:  # auto:复用 vk 的 BV 启发式(小批降 BV 填 grid)
        num_sms = torch.cuda.get_device_properties(q.device).multi_processor_count
        bv = _select_vk_bv(N * HV, V, num_sms)
    assert bv in (8, 16, 32), f"kvbuffer bv 仅支持 8/16/32 或 <=0(auto),got {bv}"
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
            raise ValueError(f"u_buffer 形状须 {(N, T, HV, V)}, got {tuple(u_buffer.shape)}")
        if tuple(kinv_buffer.shape) != (N, T, HV, K) or tuple(b_buffer.shape) != (N, T, HV, K):
            raise ValueError(f"kinv_buffer/b_buffer 形状须 {(N, T, HV, K)}")
        u_buf, kinv_buf, b_buf = u_buffer, kinv_buffer, b_buffer
    else:  # 占位(kernel 内 write_ubuf=False 不触碰)
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


# ============================================================================
# flush kernel:读 verify 写下的紧凑 u-buffer,对前 m 个被接受 token 做 rank-m 更新:
#   S_m[v,k] = b_m[k] * (S0[v,k] + Σ_{i<m} u_i[v] * kinv_i[k])
# 纯 Phase-D(无 gating/l2norm/reduce/solve;不重算 verify 已算过的东西)。
# lane=K + vk,grid/布局与 verify 一致;m 为 constexpr(按接受长度缓存)。
# ============================================================================
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
    m: cutlass.Constexpr[int],  # 接受长度(前 m 个 token)
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

        # 载 S0(本 lane 的 4 个 K 通道 × BV 个 v 列)
        for vv in cutlass.range_constexpr(BV):
            v_global = i_v * BV + vv
            h_tile = cute.local_tile(h0_source, (1, 1, vec_size), (flat_state_idx, v_global, lane))
            cute.autovec_copy(h_tile, r_h4)
            for c in cutlass.range_constexpr(vec_size):
                r_h[vv * vec_size + c] = r_h4[c]

        # b_m(token m-1 的累积衰减,本 lane 通道)
        bm_tile = cute.local_tile(b_buf, (1, 1, 1, vec_size), (i_n, m - 1, i_hv, lane))
        cute.autovec_copy(bm_tile, r_bm)

        # 累加 Σ_{i<m} u_i[v] * kinv_i[k]
        for i_i in cutlass.range_constexpr(m):
            kinv_tile = cute.local_tile(kinv_buf, (1, 1, 1, vec_size), (i_n, i_i, i_hv, lane))
            cute.autovec_copy(kinv_tile, r_kinv)
            for vv in cutlass.range_constexpr(BV):
                uval = cutlass.Float32(u_buf[i_n, i_i, i_hv, i_v * BV + vv])
                for c in cutlass.range_constexpr(vec_size):
                    r_h[vv * vec_size + c] += uval * r_kinv[c]

        # S_m = b_m * (S0 + Σ ...) 并回写(float4 连续块)
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
    """KVBuffer **flush 阶段**:用 verify 写下的紧凑 u-buffer,对前 ``accept_len`` 个被接受
    token 做 rank-m 更新 → S_m,原位写进 state pool。不重算 gating/solve。

    u_buffer [N,T,HV,V], kinv_buffer/b_buffer [N,T,HV,K] fp32(verify 产出)。
    state pool = vk [pool,HV,V,K]。accept_len ∈ [1, T](链式;per-req 变长是后续工作)。
    """
    N, T, HV, V = u_buffer.shape
    K = kinv_buffer.shape[3]
    m = int(accept_len)
    assert 1 <= m <= T, f"accept_len 须 ∈ [1,{T}], got {m}"

    if bv <= 0:
        num_sms = torch.cuda.get_device_properties(initial_state_source.device).multi_processor_count
        bv = _select_vk_bv(N * HV, V, num_sms)
    assert bv in (8, 16, 32) and V % bv == 0, f"flush bv 须 8/16/32 且整除 V, got bv={bv}, V={V}"

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
