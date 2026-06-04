"""CuTe DSL KDA MTP decode — Triton-layout 复刻 (1-warp, BK-in-register)。

练习/对照实现:把 sglang 的 ``fused_sigmoid_gating_delta_rule_update``
(``Issue 17/fused_sigmoid_gating_recurrent.py``) 的 **KDA + topk=1 单链** 路径
逐字翻译成 CuTe,**连 data layout 一起复刻**(不含 tree / varlen / GDN / lower_bound /
cache_intermediate)。目的是单独验证 Triton 那套 "1-warp 瘦 program + reduce-over-K
无 shuffle" 的 layout 在小 batch(尤其 N=4)能否补回 cuLA warp-spec 的 wave-量化缺口。

与 ``kda_decode_mtp_ws`` 的本质区别 = thread→data 映射:
  - ws  : CTA=128 线程/4 warp;K=128 分散到 32 lane(每 lane 4 channel),reduce-over-K
          走 5-step warp shuffle;V 行分给 4 warp。
  - 本文 : program=1 warp/32 线程(对齐 Triton num_warps=1),**每 lane 持 1 个 V 列的
          完整 K=128 in registers**(``r_h[128]``),reduce-over-K = 纯线程内 128-FMA
          (零 shuffle);K 维向量 q/k/g(列间共享)经 SMEM 广播。

grid = N * HV * num_v_tiles (num_v_tiles = V / BV, BV=32 = Triton 的 BV);block = 32。
lane ∈ [0,32) ↔ 该 program 负责的 BV 个 V 列里的第 lane 列(global V = i_v*BV + lane)。

Math per token (decay-first,与 ws/Triton 完全一致):
    g_t   = exp(-exp(A_log) * softplus(a_t + dt_bias))   # per-channel, K 维
    S    <- S * diag(g_t)
    s     = S @ k_norm                                    # reduce K (本文:线程内)
    v_new = sigmoid(b_t) * (v_t - s)
    S    += k_norm (x) v_new                              # rank-1
    o_t   = S @ (l2norm(q_t) * scale)                     # reduce K (本文:线程内)

bf16 累加阶不同于 ws(reduce 顺序不同),数值对齐 fp32 torch / Triton 口径
(atol 3e-2 / rtol 2e-2)。仅支持 ``state_layout='vk'``、K=V//? 约束见 entry。
"""

import logging

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

from cula.ops.kda_decode import (
    TILE_K,
    _canonicalize_state_layout,
    _get_cached_stream,
    _normalize_A_log,
    _normalize_dt_bias,
    _normalize_state_indices,
    _normalize_state_source,
    _prepare_output_tensor,
)
from cula.ops.kda_decode_mtp_ws import _normalize_mtp_a

logger = logging.getLogger(__name__)

# Triton 的 BV = min(next_pow2(V), 32);decode 形状 V=128 → BV=32 = 一个 warp 的 lane 数。
TRITON_BV = 32
# 每 lane 在 prep 阶段负责的 K channel 数:K / warp_size = 128 / 32 = 4。
VEC_SIZE = 4

_compiled_mtp_triton_kernels: dict[tuple, object] = {}


@cute.kernel
def kda_mtp_triton_style_kernel(
    h0_source: cute.Tensor,  # [pool*HV, V, K] fp32 (vk, K-last)
    A_log: cute.Tensor,  # [HV] fp32
    a: cute.Tensor,  # [N, T, HV, K]
    dt_bias: cute.Tensor,  # [HV, K]
    q: cute.Tensor,  # [N, T, H, K]
    k: cute.Tensor,  # [N, T, H, K]
    v: cute.Tensor,  # [N, T, HV, V]
    b: cute.Tensor,  # [N, T, HV]
    o: cute.Tensor,  # [N, T, HV, V]
    h0_indices: cute.Tensor,  # [N] int32
    vec_size: cutlass.Constexpr[int],
    num_v_tiles: cutlass.Constexpr[int],
    BV: cutlass.Constexpr[int],
    k_split: cutlass.Constexpr[int],
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
    fast_math: cutlass.Constexpr[bool],
):
    tidx, _, _ = cute.arch.thread_idx()
    lane = tidx  # block 恒 32 = 1 warp;tidx == lane ∈ [0,32)

    bidx, _, _ = cute.arch.block_idx()
    # 与 ws 同序解码 flat CTA index → (i_n, i_hv, i_v V-block)。
    i_v = bidx % num_v_tiles
    tmp = bidx // num_v_tiles
    i_hv = tmp % HV
    i_n = tmp // HV
    i_h = i_hv // (HV // H)  # GVA: HV//H 个 value-head 共享一个 q/k head

    cache_idx = h0_indices[i_n]

    # exp(A_log) per-head,T 个 token 共用,hoist 一次。
    r_exp_A = cute.exp(cutlass.Float32(A_log[i_hv]), fastmath=fast_math)

    # SMEM:列间共享的 K 维向量 (q_scaled / k_norm / g),由本 warp 协作算一次后广播。
    smem = cutlass.utils.SmemAllocator()
    sQ = smem.allocate_tensor(cutlass.Float32, cute.make_layout((K,), stride=(1,)), 16)
    sK = smem.allocate_tensor(cutlass.Float32, cute.make_layout((K,), stride=(1,)), 16)
    sG = smem.allocate_tensor(cutlass.Float32, cute.make_layout((K,), stride=(1,)), 16)

    # k_split:每个 V 列由 k_split 个 lane 分摊 K(各持 k_per_lane = K//k_split),reduce 后
    # 蝶形 shuffle 合并部分和。k_split=1 → 退化为原 tsl(lane=V 列,独扛 128 K,无 shuffle)。
    # BV = 32//k_split = 本 program 的 V 列数;block 恒 32 线程(1 warp)。
    k_per_lane = K // k_split    # 本 lane 常驻的 state 分量数(寄存器地板)
    v_local = lane % BV          # 本 lane 服务的 V 列(program 内)
    k_part = lane // BV          # 本 lane 管 K 的第几段(0..k_split-1)
    k_off = k_part * k_per_lane  # r_h[j] 对应全局 K[k_off + j]

    # 本 lane 只持有自己 V 列的 k_per_lane 个 K state 分量常驻寄存器。
    r_h = cute.make_rmem_tensor(cute.make_layout((k_per_lane,), stride=(1,)), cutlass.Float32)
    r_q = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    r_k = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    r_q_bf16 = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16)
    r_k_bf16 = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16)

    v_global = i_v * BV + v_local  # 本 lane 服务的全局 V 列
    k_start = lane * vec_size  # prep:全 warp 32 lane × vec_size=4 覆盖全 128 K(与 k_split 无关)

    # state 初始化 0;若该序列有 pool slot 则覆盖载入本 lane 的 k_per_lane 段(K[k_off:k_off+k_per_lane])。
    for j in cutlass.range_constexpr(k_per_lane):
        r_h[j] = cutlass.Float32(0.0)
    if cache_idx >= 0:
        flat_state_idx = cache_idx * HV + i_hv
        h_tile = cute.local_tile(h0_source, (1, 1, k_per_lane), (flat_state_idx, v_global, k_part))
        cute.autovec_copy(h_tile, r_h)

    for i_t in cutlass.range_constexpr(T):
        # ===== prep:warp 协作算 q/k l2norm + per-channel g,写入 SMEM 广播 =====
        q_tile = cute.local_tile(q, (1, 1, 1, vec_size), (i_n, i_t, i_h, lane))
        k_tile = cute.local_tile(k, (1, 1, 1, vec_size), (i_n, i_t, i_h, lane))
        cute.autovec_copy(q_tile, r_q_bf16)
        cute.autovec_copy(k_tile, r_k_bf16)
        for i in cutlass.range_constexpr(vec_size):
            r_q[i] = cutlass.Float32(r_q_bf16[i])
            r_k[i] = cutlass.Float32(r_k_bf16[i])

        if cutlass.const_expr(use_qk_l2norm):
            sum_q = cutlass.Float32(0.0)
            sum_k = cutlass.Float32(0.0)
            for i in cutlass.range_constexpr(vec_size):
                sum_q += r_q[i] * r_q[i]
                sum_k += r_k[i] * r_k[i]
            # 全 warp reduce (32 lane × vec_size=4 = 全部 128 K)。本 kernel 仅有的 shuffle。
            for offset in [16, 8, 4, 2, 1]:
                sum_q += cute.arch.shuffle_sync_bfly(sum_q, offset=offset, mask=-1, mask_and_clamp=31)
                sum_k += cute.arch.shuffle_sync_bfly(sum_k, offset=offset, mask=-1, mask_and_clamp=31)
            inv_q = cute.rsqrt(sum_q + 1e-6, fastmath=fast_math) * scale
            inv_k = cute.rsqrt(sum_k + 1e-6, fastmath=fast_math)
            for i in cutlass.range_constexpr(vec_size):
                r_q[i] = r_q[i] * inv_q
                r_k[i] = r_k[i] * inv_k
        else:
            for i in cutlass.range_constexpr(vec_size):
                r_q[i] = r_q[i] * scale

        for i in cutlass.range_constexpr(vec_size):
            kk = k_start + i
            x = cutlass.Float32(a[i_n, i_t, i_hv, kk]) + cutlass.Float32(dt_bias[i_hv, kk])
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
            sG[kk] = cute.exp(-r_exp_A * sp_x, fastmath=fast_math)
            sQ[kk] = r_q[i]
            sK[kk] = r_k[i]

        # beta 是 per-(i_n,i_t,i_hv) 标量,各 lane 各算一份(便宜,无需广播)。
        r_beta = cutlass.Float32(1.0) / (
            cutlass.Float32(1.0)
            + cute.exp(-cutlass.Float32(b[i_n, i_t, i_hv]), fastmath=fast_math)
        )

        cute.arch.barrier()  # 发布 prep 的 SMEM 写,recurrence 才能读

        # ===== recurrence:本 lane 算自己 k_per_lane 段 K 的部分和,再蝶形 shuffle 合并 k_split 个 lane =====
        # (k_split=1 → kk 是 constexpr、butterfly 0 步,与原 tsl 完全一致。)
        r_v = cutlass.Float32(v[i_n, i_t, i_hv, v_global])
        # 融合 decay + s 部分和。
        s = cutlass.Float32(0.0)
        for j in cutlass.range_constexpr(k_per_lane):
            kk = j if k_split == 1 else k_off + j
            r_h[j] = r_h[j] * sG[kk]
            s += r_h[j] * sK[kk]
        for st in cutlass.range_constexpr(k_split.bit_length() - 1):
            s += cute.arch.shuffle_sync_bfly(s, offset=BV << st, mask=-1, mask_and_clamp=31)
        v_new = (r_v - s) * r_beta
        # 融合 rank-1 + o 部分和,同样蝶形合并。
        o_val = cutlass.Float32(0.0)
        for j in cutlass.range_constexpr(k_per_lane):
            kk = j if k_split == 1 else k_off + j
            r_h[j] = r_h[j] + sK[kk] * v_new
            o_val += r_h[j] * sQ[kk]
        for st in cutlass.range_constexpr(k_split.bit_length() - 1):
            o_val += cute.arch.shuffle_sync_bfly(o_val, offset=BV << st, mask=-1, mask_and_clamp=31)
        o[(i_n, i_t, i_hv, v_global)] = cutlass.BFloat16(o_val)

        cute.arch.barrier()  # 确保各 lane 读完 sQ/sK/sG,下个 token 的 prep 才能覆盖

    if cache_idx >= 0:
        if cutlass.const_expr(not disable_state_update):
            flat_state_idx = cache_idx * HV + i_hv
            h_out = cute.local_tile(h0_source, (1, 1, k_per_lane), (flat_state_idx, v_global, k_part))
            cute.autovec_copy(r_h, h_out)


@cute.jit
def run_kda_mtp_triton_style_kernel(
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
    vec_size: cutlass.Constexpr[int],
    BV: cutlass.Constexpr[int],
    k_split: cutlass.Constexpr[int],
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
    fast_math: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    """Host-side launcher: grid = N * HV * (V//BV), block = 32 (恒 1 warp)。
    BV = 32//k_split = 每 program 的 V 列数;k_split 个 lane 分摊一个 V 列的 K。"""
    n_indices = h0_indices.layout.shape[0]
    v_dim = h0_source.layout.shape[1]
    num_v_tiles = cute.ceil_div(v_dim, BV)
    grid_size = n_indices * HV * num_v_tiles

    smem_bytes = 3 * K * 4 + 128  # sQ + sK + sG (fp32) + 对齐余量

    kda_mtp_triton_style_kernel(
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
        vec_size,
        num_v_tiles,
        BV,
        k_split,
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
        fast_math,
    ).launch(
        grid=(grid_size, 1, 1),
        block=[32, 1, 1],
        smem=smem_bytes,
        stream=stream,
    )


def _get_compiled_mtp_triton_kernel(
    N,
    T,
    H,
    HV,
    K,
    V,
    pool_size,
    BV,
    k_split,
    scale,
    use_qk_l2norm,
    disable_state_update,
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
        k_split,
        scale,
        use_qk_l2norm,
        disable_state_update,
        softplus_beta,
        softplus_threshold,
        opt_level,
        fast_math,
    )
    if key in _compiled_mtp_triton_kernels:
        return _compiled_mtp_triton_kernels[key]

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

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    compiled_kernel = cute.compile(
        run_kda_mtp_triton_style_kernel,
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
        vec_size=VEC_SIZE,
        BV=BV,
        k_split=k_split,
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
        fast_math=fast_math,
        stream=stream,
        options=f"--enable-tvm-ffi --opt-level {opt_level}",
    )

    _compiled_mtp_triton_kernels[key] = compiled_kernel
    logger.info(
        "CuTe DSL KDA MTP triton-style kernel compiled: "
        f"N={N}, T={T}, H={H}, HV={HV}, K={K}, V={V}, pool_size={pool_size}, BV={BV}, "
        f"k_split={k_split}, opt_level={opt_level}, fast_math={fast_math}"
    )
    return compiled_kernel


def kda_decode_mtp_triton_style(
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
    state_layout: str = "vk",
    disable_state_update: bool = False,
    bv: int = TRITON_BV,
    k_split: int = 1,
    opt_level: int = 3,
    fast_math: bool = True,
) -> torch.Tensor:
    """KDA MTP decode,Triton-layout 复刻(1-warp,reduce-over-K 无 shuffle)。

    仅 KDA + topk=1 单链;无 tree / varlen / GDN / lower_bound / cache_intermediate。
    签名与 ``kda_decode_mtp_ws`` 的核心子集对齐(去掉 tile_v/ilp/use_smem_v/
    use_packed_fma/precompute 等 ws 专属;新增 ``bv`` = Triton 的 BV)。
    """
    N, T, H, K = q.shape
    HV = v.shape[2]
    V = v.shape[3]

    if scale is None:
        scale = K**-0.5
    else:
        assert scale > 0, f"scale must be positive, got {scale}"

    assert K == TILE_K, f"KDA MTP (triton-style) requires K={TILE_K}, got {K}"
    assert K % VEC_SIZE == 0 and K // VEC_SIZE == 32, (
        f"triton-style 假定 K//vec_size==32(一个 warp),got K={K}, vec_size={VEC_SIZE}"
    )
    assert bv == TRITON_BV, f"triton-style 固定 1 warp,bv 必须为 {TRITON_BV},got {bv}"
    assert k_split in (1, 2, 4), f"k_split 仅支持 1/2/4,got {k_split}"
    assert bv % k_split == 0 and K % k_split == 0, (
        f"需 bv%k_split==0 且 K%k_split==0,got bv={bv}, K={K}, k_split={k_split}"
    )
    vcols = bv // k_split  # 每 program 的 V 列数(= kernel 内 BV);k_split 个 lane 分摊一个 V 列
    assert V % vcols == 0, f"triton-style requires V % (bv//k_split) == 0, got V={V}, vcols={vcols}"

    state_layout = _canonicalize_state_layout(state_layout)
    if state_layout != "vk":
        raise NotImplementedError(
            f"kda_decode_mtp_triton_style only supports state_layout='vk'; got {state_layout!r}"
        )

    h0_source, pool_size, state_layout_is_kv = _normalize_state_source(
        initial_state_source,
        N=N,
        HV=HV,
        K=K,
        V=V,
        device=q.device,
        state_layout=state_layout,
    )
    assert not state_layout_is_kv  # vk-only

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

    h0_source_flat = h0_source.view(pool_size * HV, V, K)

    stream = _get_cached_stream(q.device)

    compiled_kernel = _get_compiled_mtp_triton_kernel(
        N,
        T,
        H,
        HV,
        K,
        V,
        pool_size,
        vcols,
        k_split,
        scale=scale,
        use_qk_l2norm=use_qk_l2norm_in_kernel,
        disable_state_update=disable_state_update,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        opt_level=opt_level,
        fast_math=fast_math,
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
        stream,
    )

    return o
