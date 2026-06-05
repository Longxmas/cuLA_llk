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
    state_is_kv: cutlass.Constexpr[bool],
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
    # XOR swizzle:存储位置 swz(kk)=kk ^ (kk//k_per_lane) —— 把第 k_part 段按 k_part 异或重排
    # bank,让 k_split 个段错开到不同 bank(零 padding、双射无碰撞;k_split=1 时 swz 恒等)。
    smem_k = K
    smem = cutlass.utils.SmemAllocator()
    sQ = smem.allocate_tensor(cutlass.Float32, cute.make_layout((smem_k,), stride=(1,)), 16)
    sK = smem.allocate_tensor(cutlass.Float32, cute.make_layout((smem_k,), stride=(1,)), 16)
    sG = smem.allocate_tensor(cutlass.Float32, cute.make_layout((smem_k,), stride=(1,)), 16)
    # ks=1 的 coalesced state-load 转置缓冲:8 V列 × K = 4KB,4 组间复用(见下方 state 载入)。
    # 仅 vk+ks=1 分配(N≥4 的档);+4KB → 总 5.5KB SMEM,仍 8 blk/SM、不掉 occupancy。
    # kv 布局 V 连续、直读已 coalesced,无需 sH 转置。
    if cutlass.const_expr(k_split == 1 and not state_is_kv):
        sH = smem.allocate_tensor(cutlass.Float32, cute.make_layout((8 * smem_k,), stride=(1,)), 16)

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

    # ===== state 载入 =====
    if cache_idx >= 0:
        flat_state_idx = cache_idx * HV + i_hv
        if cutlass.const_expr(state_is_kv):
            # kv 布局 [.., K, V]:V 连续,lane=v_global 固定 k 跨 lane 地址连续 → 直接 coalesced,
            # 零 SMEM/barrier/divergence(ks>1 时按 k_part 分 ks 段,段内 BV 个 lane 仍连续)。
            for j in cutlass.range_constexpr(k_per_lane):
                r_h[j] = cutlass.Float32(h0_source[flat_state_idx, k_off + j, v_global])
        elif cutlass.const_expr(k_split == 1):
            # lane=V列 直读 = 512B-strided uncoalesced(实测 ~6us 拖死 N≥4)。改「分块 coalesced
            # 读 + SMEM 转置」:本 CTA 的 state 块 h0[flat, i_v*32:+32, 0:K] 是 32 V列×K 连续 4096
            # fp32。分 NGRP=4 组 × VPG=8 V列(sH 仅 4KB、组间复用)。每组:32 lane coalesced 读
            # VPG*K 连续元素(相邻 lane 相邻地址)写 sH(XOR swizzle k^v 避 bank 冲突),barrier,
            # 本组 8 个 lane 取自己 V 列的 128 K 进 r_h。
            VPG = 8
            NGRP = BV // VPG
            my_grp = lane // VPG
            my_vig = lane % VPG
            for g in cutlass.range_constexpr(NGRP):
                for s in cutlass.range_constexpr((VPG * K) // 32):
                    e = s * 32 + lane
                    vig = e // K
                    kk2 = e % K
                    gv = i_v * BV + g * VPG + vig  # 全局 V 列;GMEM 地址 = C + s*32 + lane → coalesced
                    sH[vig * K + (kk2 ^ vig)] = cutlass.Float32(h0_source[flat_state_idx, gv, kk2])
                cute.arch.barrier()
                if my_grp == g:
                    for j in cutlass.range_constexpr(K):
                        r_h[j] = sH[my_vig * K + (j ^ my_vig)]
                cute.arch.barrier()  # 复用 sH 前同步
        else:
            h_tile = cute.local_tile(h0_source, (1, 1, k_per_lane), (flat_state_idx, v_global, k_part))
            cute.autovec_copy(h_tile, r_h)
    else:
        for j in cutlass.range_constexpr(k_per_lane):
            r_h[j] = cutlass.Float32(0.0)

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
            sw = kk ^ (kk // k_per_lane)  # XOR swizzle SMEM 写位置(a/dt_bias 仍用原 kk 读 GMEM)
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
            sG[sw] = cute.exp(-r_exp_A * sp_x, fastmath=fast_math)
            sQ[sw] = r_q[i]
            sK[sw] = r_k[i]

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
            sw = j if k_split == 1 else (k_off + j) ^ k_part  # XOR swizzle 读位置(= swz(k_off+j))
            r_h[j] = r_h[j] * sG[sw]
            s += r_h[j] * sK[sw]
        for st in cutlass.range_constexpr(k_split.bit_length() - 1):
            s += cute.arch.shuffle_sync_bfly(s, offset=BV << st, mask=-1, mask_and_clamp=31)
        v_new = (r_v - s) * r_beta
        # 融合 rank-1 + o 部分和,同样蝶形合并。
        o_val = cutlass.Float32(0.0)
        for j in cutlass.range_constexpr(k_per_lane):
            sw = j if k_split == 1 else (k_off + j) ^ k_part  # XOR swizzle 读位置
            r_h[j] = r_h[j] + sK[sw] * v_new
            o_val += r_h[j] * sQ[sw]
        for st in cutlass.range_constexpr(k_split.bit_length() - 1):
            o_val += cute.arch.shuffle_sync_bfly(o_val, offset=BV << st, mask=-1, mask_and_clamp=31)
        o[(i_n, i_t, i_hv, v_global)] = cutlass.BFloat16(o_val)

        cute.arch.barrier()  # 确保各 lane 读完 sQ/sK/sG,下个 token 的 prep 才能覆盖

    if cache_idx >= 0:
        if cutlass.const_expr(not disable_state_update):
            flat_state_idx = cache_idx * HV + i_hv
            if cutlass.const_expr(state_is_kv):
                # kv:V 连续 → 回写同样 coalesced。
                for j in cutlass.range_constexpr(k_per_lane):
                    h0_source[(flat_state_idx, k_off + j, v_global)] = r_h[j]
            else:
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
    state_is_kv: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    """Host-side launcher: grid = N * HV * (V//BV), block = 32 (恒 1 warp)。
    BV = 32//k_split = 每 program 的 V 列数;k_split 个 lane 分摊一个 V 列的 K。"""
    n_indices = h0_indices.layout.shape[0]
    # kv 布局时 h0_source.shape[1]=K,不能用来推 V;统一用 V 常量算 num_v_tiles。
    num_v_tiles = cute.ceil_div(V, BV)
    grid_size = n_indices * HV * num_v_tiles

    # sQ + sK + sG (3*K) + vk&ks=1 的 coalesced-load 转置缓冲 sH (8*K) + 对齐余量。
    smem_bytes = 3 * K * 4 + (8 * K * 4 if (k_split == 1 and not state_is_kv) else 0) + 256

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
        state_is_kv,
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
    state_is_kv=False,
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
        state_is_kv,
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
    if state_is_kv:
        h0_source = torch.zeros(pool_size * HV, K, V, dtype=torch.float32, device="cuda")
    else:
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
        state_is_kv=state_is_kv,
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


# B200(GB200)ncu 实测:k_split 对应的 register-limited occupancy(CTAs/SM)。
# r_h 地板 = 128//k_split fp32 → reg 255/166/111 → Block Limit Registers 8/12/16。
_TSL_CTAS_PER_SM = {1: 8, 2: 12, 4: 16}


def _select_tsl_k_split(work_units, V, num_sms):
    """按 ks=1 的 wave 占用挑「够填就好」的最小 k_split(work_units = N*HV)。

    k_split 的唯一正收益 = 把 state 拆薄、提 occupancy 来填满空闲 SM;代价是每趟
    reduction 多 log2(k_split) 步 butterfly-shuffle(压在 recurrence 串行关键路径上)
    + grid×k_split 易跨过整数 wave 边界。所以只在 ks=1 时 GPU 明显欠载(grid 远不到
    一个 wave)才 split,且「够填一个 wave 就停」,不要过切。

    旧的 ceil(waves)/CTAs_per_SM cost 模型把「占用率高」当成线性加速,完全没算
    shuffle 关键路径惩罚 → 会给 N=2 误选 ks=4(实测 0.79–0.84x),其实 ks=2 才对
    (1.00–1.12x)。B200(GB200, HV=64, V=128)实测最优:N=1→4, N=2→2, N≥4→1。
    """
    waves1 = work_units * (V // 32) / (num_sms * _TSL_CTAS_PER_SM[1])  # ks=1 的波数
    for ks, thresh in ((4, 0.3), (2, 0.6)):
        vcols = 32 // ks
        # 越欠载越该多切:waves1<0.3→ks4;<0.6→ks2;否则(≥0.6,接近填满)→ ks1。
        if V % vcols == 0 and waves1 < thresh:
            return ks
    return 1


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
    if k_split <= 0:  # auto:按 work_units(N*HV) 的 wave 适配挑 k_split
        num_sms = torch.cuda.get_device_properties(q.device).multi_processor_count
        k_split = _select_tsl_k_split(N * HV, V, num_sms)
    assert k_split in (1, 2, 4), f"k_split 仅支持 1/2/4 或 <=0(auto),got {k_split}"
    assert bv % k_split == 0 and K % k_split == 0, (
        f"需 bv%k_split==0 且 K%k_split==0,got bv={bv}, K={K}, k_split={k_split}"
    )
    vcols = bv // k_split  # 每 program 的 V 列数(= kernel 内 BV);k_split 个 lane 分摊一个 V 列
    assert V % vcols == 0, f"triton-style requires V % (bv//k_split) == 0, got V={V}, vcols={vcols}"

    state_layout = _canonicalize_state_layout(state_layout)
    if state_layout not in ("vk", "kv"):
        raise NotImplementedError(
            f"kda_decode_mtp_triton_style supports state_layout in ('vk', 'kv'); got {state_layout!r}"
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
    # state_layout_is_kv=True: kv 布局让 lane=V列 的 state load/store 直接 coalesced(见 kernel)。

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

    if state_layout_is_kv:
        h0_source_flat = h0_source.view(pool_size * HV, K, V)
    else:
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
        state_is_kv=state_layout_is_kv,
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


# ============================================================================
# 变体 B:lane=K + warp-shuffle reduce —— 完全对齐 triton 的 thread→data 映射
# ----------------------------------------------------------------------------
# 上面的 lane=V 版(每 lane 1 个 V 列、独扛 128 K 进寄存器、线程内 128-FMA、零 shuffle)
# 在 vk 下 load 按 lane=V → 512B-strided uncoalesced。本变体改成 triton 的做法:
#   - lane t 沿 K 连续块 持 vec_size 个 K(k = vec_size*lane + c, = K[4t:4t+4])× 全
#     BV 个 V 列(r_h[BV*vec_size])。复刻 triton sizePerThread=[4,1]:每 lane 的 vec_size 个 K
#     连续 → local_tile + autovec_copy 发 float4(LDG.128),既 coalesced 又向量化。
#     (历史:曾用 interleaved k=c*32+lane → 虽 coalesced 但每 lane 4 个 K 跨 stride32 不连续
#      → 只能 scalar load、指令 4×,实测 0.71–0.92x 慢过 triton;见 FINDINGS §8.3。已改连续块。)
#   - reduce-over-K = 线程内 vec_size + 32-lane 5 步 butterfly shuffle(= triton
#     tl.sum(axis=0) 的展开,也是 ws 的结构);q/k/g 各 lane 只算自己 vec_size 个 K、留
#     寄存器,不再走 SMEM 广播。
#   - s[v]/o[v] all-reduce 后每 lane 都有全 BV 个;o 用「32 lane 同址同值幂等写」落盘。
# 数学与 lane=V 版完全一致(都已对齐 triton/torch);差别纯在 thread 映射,用于验证
# 「lane=V 才是 vk uncoalesced 的病根」。预期 ≈ triton-parity(它就是 triton 的 CuTe 复刻)。
# 仅 vk、单 warp(K//vec_size==32)、不 split。
# ============================================================================
@cute.kernel
def kda_mtp_triton_aligned_kernel(
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

    # lane t 沿 K 连续块 持 vec_size 个 K(k = vec_size*lane + c, = K[4t:4t+4])× 全 BV 个 V 列。
    # 复刻 triton 的 sizePerThread=[4,1](连续块),让 GMEM load 能 float4 向量化(autovec_copy)。
    # r_h[vv*vec_size + c] = state[i_v*BV+vv, vec_size*lane+c]。
    r_h = cute.make_rmem_tensor(cute.make_layout((BV * vec_size,), stride=(1,)), cutlass.Float32)
    r_q = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    r_k = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    r_g = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    r_bv = cute.make_rmem_tensor(cute.make_layout((BV,), stride=(1,)), cutlass.Float32)
    # float4 临时缓冲(state load/store)。
    r_h4 = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    # ===== 软件流水双缓冲(对齐 triton num_stages):算 token t 时预取 t+1 的 q/k/a/b 输入 =====
    # 只双缓冲前端关键路径输入(q/k/a/b,小);v 迭代内载入(延迟 overlap 进 gate)。+~18 reg(168→~186<255)。
    r_qbf = [cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16) for _ in range(2)]
    r_kbf = [cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16) for _ in range(2)]
    r_abf = [cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32) for _ in range(2)]
    r_bbf = [cute.make_rmem_tensor(cute.make_layout((1,), stride=(1,)), cutlass.Float32) for _ in range(2)]
    # dt_bias 与 token 无关,循环外载一次。
    r_dtb = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)

    # ===== state 载入(连续块 + float4:lane t 取 K[4t:4t+4],跨 lane 拼成连续 512B = coalesced+向量化) =====
    if cache_idx >= 0:
        flat_state_idx = cache_idx * HV + i_hv
        for vv in cutlass.range_constexpr(BV):
            v_global = i_v * BV + vv
            # local_tile 第三坐标 lane、tile=vec_size → K[lane*vec_size : +vec_size] 连续块 → autovec float4
            h_tile = cute.local_tile(h0_source, (1, 1, vec_size), (flat_state_idx, v_global, lane))
            cute.autovec_copy(h_tile, r_h4)
            for c in cutlass.range_constexpr(vec_size):
                r_h[vv * vec_size + c] = r_h4[c]
    else:
        for j in cutlass.range_constexpr(BV * vec_size):
            r_h[j] = cutlass.Float32(0.0)

    # dt_bias 与 token 无关,循环外载一次(连续块 K[4t:4t+4])。
    for c in cutlass.range_constexpr(vec_size):
        r_dtb[c] = cutlass.Float32(dt_bias[i_hv, vec_size * lane + c])

    # 预取 token 0 的 q/k/a/b 到 stage 0(流水填充)。
    q_t0 = cute.local_tile(q, (1, 1, 1, vec_size), (i_n, 0, i_h, lane))
    k_t0 = cute.local_tile(k, (1, 1, 1, vec_size), (i_n, 0, i_h, lane))
    cute.autovec_copy(q_t0, r_qbf[0])
    cute.autovec_copy(k_t0, r_kbf[0])
    for c in cutlass.range_constexpr(vec_size):
        r_abf[0][c] = cutlass.Float32(a[i_n, 0, i_hv, vec_size * lane + c])
    r_bbf[0][0] = cutlass.Float32(b[i_n, 0, i_hv])

    for i_t in cutlass.range_constexpr(T):
        cur = i_t % 2
        # ===== 预取 t+1 的 q/k/a/b(LDG 提前发,延迟 overlap 进本 token 的 l2norm/gate/recurrence)=====
        if cutlass.const_expr(i_t + 1 < T):
            nxt = (i_t + 1) % 2
            q_tn = cute.local_tile(q, (1, 1, 1, vec_size), (i_n, i_t + 1, i_h, lane))
            k_tn = cute.local_tile(k, (1, 1, 1, vec_size), (i_n, i_t + 1, i_h, lane))
            cute.autovec_copy(q_tn, r_qbf[nxt])
            cute.autovec_copy(k_tn, r_kbf[nxt])
            for c in cutlass.range_constexpr(vec_size):
                r_abf[nxt][c] = cutlass.Float32(a[i_n, i_t + 1, i_hv, vec_size * lane + c])
            r_bbf[nxt][0] = cutlass.Float32(b[i_n, i_t + 1, i_hv])

        # ===== prep:从 cur 缓冲读 q/k(已在寄存器,无 LDG 阻塞),l2norm + g =====
        for c in cutlass.range_constexpr(vec_size):
            r_q[c] = cutlass.Float32(r_qbf[cur][c])
            r_k[c] = cutlass.Float32(r_kbf[cur][c])

        if cutlass.const_expr(use_qk_l2norm):
            sum_q = cutlass.Float32(0.0)
            sum_k = cutlass.Float32(0.0)
            for c in cutlass.range_constexpr(vec_size):
                sum_q += r_q[c] * r_q[c]
                sum_k += r_k[c] * r_k[c]
            for off in [16, 8, 4, 2, 1]:
                sum_q += cute.arch.shuffle_sync_bfly(sum_q, offset=off, mask=-1, mask_and_clamp=31)
                sum_k += cute.arch.shuffle_sync_bfly(sum_k, offset=off, mask=-1, mask_and_clamp=31)
            inv_q = cute.rsqrt(sum_q + 1e-6, fastmath=fast_math) * scale
            inv_k = cute.rsqrt(sum_k + 1e-6, fastmath=fast_math)
            for c in cutlass.range_constexpr(vec_size):
                r_q[c] = r_q[c] * inv_q
                r_k[c] = r_k[c] * inv_k
        else:
            for c in cutlass.range_constexpr(vec_size):
                r_q[c] = r_q[c] * scale

        for c in cutlass.range_constexpr(vec_size):
            x = r_abf[cur][c] + r_dtb[c]  # a 已预取、dt_bias 循环外载
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
            r_g[c] = cute.exp(-r_exp_A * sp_x, fastmath=fast_math)

        r_beta = cutlass.Float32(1.0) / (
            cutlass.Float32(1.0)
            + cute.exp(-r_bbf[cur][0], fastmath=fast_math)
        )

        # v_t 迭代内载入(延迟 overlap 进上面的 l2norm/gate;各 v 同址广播,便宜)
        for vv in cutlass.range_constexpr(BV):
            r_bv[vv] = cutlass.Float32(v[i_n, i_t, i_hv, i_v * BV + vv])

        # ===== recurrence =====
        # decay:h *= exp(g)(per K)
        for vv in cutlass.range_constexpr(BV):
            for c in cutlass.range_constexpr(vec_size):
                r_h[vv * vec_size + c] = r_h[vv * vec_size + c] * r_g[c]

        # s[v]=Σ_k h·k_norm(线程内 vec_size + 32-lane butterfly)→ v_new=beta*(v_t-s)→ rank-1
        for vv in cutlass.range_constexpr(BV):
            s = cutlass.Float32(0.0)
            for c in cutlass.range_constexpr(vec_size):
                s += r_h[vv * vec_size + c] * r_k[c]
            for off in [16, 8, 4, 2, 1]:
                s += cute.arch.shuffle_sync_bfly(s, offset=off, mask=-1, mask_and_clamp=31)
            v_new = (r_bv[vv] - s) * r_beta
            for c in cutlass.range_constexpr(vec_size):
                r_h[vv * vec_size + c] = r_h[vv * vec_size + c] + r_k[c] * v_new

        # o[v]=Σ_k h·q_scaled(同 reduce)→ all-reduce 后每 lane 同值 → 32 lane 同址幂等写
        for vv in cutlass.range_constexpr(BV):
            ov = cutlass.Float32(0.0)
            for c in cutlass.range_constexpr(vec_size):
                ov += r_h[vv * vec_size + c] * r_q[c]
            for off in [16, 8, 4, 2, 1]:
                ov += cute.arch.shuffle_sync_bfly(ov, offset=off, mask=-1, mask_and_clamp=31)
            o[(i_n, i_t, i_hv, i_v * BV + vv)] = cutlass.BFloat16(ov)

    # ===== epilogue:回写 state(连续块 + float4,与载入对称) =====
    if cache_idx >= 0:
        if cutlass.const_expr(not disable_state_update):
            flat_state_idx = cache_idx * HV + i_hv
            for vv in cutlass.range_constexpr(BV):
                v_global = i_v * BV + vv
                for c in cutlass.range_constexpr(vec_size):
                    r_h4[c] = r_h[vv * vec_size + c]
                h_out = cute.local_tile(h0_source, (1, 1, vec_size), (flat_state_idx, v_global, lane))
                cute.autovec_copy(r_h4, h_out)


@cute.jit
def run_kda_mtp_triton_aligned_kernel(
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
    """lane=K aligned launcher:grid = N*HV*(V//BV),block = 32(1 warp)。无 SMEM。"""
    n_indices = h0_indices.layout.shape[0]
    num_v_tiles = cute.ceil_div(V, BV)
    grid_size = n_indices * HV * num_v_tiles

    kda_mtp_triton_aligned_kernel(
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
        smem=0,
        stream=stream,
    )


_compiled_mtp_aligned_kernels: dict[tuple, object] = {}


def _get_compiled_mtp_aligned_kernel(
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
        softplus_beta,
        softplus_threshold,
        opt_level,
        fast_math,
    )
    if key in _compiled_mtp_aligned_kernels:
        return _compiled_mtp_aligned_kernels[key]

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
        run_kda_mtp_triton_aligned_kernel,
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

    _compiled_mtp_aligned_kernels[key] = compiled_kernel
    logger.info(
        "CuTe DSL KDA MTP triton-ALIGNED(lane=K) kernel compiled: "
        f"N={N}, T={T}, H={H}, HV={HV}, K={K}, V={V}, pool_size={pool_size}, BV={BV}, "
        f"opt_level={opt_level}, fast_math={fast_math}"
    )
    return compiled_kernel


# B200(GB200)ncu:aligned BV=32 时 168 reg → Block Limit Registers ≈12。
def _select_aligned_bv(work_units, V, num_sms):
    """aligned 的「split」轴是 V(BV),不是 K(K 已 4/lane 摊满 32 lane)。BV = lane=K 下每
    program 处理的 V 列数,state = vec_size*BV fp32/lane。降 BV → 寄存器↓ occupancy↑ +
    grid = work_units*(V/BV)↑ → 小批(N=1/2,远不到 1 wave)把空闲 SM 填上、多驻 warp 去藏
    lane=K butterfly-shuffle 的串行延迟(那是小批病根,软件流水治 load 治不了它)。

    B200 实测(N≤8,bench --aligned-bv sweep):BV=8 全程最优或并列——N=1 直接 match
    lane=V 的 tsl(1.25–1.40x 超 triton)。**扩展 sweep(N≤16, T≤8)实证 BV=8 全程碾压
    BV=32**——连 N=16 都 0.92–1.15x vs BV=32 的 0.64–0.99x。BV=16 从不严格最优(N=4 T=4
    掉 0.85x),不入 auto(仍可手动 --aligned-bv 16)。N≥32 未测,保守回 BV=32。
    注:BV 最优可能随「向量化 reduce」优化(消掉逐 V 串行 shuffle 链)后改变,届时重 sweep。"""
    waves32 = work_units * (V // 32) / (num_sms * 12)  # BV=32 的波数(Block Limit Reg≈12)
    # waves32: N1=0.14 N2=0.28 N4=0.56 N8=1.12 N16=2.25 N32=4.5 → 阈值 3.0 覆盖 N≤16。
    if V % 8 == 0 and waves32 < 3.0:
        return 8
    return 32


def kda_decode_mtp_triton_aligned(
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
    opt_level: int = 3,
    fast_math: bool = True,
) -> torch.Tensor:
    """KDA MTP decode,lane=K + warp-shuffle reduce(完全对齐 triton 的 thread→data 映射)。

    仅 vk;lane=K + 连续块 float4 load + 2-stage 软件流水(预取 t+1 输入)。``bv`` = 每 program
    的 V 列数,可调({8,16,32} 或 <=0 auto):降 BV → 寄存器↓ occupancy↑ + grid↑,小批(N=1/2)
    填 wave、藏 lane=K shuffle 延迟。见 ``_select_aligned_bv``。
    """
    N, T, H, K = q.shape
    HV = v.shape[2]
    V = v.shape[3]

    if scale is None:
        scale = K**-0.5
    else:
        assert scale > 0, f"scale must be positive, got {scale}"

    assert K == TILE_K, f"KDA MTP (aligned) requires K={TILE_K}, got {K}"
    assert K % VEC_SIZE == 0 and K // VEC_SIZE == 32, (
        f"aligned 假定 K//vec_size==32(一个 warp),got K={K}, vec_size={VEC_SIZE}"
    )
    if bv <= 0:  # auto:按 work_units(N*HV) 的 wave 占用挑 BV(小批降 BV 填 grid 提 occupancy)
        num_sms = torch.cuda.get_device_properties(q.device).multi_processor_count
        bv = _select_aligned_bv(N * HV, V, num_sms)
    # block 恒 32(=K//vec_size=1 warp),与 BV 无关;BV 只是每 program 的 V 列数(state 大小)。
    assert bv in (8, 16, 32), f"aligned BV 仅支持 8/16/32 或 <=0(auto),got {bv}"
    assert V % bv == 0, f"aligned requires V % bv == 0, got V={V}, bv={bv}"

    state_layout = _canonicalize_state_layout(state_layout)
    if state_layout != "vk":
        raise NotImplementedError(
            f"kda_decode_mtp_triton_aligned only supports state_layout='vk'; got {state_layout!r}"
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
    assert not state_layout_is_kv

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

    compiled_kernel = _get_compiled_mtp_aligned_kernel(
        N,
        T,
        H,
        HV,
        K,
        V,
        pool_size,
        bv,
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
