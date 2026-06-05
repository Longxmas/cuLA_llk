"""CuTe DSL KDA MTP decode (KDA/topk=1/decode-only),两个 1-warp 算子:
(1) kda_decode_mtp_small_batch:lane=V + kv 布局(V 连续 coalesced)+ 线程内零-shuffle reduce + k_split 可调。
(2) kda_decode_mtp_small_batch_aligned:lane=K + vk 布局 + float4 连续块 load + 2-stage 软件流水 + BV 可调。
数学(decay-first)对齐 fp32 torch 参考口径(atol 3e-2 / rtol 2e-2)。
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

# 1 warp = 32 lane;decode V=128 → 每 program 默认 32 个 V 列(可由 bv/k_split 再切)。
WARP_BV = 32
# 每 lane 在 prep 负责的 K channel 数:K / warp_size = 128 / 32 = 4。
VEC_SIZE = 4

_compiled_mtp_small_batch_kernels: dict[tuple, object] = {}


@cute.kernel
def kda_mtp_small_batch_kernel(
    h0_source: cute.Tensor,  # [pool*HV, K, V] fp32 (kv, V-last)
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
    lane = tidx  # block 恒 32 = 1 warp

    bidx, _, _ = cute.arch.block_idx()
    i_v = bidx % num_v_tiles  # flat CTA → (i_n, i_hv, i_v V-block)
    tmp = bidx // num_v_tiles
    i_hv = tmp % HV
    i_n = tmp // HV
    i_h = i_hv // (HV // H)  # GVA: HV//H 个 v-head 共享 1 个 q/k head

    cache_idx = h0_indices[i_n]
    r_exp_A = cute.exp(cutlass.Float32(A_log[i_hv]), fastmath=fast_math)  # per-head, T 共用

    # SMEM 广播 q/k/g(K 维列间共享);XOR swizzle 让 k_split 段错开 bank(ks=1 恒等)。
    smem_k = K
    smem = cutlass.utils.SmemAllocator()
    sQ = smem.allocate_tensor(cutlass.Float32, cute.make_layout((smem_k,), stride=(1,)), 16)
    sK = smem.allocate_tensor(cutlass.Float32, cute.make_layout((smem_k,), stride=(1,)), 16)
    sG = smem.allocate_tensor(cutlass.Float32, cute.make_layout((smem_k,), stride=(1,)), 16)

    # k_split 个 lane 分摊一个 V 列的 K(各持 k_per_lane),reduce 后蝶形合并。
    k_per_lane = K // k_split
    v_local = lane % BV
    k_part = lane // BV
    k_off = k_part * k_per_lane

    r_h = cute.make_rmem_tensor(cute.make_layout((k_per_lane,), stride=(1,)), cutlass.Float32)
    r_q = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    r_k = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    r_q_bf16 = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16)
    r_k_bf16 = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16)

    v_global = i_v * BV + v_local  # 本 lane 服务的全局 V 列
    k_start = lane * vec_size  # prep:全 warp 32 lane × 4 = 全 128 K

    # ===== state 载入(kv:V 连续 → 固定 v_global、k 跨 lane 连续 = coalesced) =====
    if cache_idx >= 0:
        flat_state_idx = cache_idx * HV + i_hv
        for j in cutlass.range_constexpr(k_per_lane):
            r_h[j] = cutlass.Float32(h0_source[flat_state_idx, k_off + j, v_global])
    else:
        for j in cutlass.range_constexpr(k_per_lane):
            r_h[j] = cutlass.Float32(0.0)

    for i_t in cutlass.range_constexpr(T):
        # ===== prep:warp 协作算 q/k l2norm + per-channel g,写 SMEM 广播 =====
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
            # 全 warp reduce(32 lane × 4 = 全 128 K);本 kernel 仅有的 shuffle。
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
            sw = kk ^ (kk // k_per_lane)  # XOR swizzle SMEM 写位(a/dt_bias 用原 kk 读 GMEM)
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

        # beta = per-(i_n,i_t,i_hv) 标量,各 lane 各算一份(便宜,无需广播)。
        r_beta = cutlass.Float32(1.0) / (
            cutlass.Float32(1.0)
            + cute.exp(-cutlass.Float32(b[i_n, i_t, i_hv]), fastmath=fast_math)
        )

        cute.arch.barrier()  # 发布 prep 的 SMEM 写,recurrence 才能读

        # ===== recurrence:本 lane 算 k_per_lane 段 K 部分和,蝶形合并 k_split 个 lane(ks=1 时 0 步) =====
        r_v = cutlass.Float32(v[i_n, i_t, i_hv, v_global])
        # 融合 decay + s 部分和。
        s = cutlass.Float32(0.0)
        for j in cutlass.range_constexpr(k_per_lane):
            sw = j if k_split == 1 else (k_off + j) ^ k_part  # XOR swizzle 读位 = swz(k_off+j)
            r_h[j] = r_h[j] * sG[sw]
            s += r_h[j] * sK[sw]
        for st in cutlass.range_constexpr(k_split.bit_length() - 1):
            s += cute.arch.shuffle_sync_bfly(s, offset=BV << st, mask=-1, mask_and_clamp=31)
        v_new = (r_v - s) * r_beta
        # 融合 rank-1 + o 部分和,同样蝶形合并。
        o_val = cutlass.Float32(0.0)
        for j in cutlass.range_constexpr(k_per_lane):
            sw = j if k_split == 1 else (k_off + j) ^ k_part  # XOR swizzle 读位
            r_h[j] = r_h[j] + sK[sw] * v_new
            o_val += r_h[j] * sQ[sw]
        for st in cutlass.range_constexpr(k_split.bit_length() - 1):
            o_val += cute.arch.shuffle_sync_bfly(o_val, offset=BV << st, mask=-1, mask_and_clamp=31)
        o[(i_n, i_t, i_hv, v_global)] = cutlass.BFloat16(o_val)

        cute.arch.barrier()  # 确保各 lane 读完 sQ/sK/sG,下个 token prep 才能覆盖

    if cache_idx >= 0:
        if cutlass.const_expr(not disable_state_update):
            flat_state_idx = cache_idx * HV + i_hv
            for j in cutlass.range_constexpr(k_per_lane):  # kv:V 连续 → 回写同样 coalesced
                h0_source[(flat_state_idx, k_off + j, v_global)] = r_h[j]


@cute.jit
def run_kda_mtp_small_batch_kernel(
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
    """kv-layout host launcher:grid = N*HV*(V//BV),block = 32(1 warp)。
    BV = 32//k_split = 每 program 的 V 列数,k_split 个 lane 分摊一个 V 列的 K。"""
    n_indices = h0_indices.layout.shape[0]
    num_v_tiles = cute.ceil_div(V, BV)  # kv 下 h0.shape[1]=K,统一用 V 常量
    grid_size = n_indices * HV * num_v_tiles

    smem_bytes = 3 * K * 4 + 256  # sQ + sK + sG

    kda_mtp_small_batch_kernel(
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


def _get_compiled_mtp_small_batch_kernel(
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
    if key in _compiled_mtp_small_batch_kernels:
        return _compiled_mtp_small_batch_kernels[key]

    q = torch.zeros(N, T, H, K, dtype=torch.bfloat16, device="cuda")
    k = torch.zeros(N, T, H, K, dtype=torch.bfloat16, device="cuda")
    v = torch.zeros(N, T, HV, V, dtype=torch.bfloat16, device="cuda")
    a = torch.zeros(N, T, HV, K, dtype=torch.bfloat16, device="cuda")
    b = torch.zeros(N, T, HV, dtype=torch.bfloat16, device="cuda")
    o = torch.zeros(N, T, HV, V, dtype=torch.bfloat16, device="cuda")
    A_log = torch.zeros(HV, dtype=torch.float32, device="cuda")
    dt_bias = torch.zeros(HV, K, dtype=torch.float32, device="cuda")
    h0_source = torch.zeros(pool_size * HV, K, V, dtype=torch.float32, device="cuda")  # kv
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
        run_kda_mtp_small_batch_kernel,
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

    _compiled_mtp_small_batch_kernels[key] = compiled_kernel
    logger.info(
        "CuTe DSL KDA MTP small-batch kernel compiled: "
        f"N={N}, T={T}, H={H}, HV={HV}, K={K}, V={V}, pool_size={pool_size}, BV={BV}, "
        f"k_split={k_split}, opt_level={opt_level}, fast_math={fast_math}"
    )
    return compiled_kernel


# B200 ncu:k_split 对应 register-limited CTAs/SM(r_h=128//ks fp32 → reg 255/166/111)。
_KV_CTAS_PER_SM = {1: 8, 2: 12, 4: 16}


def _select_k_split(work_units, V, num_sms):
    """按 ks=1 的 wave 占用挑「够填就好」的最小 k_split(work_units = N*HV)。
    B200(HV=64,V=128)实测最优:N=1→4, N=2→2, N≥4→1(多切只为填 wave,代价是 shuffle 串行延迟)。"""
    waves1 = work_units * (V // 32) / (num_sms * _KV_CTAS_PER_SM[1])  # ks=1 的波数
    for ks, thresh in ((4, 0.3), (2, 0.6)):
        vcols = 32 // ks
        # 越欠载越该多切:waves1<0.3→ks4;<0.6→ks2;否则→ks1。
        if V % vcols == 0 and waves1 < thresh:
            return ks
    return 1


def kda_decode_mtp_small_batch(
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
    state_layout: str = "kv",
    disable_state_update: bool = False,
    bv: int = WARP_BV,
    k_split: int = 1,
    opt_level: int = 3,
    fast_math: bool = True,
) -> torch.Tensor:
    """lane=V + kv 布局(V 连续 → coalesced)+ 线程内零-shuffle reduce。
    仅 KDA/topk=1/decode;kv-only(state 全程 kv,不和 vk 的 ws 共享)。"""
    N, T, H, K = q.shape
    HV = v.shape[2]
    V = v.shape[3]

    if scale is None:
        scale = K**-0.5
    else:
        assert scale > 0, f"scale must be positive, got {scale}"

    assert K == TILE_K, f"KDA MTP (small_batch) requires K={TILE_K}, got {K}"
    assert K % VEC_SIZE == 0 and K // VEC_SIZE == 32, (
        f"small_batch 假定 K//vec_size==32(一个 warp),got K={K}, vec_size={VEC_SIZE}"
    )
    assert bv == WARP_BV, f"small_batch 固定 1 warp,bv 必须为 {WARP_BV},got {bv}"
    if k_split <= 0:  # auto:按 work_units(N*HV)的 wave 适配挑 k_split
        num_sms = torch.cuda.get_device_properties(q.device).multi_processor_count
        k_split = _select_k_split(N * HV, V, num_sms)
    assert k_split in (1, 2, 4), f"k_split 仅支持 1/2/4 或 <=0(auto),got {k_split}"
    assert bv % k_split == 0 and K % k_split == 0, (
        f"需 bv%k_split==0 且 K%k_split==0,got bv={bv}, K={K}, k_split={k_split}"
    )
    vcols = bv // k_split  # 每 program 的 V 列数(= kernel 内 BV)
    assert V % vcols == 0, f"small_batch requires V % (bv//k_split) == 0, got V={V}, vcols={vcols}"

    state_layout = _canonicalize_state_layout(state_layout)
    if state_layout != "kv":
        raise NotImplementedError(f"small_batch is kv-only; got state_layout={state_layout!r}")

    h0_source, pool_size, _ = _normalize_state_source(
        initial_state_source, N=N, HV=HV, K=K, V=V, device=q.device, state_layout=state_layout,
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

    h0_source_flat = h0_source.view(pool_size * HV, K, V)  # kv

    stream = _get_cached_stream(q.device)

    compiled_kernel = _get_compiled_mtp_small_batch_kernel(
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


# ============================================================================
# aligned:lane=K + warp-shuffle reduce —— lane t 沿 K 连续块映射(仅 vk、单 warp、不 split)。
# lane t 沿 K 连续块持 vec_size 个 K × 全 BV 个 V 列(float4 load,coalesced+向量化);
# reduce-over-K = 线程内 vec_size + 32-lane 5 步 butterfly;o 用 32 lane 同址幂等写落盘。
# 数学与 kv 布局变体完全一致(对齐 fp32 torch 参考)。
# ============================================================================
@cute.kernel
def kda_mtp_small_batch_aligned_kernel(
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

    # lane t 沿 K 连续块持 vec_size 个 K(K[4t:4t+4])× 全 BV 个 V 列;r_h[vv*vec_size+c]=state[i_v*BV+vv, vec_size*lane+c]。
    r_h = cute.make_rmem_tensor(cute.make_layout((BV * vec_size,), stride=(1,)), cutlass.Float32)
    r_q = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    r_k = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    r_g = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    r_bv = cute.make_rmem_tensor(cute.make_layout((BV,), stride=(1,)), cutlass.Float32)
    r_h4 = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)  # float4 临时缓冲(state load/store)
    # ===== 2-stage 软件流水双缓冲:算 token t 时预取 t+1 的 q/k/a/b 输入 =====
    r_qbf = [cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16) for _ in range(2)]
    r_kbf = [cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16) for _ in range(2)]
    r_abf = [cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32) for _ in range(2)]
    r_bbf = [cute.make_rmem_tensor(cute.make_layout((1,), stride=(1,)), cutlass.Float32) for _ in range(2)]
    r_dtb = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)  # dt_bias 与 token 无关,循环外载一次

    # ===== state 载入(连续块 + float4:lane t 取 K[4t:4t+4],跨 lane 拼成连续 512B = coalesced+向量化) =====
    if cache_idx >= 0:
        flat_state_idx = cache_idx * HV + i_hv
        for vv in cutlass.range_constexpr(BV):
            v_global = i_v * BV + vv
            # local_tile 第三坐标 lane、tile=vec_size → K 连续块 → autovec float4
            h_tile = cute.local_tile(h0_source, (1, 1, vec_size), (flat_state_idx, v_global, lane))
            cute.autovec_copy(h_tile, r_h4)
            for c in cutlass.range_constexpr(vec_size):
                r_h[vv * vec_size + c] = r_h4[c]
    else:
        for j in cutlass.range_constexpr(BV * vec_size):
            r_h[j] = cutlass.Float32(0.0)

    for c in cutlass.range_constexpr(vec_size):  # dt_bias 循环外载一次(连续块 K[4t:4t+4])
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
        # ===== 预取 t+1 的 q/k/a/b(LDG 提前发,延迟 overlap 进本 token l2norm/gate/recurrence)=====
        if cutlass.const_expr(i_t + 1 < T):
            nxt = (i_t + 1) % 2
            q_tn = cute.local_tile(q, (1, 1, 1, vec_size), (i_n, i_t + 1, i_h, lane))
            k_tn = cute.local_tile(k, (1, 1, 1, vec_size), (i_n, i_t + 1, i_h, lane))
            cute.autovec_copy(q_tn, r_qbf[nxt])
            cute.autovec_copy(k_tn, r_kbf[nxt])
            for c in cutlass.range_constexpr(vec_size):
                r_abf[nxt][c] = cutlass.Float32(a[i_n, i_t + 1, i_hv, vec_size * lane + c])
            r_bbf[nxt][0] = cutlass.Float32(b[i_n, i_t + 1, i_hv])

        # ===== prep:从 cur 缓冲读 q/k(已在寄存器,无 LDG 阻塞)+ l2norm + g =====
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
            x = r_abf[cur][c] + r_dtb[c]  # a 已预取,dt_bias 循环外载
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

        # v_t 迭代内载入(延迟 overlap 进上面 l2norm/gate;各 v 同址广播,便宜)
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
def run_kda_mtp_small_batch_aligned_kernel(
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

    kda_mtp_small_batch_aligned_kernel(
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
        run_kda_mtp_small_batch_aligned_kernel,
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
        "CuTe DSL KDA MTP small-batch ALIGNED(lane=K) kernel compiled: "
        f"N={N}, T={T}, H={H}, HV={HV}, K={K}, V={V}, pool_size={pool_size}, BV={BV}, "
        f"opt_level={opt_level}, fast_math={fast_math}"
    )
    return compiled_kernel


# B200 ncu:aligned BV=32 时 168 reg → Block Limit Registers ≈12。
def _select_aligned_bv(work_units, V, num_sms):
    """aligned split 轴是 V(BV)= 每 program 处理的 V 列数:降 BV → 寄存器↓ occupancy↑ grid↑,小批填 wave、藏 shuffle 延迟。
    B200 实测(N≤16,T≤8)BV=8 全程碾压 BV=32(0.92–1.15x vs 0.64–0.99x);N≥32 未测,保守回 BV=32。"""
    waves32 = work_units * (V // 32) / (num_sms * 12)  # BV=32 的波数(Block Limit Reg≈12)
    # waves32: N1=0.14 N2=0.28 N4=0.56 N8=1.12 N16=2.25 N32=4.5 → 阈值 3.0 覆盖 N≤16。
    if V % 8 == 0 and waves32 < 3.0:
        return 8
    return 32


def kda_decode_mtp_small_batch_aligned(
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
    bv: int = WARP_BV,
    opt_level: int = 3,
    fast_math: bool = True,
) -> torch.Tensor:
    """aligned:lane=K + warp-shuffle reduce + 连续块 float4 load + 2-stage 软件流水;仅 vk。
    ``bv`` = 每 program 的 V 列数,可调({8,16,32} 或 <=0 auto,见 ``_select_aligned_bv``)。"""
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
    if bv <= 0:  # auto:按 work_units(N*HV)的 wave 占用挑 BV(小批降 BV 填 grid 提 occupancy)
        num_sms = torch.cuda.get_device_properties(q.device).multi_processor_count
        bv = _select_aligned_bv(N * HV, V, num_sms)
    # block 恒 32(=K//vec_size=1 warp),与 BV 无关;BV 只是每 program 的 V 列数。
    assert bv in (8, 16, 32), f"aligned BV 仅支持 8/16/32 或 <=0(auto),got {bv}"
    assert V % bv == 0, f"aligned requires V % bv == 0, got V={V}, bv={bv}"

    state_layout = _canonicalize_state_layout(state_layout)
    if state_layout != "vk":
        raise NotImplementedError(
            f"kda_decode_mtp_small_batch_aligned only supports state_layout='vk'; got {state_layout!r}"
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
