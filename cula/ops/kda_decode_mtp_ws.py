"""CuTe DSL KDA MTP decode — warp-specialized variant (issue #17, Route 1).

This is the "migrate flashinfer's warp-specialized GDN MTP kernel into cuLA,
changing only the decay" route (P3 plan Appendix A). It is an alternative
implementation of the same public contract as ``kda_decode_mtp`` (in
``kda_decode.py``'s sibling ``kda_decode_mtp.py``), kept as a SEPARATE module so
the two routes can be benchmarked head-to-head (``bench_kda_decode_mtp.py``)
before deciding which one goes upstream.

Source attribution: the kernel structure (warp specialization, register-resident
state, full-warp shuffle reduction, CTA decomposition ``B*HV*num_v_tiles``) is
ported from FlashInfer's ``flashinfer/gdn_kernels/gdn_decode_mtp.py``
(``gdn_verify_kernel_mtp``), Apache-2.0. The ONLY algorithmic change is the
decay: GDN uses a scalar gate ``g_t`` per (head, token); KDA uses a per-K-channel
gate ``g_t in R^K``. Concretely:
    - ``a``: [N, T, HV]      -> [N, T, HV, K]     (per channel)
    - ``dt_bias``: [HV]      -> [HV, K]           (per channel)
    - SMEM gate ``sG``: [T]  -> [T, K]            (per channel, staged like sK)
    - decay step ``r_h[row,i] *= r_g``  ->  ``*= r_g[i]``  (lane owns channels
      ``k_start .. k_start+vec_size``, so it scales each with the matching g)
    - ``beta`` stays a per-(head, token) scalar.
Everything else (steps 2-5 of the gated delta rule, the warp-spec Phase 1, the
state prefetch, L2 norm) is copied verbatim.

This kernel keeps FlashInfer's DECAY-FIRST order (decay the whole state, then dot
with the raw k), NOT cuLA's gk-premultiply order. That divergence is intentional:
Route 1 is a faithful port; cuLA's gk-premultiply lives in the Route-2 kernel
(``kda_decode_mtp.py``). The two will produce slightly different bf16 rounding;
both are validated against the fp32 torch oracle at atol 3e-2 / rtol 2e-2.

Scope (this file):
- Warp-specialized variant only (no inline variant — Route 1 drops it).
- ``ilp_rows in {2, 4}``. ilp=2 (Stage 1) covers every tile_v in {8,16,32,64};
  ilp=4 (Stage 2) fuses steps 1+2 and 4+5, uses double accumulators + packed
  F32x2 FMA on SM100 (scalar ``fma_pair`` fallback elsewhere), and requires
  ``tile_v % 16 == 0`` (so {16,32,64}). ilp=8 is not ported.
- ``vk`` state layout only (FlashInfer is vk-only; kv is a later add-back).
- ``use_smem_v`` (Stage C): preload the v-tile into SMEM + merged coalesced
  output writeback (``sOutput``); constexpr, off by default unless the heuristic
  (large batch / tile_v=64) or an explicit arg turns it on. Works with ilp 2/4.
- ``cache_intermediate_states`` (Stage D): when an ``intermediate_states_buffer``
  ([N, T, HV, V, K] vk) is passed, snapshot every token's post-state to GMEM
  (sequence-indexed) for speculative-decoding rollback. Produce-only: cuLA fills
  the buffer; it does NOT implement the rollback. Constexpr, off by default.
- ``disable_state_update`` supported (cheap; default False = always write back).

Math per token t (decay-first, per-channel g):
    g_t   = exp(-exp(A_log) * softplus(a_t + dt_bias))       # (K,) per-channel
    S    <- diag... S * g_t                                   # step 1 (per channel)
    s     = S @ k_norm                                        # step 2 (reduce K)
    v_new = sigmoid(b_t) * (v_t - s)                          # step 3
    S    += v_new (x) k_norm                                  # step 4 (rank-1, raw k)
    o_t   = S @ (l2norm(q_t) * scale)                         # step 5 (reduce K)
"""

import logging

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

from cula.ops.kda_decode import (
    NUM_THREADS,
    TILE_K,
    _canonicalize_state_layout,
    _get_cached_stream,
    _normalize_A_log,
    _normalize_dt_bias,
    _normalize_state_indices,
    _normalize_state_source,
    _prepare_output_tensor,
)
from cula.ops.kda_decode_mtp import _normalize_mtp_a, _select_mtp_config

logger = logging.getLogger(__name__)

# vec_size = 4 -> 32 threads/group = a full warp, 4 groups (warps) per block.
# Full-warp shuffle reduction over K is independent of tile_v, so unlike the
# Route-2 kernel we do NOT specialize the kernel body per tile_v.
VEC_SIZE_MTP = 4

# Warp-spec MTP kernels are compiled once per shape/config (including T, tile_v,
# ilp_rows, use_packed_fma) and cached.
_compiled_mtp_ws_kernels: dict[tuple, object] = {}

# Inline-variant MTP kernels (small batch). Same cache-key shape as the warp-spec
# table above; kept in a separate dict so the two variants never collide.
_compiled_mtp_inline_kernels: dict[tuple, object] = {}


@cute.jit
def fma_pair(a1, a2, b1, b2, c1, c2):
    """FMA two pairs: (a1*b1+c1, a2*b2+c2). SM90-compatible scalar fallback.

    ``cute.arch.fma_packed_f32x2`` emits an F32x2 instruction that only exists on
    SM100+ (Blackwell). The ilp=4 path pairs two FMAs per loop step to expose ILP;
    when ``use_packed_fma`` is False (SM90, or forced off) we issue the two scalar
    FMAs explicitly so the compiler still schedules them independently. Ported
    verbatim from FlashInfer ``gdn_decode_mtp.py:fma_pair``.
    """
    result1 = a1 * b1 + c1
    result2 = a2 * b2 + c2
    return result1, result2


@cute.kernel
def kda_verify_kernel_mtp_ws(
    h0_source: cute.Tensor,  # [pool_size * HV, V, K] fp32, K-last (VK layout)
    intermediate_states: cute.Tensor,  # [N*T*HV, V, K] fp32 snapshot cache (or dummy)
    vec_size: cutlass.Constexpr[int],
    num_v_tiles: cutlass.Constexpr[int],
    tile_v: cutlass.Constexpr[int],
    A_log: cute.Tensor,  # [HV] fp32
    a: cute.Tensor,  # [N, T, HV, K] (KDA: per-channel decay input)
    dt_bias: cute.Tensor,  # [HV, K] (KDA: per-channel decay bias)
    q: cute.Tensor,  # [N, T, H, K]
    k: cute.Tensor,  # [N, T, H, K]
    v: cute.Tensor,  # [N, T, HV, V]
    b: cute.Tensor,  # [N, T, HV]
    o: cute.Tensor,  # [N, T, HV, V] output
    h0_indices: cute.Tensor,  # [N] int32 (state-pool slot per sequence; <0 = pad)
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
    ilp_rows: cutlass.Constexpr[int],
    use_packed_fma: cutlass.Constexpr[bool],
    use_smem_v: cutlass.Constexpr[bool],
    cache_intermediate_states: cutlass.Constexpr[bool],
):
    """Warp-specialized KDA MTP kernel. One CTA owns one (i_n, i_hv, i_v) tile.

    Phase 1: warp 0 computes q/k (L2-normed) + per-channel g + scalar beta for
    all T tokens and writes them to SMEM; warps 1-3 prefetch the first ILP set
    of state rows from GMEM into registers. If ``use_smem_v`` (Stage C), all warps
    also cooperatively preload the v-tile into ``sVdata``. A barrier publishes the
    SMEM. Then all 4 warps run the T-step recurrence with state register-resident,
    one CTA covering ``tile_v`` V-rows (4 warps x rows_per_group), each lane owning
    ``vec_size`` K-channels of a V-row and reducing over K via full-warp shuffle.
    Outputs go straight to ``o`` (default) or, under ``use_smem_v``, accumulate in
    ``sOutput`` for a single coalesced merged writeback after the recurrence. If
    ``cache_intermediate_states`` (Stage D), each token's post-state is snapshotted
    fire-and-forget to ``intermediate_states`` (sequence-indexed) for spec-decode.
    """
    tidx, _, _ = cute.arch.thread_idx()
    lane_id = tidx % 32
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)

    # vec_size=4 -> threads_per_group=32 (full warp), 4 groups (one per warp).
    threads_per_group: cutlass.Constexpr[int] = K // vec_size  # 32
    num_groups: cutlass.Constexpr[int] = 4
    lane_in_group = lane_id % threads_per_group
    group_idx = warp_idx

    batch_idx, _, _ = cute.arch.block_idx()

    # Decode the flat CTA index into (i_n sequence, i_hv value-head, i_v V-tile).
    i_v = batch_idx % num_v_tiles
    tmp = batch_idx // num_v_tiles
    i_hv = tmp % HV
    i_n = tmp // HV
    i_h = i_hv // (HV // H)  # GVA: HV//H value-heads share one q/k head

    cache_idx = h0_indices[i_n]

    # A_log/dt_bias don't vary with token. exp(A_log) is per-head and (for KDA)
    # shared across all K channels, so hoist it out of the per-channel/token loop.
    r_A_log = cutlass.Float32(A_log[i_hv])
    r_exp_A = cute.exp(r_A_log)

    # SMEM broadcast buffers (warp 0 -> all warps). sG is [T, K] (per-channel),
    # unlike GDN's scalar [T]; staged exactly like sK. +8 K-padding keeps each
    # row 16B-aligned for vectorized access.
    smem = cutlass.utils.SmemAllocator()
    sQ = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout((T, K), stride=(K + 8, 1)), 16
    )
    sK = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout((T, K), stride=(K + 8, 1)), 16
    )
    sG = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout((T, K), stride=(K + 8, 1)), 16
    )
    sBeta = smem.allocate_tensor(cutlass.Float32, cute.make_layout((T,)), 16)

    # use_smem_v (Stage C): preload the CTA's v-tile into SMEM (one cooperative
    # load up front instead of a GMEM read every token) and accumulate this
    # CTA's outputs in SMEM for a single coalesced merged writeback at the end
    # (vs lane-0 scatter writes per token). Helps large batch / tile_v=64 write
    # bandwidth. Allocated LAST and only when enabled, so the offsets of the
    # unconditional broadcast buffers (sQ/sK/sG/sBeta) — and the off-path's total
    # SMEM footprint — never shift. 16B alignment like the other ws buffers (NOT
    # Route-2's 128B; see cdf6a89), so the launcher's flat +128 slack covers the
    # cumulative per-tensor padding without per-tensor 128B rounding.
    if cutlass.const_expr(use_smem_v):
        sVdata = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((T, tile_v), stride=(tile_v, 1)), 16
        )
        sOutput = smem.allocate_tensor(
            cutlass.BFloat16, cute.make_layout((T, tile_v), stride=(tile_v, 1)), 16
        )

    # Per-lane registers. r_g holds this lane's vec_size channels of g (the KDA
    # change vs GDN's scalar). r_h holds up to 8 V-rows of state (only ilp_rows
    # used); each r_h[row] spans the warp's 32 lanes to cover all K=128 channels.
    # Explicit-layout form (matches la_decode.py and FlashInfer); the row-major
    # (vec_size, 1) stride keeps each r_h[row] contiguous for autovec_copy/slice_.
    r_q = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    r_k = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    r_g = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    r_h = cute.make_rmem_tensor(
        cute.make_layout((8, vec_size), stride=(vec_size, 1)), cutlass.Float32
    )
    r_q_bf16 = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16)
    r_k_bf16 = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16)

    if cache_idx >= 0:
        k_start = lane_in_group * vec_size  # this lane's first K channel
        rows_per_group: cutlass.Constexpr[int] = tile_v // num_groups
        flat_state_idx = cache_idx * HV + i_hv  # row in [pool*HV, V, K]

        # ============ Phase 1: warp specialization ============
        if warp_idx == 0:
            # Warp 0 computes q/k/g/beta for all T tokens, broadcasts via SMEM.
            for i_t in cutlass.range_constexpr(T):
                q_tile = cute.local_tile(
                    q, (1, 1, 1, vec_size), (i_n, i_t, i_h, lane_in_group)
                )
                k_tile = cute.local_tile(
                    k, (1, 1, 1, vec_size), (i_n, i_t, i_h, lane_in_group)
                )
                cute.autovec_copy(q_tile, r_q_bf16)
                cute.autovec_copy(k_tile, r_k_bf16)
                for i in cutlass.range_constexpr(vec_size):
                    r_q[i] = cutlass.Float32(r_q_bf16[i])
                    r_k[i] = cutlass.Float32(r_k_bf16[i])

                if cutlass.const_expr(use_qk_l2norm):
                    sum_q = 0.0
                    sum_k = 0.0
                    for i in cutlass.range_constexpr(vec_size):
                        sum_q += r_q[i] * r_q[i]
                        sum_k += r_k[i] * r_k[i]
                    # Full-warp reduction (32 lanes x vec_size=4 = all 128 K).
                    for offset in [16, 8, 4, 2, 1]:
                        sum_q += cute.arch.shuffle_sync_bfly(
                            sum_q, offset=offset, mask=-1, mask_and_clamp=31
                        )
                        sum_k += cute.arch.shuffle_sync_bfly(
                            sum_k, offset=offset, mask=-1, mask_and_clamp=31
                        )
                    inv_norm_q_scaled = cute.rsqrt(sum_q + 1e-6) * scale
                    inv_norm_k = cute.rsqrt(sum_k + 1e-6)
                    for i in cutlass.range_constexpr(vec_size):
                        r_q[i] = r_q[i] * inv_norm_q_scaled
                        r_k[i] = r_k[i] * inv_norm_k
                else:
                    for i in cutlass.range_constexpr(vec_size):
                        r_q[i] = r_q[i] * scale

                # vec_size=4 -> warp 0's 32 lanes cover all 128 K channels.
                for i in cutlass.range_constexpr(vec_size):
                    sQ[(i_t, k_start + i)] = r_q[i]
                    sK[(i_t, k_start + i)] = r_k[i]

                # KDA per-channel decay gate: each lane computes g for its own
                # vec_size channels. g[kk] = exp(-exp(A_log) * softplus(a+dt_bias)).
                for i in cutlass.range_constexpr(vec_size):
                    kk = k_start + i
                    x = cutlass.Float32(a[i_n, i_t, i_hv, kk]) + cutlass.Float32(
                        dt_bias[i_hv, kk]
                    )
                    beta_x = softplus_beta * x
                    exp_beta_x = cute.exp(beta_x)
                    softplus_val = (cutlass.Float32(1.0) / softplus_beta) * cute.log(
                        cutlass.Float32(1.0) + exp_beta_x
                    )
                    use_softplus = (
                        cutlass.Float32(1.0)
                        if beta_x <= softplus_threshold
                        else cutlass.Float32(0.0)
                    )
                    softplus_x = (
                        use_softplus * softplus_val
                        + (cutlass.Float32(1.0) - use_softplus) * x
                    )
                    sG[(i_t, kk)] = cute.exp(-r_exp_A * softplus_x)

                # Update gate beta is a per-(head, token) scalar (warp-uniform).
                r_b = cutlass.Float32(b[i_n, i_t, i_hv])
                r_beta = cutlass.Float32(1.0) / (
                    cutlass.Float32(1.0) + cute.exp(-r_b)
                )
                sBeta[i_t] = r_beta

                # Cooperatively preload this CTA's v-tile into SMEM. Warp 0 covers
                # the first 32 tile-local columns (tidx<32); warps 1-3 cover the
                # rest below (their tidx 32..127). Guarded by tidx<tile_v so only
                # the tile_v owners write — together all columns 0..tile_v-1 are
                # filled exactly once (no overlap; tidx is the global thread id).
                # Mirrors FlashInfer gdn_decode_mtp.py:405-412.
                if cutlass.const_expr(use_smem_v):
                    if tidx < tile_v:
                        v_global_idx = i_v * tile_v + tidx
                        if v_global_idx < V:
                            sVdata[(i_t, tidx)] = cutlass.Float32(
                                v[i_n, i_t, i_hv, v_global_idx]
                            )
        else:
            # Warps 1-3: prefetch the first ILP set of state rows into registers,
            # overlapping the h-state DRAM latency with warp 0's Phase 1 compute.
            v_base_prefetch = i_v * tile_v + group_idx * rows_per_group
            if cutlass.const_expr(ilp_rows == 4):
                # Prefetch 4 h-state rows (4 independent load streams).
                v_pf_d = v_base_prefetch + 3
                if v_pf_d < V:
                    pf_a = cute.local_tile(
                        h0_source,
                        (1, 1, vec_size),
                        (flat_state_idx, v_base_prefetch, lane_in_group),
                    )
                    pf_b = cute.local_tile(
                        h0_source,
                        (1, 1, vec_size),
                        (flat_state_idx, v_base_prefetch + 1, lane_in_group),
                    )
                    pf_c = cute.local_tile(
                        h0_source,
                        (1, 1, vec_size),
                        (flat_state_idx, v_base_prefetch + 2, lane_in_group),
                    )
                    pf_d = cute.local_tile(
                        h0_source,
                        (1, 1, vec_size),
                        (flat_state_idx, v_base_prefetch + 3, lane_in_group),
                    )
                    cute.autovec_copy(pf_a, cute.slice_(r_h, (0, None)))
                    cute.autovec_copy(pf_b, cute.slice_(r_h, (1, None)))
                    cute.autovec_copy(pf_c, cute.slice_(r_h, (2, None)))
                    cute.autovec_copy(pf_d, cute.slice_(r_h, (3, None)))
            elif cutlass.const_expr(ilp_rows == 2):
                v_pf_b = v_base_prefetch + 1
                if v_pf_b < V:
                    pf_a = cute.local_tile(
                        h0_source,
                        (1, 1, vec_size),
                        (flat_state_idx, v_base_prefetch, lane_in_group),
                    )
                    pf_b = cute.local_tile(
                        h0_source,
                        (1, 1, vec_size),
                        (flat_state_idx, v_base_prefetch + 1, lane_in_group),
                    )
                    cute.autovec_copy(pf_a, cute.slice_(r_h, (0, None)))
                    cute.autovec_copy(pf_b, cute.slice_(r_h, (1, None)))

            # Warps 1-3 help preload the v-tile (their tidx 32..127 cover the
            # tile-local columns warp 0's 32 lanes can't reach, e.g. cols 32..63
            # at tile_v=64). Same tidx<tile_v guard as warp 0 -> every column
            # written exactly once. Mirrors FlashInfer gdn_decode_mtp.py:471-480.
            if cutlass.const_expr(use_smem_v):
                for i_t in cutlass.range_constexpr(T):
                    if tidx < tile_v:
                        v_global_idx = i_v * tile_v + tidx
                        if v_global_idx < V:
                            sVdata[(i_t, tidx)] = cutlass.Float32(
                                v[i_n, i_t, i_hv, v_global_idx]
                            )

        # Publish warp 0's SMEM writes (q/k/g/beta + preloaded v) to all warps
        # before the recurrence reads them.
        cute.arch.barrier()

        # ============ Recurrence: ilp_rows == 2 (process 2 V-rows together) ===
        if cutlass.const_expr(ilp_rows == 2):
            half_rows: cutlass.Constexpr[int] = rows_per_group // 2

            for row_pair in cutlass.range_constexpr(half_rows):
                v_idx_a = i_v * tile_v + group_idx * rows_per_group + row_pair * 2
                v_idx_b = v_idx_a + 1

                if v_idx_b < V:
                    # Load state for both rows. Warps 1-3 reuse the Phase-1
                    # prefetch on the first pair; everyone else loads in place.
                    if warp_idx == 0 or row_pair > 0:
                        h_tile_a = cute.local_tile(
                            h0_source,
                            (1, 1, vec_size),
                            (flat_state_idx, v_idx_a, lane_in_group),
                        )
                        h_tile_b = cute.local_tile(
                            h0_source,
                            (1, 1, vec_size),
                            (flat_state_idx, v_idx_b, lane_in_group),
                        )
                        cute.autovec_copy(h_tile_a, cute.slice_(r_h, (0, None)))
                        cute.autovec_copy(h_tile_b, cute.slice_(r_h, (1, None)))

                    for i_t in cutlass.range_constexpr(T):
                        # Read warp-0-staged q/k/g for this token (shared by both rows).
                        sQ_tile = cute.local_tile(sQ, (1, vec_size), (i_t, lane_in_group))
                        sK_tile = cute.local_tile(sK, (1, vec_size), (i_t, lane_in_group))
                        sG_tile = cute.local_tile(sG, (1, vec_size), (i_t, lane_in_group))
                        cute.autovec_copy(sQ_tile, r_q)
                        cute.autovec_copy(sK_tile, r_k)
                        cute.autovec_copy(sG_tile, r_g)
                        r_beta = sBeta[i_t]

                        # Step 1: per-channel decay (KDA: r_g[i], not a scalar).
                        for i in cutlass.range_constexpr(vec_size):
                            r_h[0, i] = r_h[0, i] * r_g[i]
                            r_h[1, i] = r_h[1, i] * r_g[i]

                        # Step 2: s = (decayed S) @ k_norm  (reduce over K).
                        sum_hk_a = 0.0
                        sum_hk_b = 0.0
                        for i in cutlass.range_constexpr(vec_size):
                            sum_hk_a += r_h[0, i] * r_k[i]
                            sum_hk_b += r_h[1, i] * r_k[i]
                        for offset in [16, 8, 4, 2, 1]:
                            sum_hk_a += cute.arch.shuffle_sync_bfly(
                                sum_hk_a, offset=offset, mask=-1, mask_and_clamp=31
                            )
                            sum_hk_b += cute.arch.shuffle_sync_bfly(
                                sum_hk_b, offset=offset, mask=-1, mask_and_clamp=31
                            )

                        # Step 3: delta rule. v from SMEM (preloaded) or GMEM.
                        if cutlass.const_expr(use_smem_v):
                            v_local_a = v_idx_a - i_v * tile_v
                            r_v_a = sVdata[(i_t, v_local_a)]
                            r_v_b = sVdata[(i_t, v_local_a + 1)]
                        else:
                            r_v_a = cutlass.Float32(v[i_n, i_t, i_hv, v_idx_a])
                            r_v_b = cutlass.Float32(v[i_n, i_t, i_hv, v_idx_b])
                        v_new_a = (r_v_a - sum_hk_a) * r_beta
                        v_new_b = (r_v_b - sum_hk_b) * r_beta

                        # Step 4: rank-1 update with raw k (decay already applied).
                        for i in cutlass.range_constexpr(vec_size):
                            r_h[0, i] += r_k[i] * v_new_a
                            r_h[1, i] += r_k[i] * v_new_b

                        # Stage D: snapshot the post-token state (r_h now holds the
                        # state AFTER consuming token i_t) to the GMEM cache. Indexed
                        # by SEQUENCE i_n (NOT the pool slot cache_idx used for the
                        # h0_source writeback): flat_idx = i_n*T*HV + i_t*HV + i_hv,
                        # i.e. intermediate_states[N,T,HV,V,K] flattened. Each lane
                        # writes its own vec_size K-channels of rows v_idx_a/b; rows
                        # and (i_n,i_t,i_hv,i_v) tiles are disjoint across the CTA,
                        # so the stores are race-free and fire-and-forget — placed
                        # before step 5 so they overlap the readout + reduction +
                        # output (mirrors FlashInfer gdn_decode_mtp.py:1265-1279).
                        if cutlass.const_expr(cache_intermediate_states):
                            flat_idx = i_n * T * HV + i_t * HV + i_hv
                            inter_a = cute.local_tile(
                                intermediate_states,
                                (1, 1, vec_size),
                                (flat_idx, v_idx_a, lane_in_group),
                            )
                            cute.autovec_copy(cute.slice_(r_h, (0, None)), inter_a)
                            inter_b = cute.local_tile(
                                intermediate_states,
                                (1, 1, vec_size),
                                (flat_idx, v_idx_b, lane_in_group),
                            )
                            cute.autovec_copy(cute.slice_(r_h, (1, None)), inter_b)

                        # Step 5: o = S_new @ q_scaled  (reduce over K).
                        sum_hq_a = 0.0
                        sum_hq_b = 0.0
                        for i in cutlass.range_constexpr(vec_size):
                            sum_hq_a += r_h[0, i] * r_q[i]
                            sum_hq_b += r_h[1, i] * r_q[i]
                        for offset in [16, 8, 4, 2, 1]:
                            sum_hq_a += cute.arch.shuffle_sync_bfly(
                                sum_hq_a, offset=offset, mask=-1, mask_and_clamp=31
                            )
                            sum_hq_b += cute.arch.shuffle_sync_bfly(
                                sum_hq_b, offset=offset, mask=-1, mask_and_clamp=31
                            )

                        # Reduction result is identical on all lanes -> lane 0
                        # writes. To SMEM (merged flush at kernel end) or GMEM.
                        if lane_in_group == 0:
                            if cutlass.const_expr(use_smem_v):
                                vla = v_idx_a - i_v * tile_v
                                sOutput[(i_t, vla)] = cutlass.BFloat16(sum_hq_a)
                                sOutput[(i_t, vla + 1)] = cutlass.BFloat16(sum_hq_b)
                            else:
                                o[(i_n, i_t, i_hv, v_idx_a)] = cutlass.BFloat16(sum_hq_a)
                                o[(i_n, i_t, i_hv, v_idx_b)] = cutlass.BFloat16(sum_hq_b)

                    # Write final state for both rows back to the pool (once).
                    if cutlass.const_expr(not disable_state_update):
                        h_tile_out_a = cute.local_tile(
                            h0_source,
                            (1, 1, vec_size),
                            (flat_state_idx, v_idx_a, lane_in_group),
                        )
                        cute.autovec_copy(cute.slice_(r_h, (0, None)), h_tile_out_a)
                        h_tile_out_b = cute.local_tile(
                            h0_source,
                            (1, 1, vec_size),
                            (flat_state_idx, v_idx_b, lane_in_group),
                        )
                        cute.autovec_copy(cute.slice_(r_h, (1, None)), h_tile_out_b)

        # ============ Recurrence: ilp_rows == 4 (process 4 V-rows together) ===
        # Mirrors FlashInfer's ilp_rows==4 path: steps 1+2 fused (decay then h@k)
        # and steps 4+5 fused (rank-1 update then h@q), DOUBLE accumulators to halve
        # the K-reduce FFMA dependency chain, and packed F32x2 FMA on SM100. KDA
        # change vs GDN: per-channel decay r_g[i]/r_g[i+1] (a vec_size register
        # loaded from sG, not a scalar). The h@k / h@q FMAs are byte-identical to
        # GDN. Stage-C use_smem_v (SMEM v read + sOutput merged writeback) and
        # Stage-D intermediate-state snapshots are wired in below under their
        # respective constexpr guards.
        elif cutlass.const_expr(ilp_rows == 4):
            quarter_rows: cutlass.Constexpr[int] = rows_per_group // 4

            for row_quad in cutlass.range_constexpr(quarter_rows):
                v_idx_a = i_v * tile_v + group_idx * rows_per_group + row_quad * 4
                v_idx_b = v_idx_a + 1
                v_idx_c = v_idx_a + 2
                v_idx_d = v_idx_a + 3

                if v_idx_d < V:
                    # Load state for 4 rows. Warps 1-3 reuse the Phase-1 prefetch on
                    # the first quad; everyone else loads in place.
                    if warp_idx == 0 or row_quad > 0:
                        h_tile_a = cute.local_tile(
                            h0_source,
                            (1, 1, vec_size),
                            (flat_state_idx, v_idx_a, lane_in_group),
                        )
                        h_tile_b = cute.local_tile(
                            h0_source,
                            (1, 1, vec_size),
                            (flat_state_idx, v_idx_b, lane_in_group),
                        )
                        h_tile_c = cute.local_tile(
                            h0_source,
                            (1, 1, vec_size),
                            (flat_state_idx, v_idx_c, lane_in_group),
                        )
                        h_tile_d = cute.local_tile(
                            h0_source,
                            (1, 1, vec_size),
                            (flat_state_idx, v_idx_d, lane_in_group),
                        )
                        cute.autovec_copy(h_tile_a, cute.slice_(r_h, (0, None)))
                        cute.autovec_copy(h_tile_b, cute.slice_(r_h, (1, None)))
                        cute.autovec_copy(h_tile_c, cute.slice_(r_h, (2, None)))
                        cute.autovec_copy(h_tile_d, cute.slice_(r_h, (3, None)))

                    for i_t in cutlass.range_constexpr(T):
                        # Warp-0-staged q/k/g for this token (shared by all 4 rows).
                        sQ_tile = cute.local_tile(sQ, (1, vec_size), (i_t, lane_in_group))
                        sK_tile = cute.local_tile(sK, (1, vec_size), (i_t, lane_in_group))
                        sG_tile = cute.local_tile(sG, (1, vec_size), (i_t, lane_in_group))
                        cute.autovec_copy(sQ_tile, r_q)
                        cute.autovec_copy(sK_tile, r_k)
                        cute.autovec_copy(sG_tile, r_g)
                        r_beta = sBeta[i_t]

                        # Steps 1+2 FUSED: per-channel decay (step 1) then h@k (step
                        # 2). Two accumulators (a/a2) split even/odd K channels so the
                        # FFMA chain is half-length; combined after the strided loop.
                        sum_hk_a = cutlass.Float32(0.0)
                        sum_hk_a2 = cutlass.Float32(0.0)
                        sum_hk_b = cutlass.Float32(0.0)
                        sum_hk_b2 = cutlass.Float32(0.0)
                        sum_hk_c = cutlass.Float32(0.0)
                        sum_hk_c2 = cutlass.Float32(0.0)
                        sum_hk_d = cutlass.Float32(0.0)
                        sum_hk_d2 = cutlass.Float32(0.0)
                        for i in cutlass.range_constexpr(0, vec_size, 2):
                            # Step 1: per-channel decay (KDA: r_g[i]/r_g[i+1]).
                            r_h[0, i] = r_h[0, i] * r_g[i]
                            r_h[0, i + 1] = r_h[0, i + 1] * r_g[i + 1]
                            r_h[1, i] = r_h[1, i] * r_g[i]
                            r_h[1, i + 1] = r_h[1, i + 1] * r_g[i + 1]
                            r_h[2, i] = r_h[2, i] * r_g[i]
                            r_h[2, i + 1] = r_h[2, i + 1] * r_g[i + 1]
                            r_h[3, i] = r_h[3, i] * r_g[i]
                            r_h[3, i + 1] = r_h[3, i + 1] * r_g[i + 1]
                            # Step 2: h@k, two channels per step (packed on SM100).
                            if cutlass.const_expr(use_packed_fma):
                                sum_hk_a, sum_hk_a2 = cute.arch.fma_packed_f32x2(
                                    src_a=(r_h[0, i], r_h[0, i + 1]),
                                    src_b=(r_k[i], r_k[i + 1]),
                                    src_c=(sum_hk_a, sum_hk_a2),
                                )
                                sum_hk_b, sum_hk_b2 = cute.arch.fma_packed_f32x2(
                                    src_a=(r_h[1, i], r_h[1, i + 1]),
                                    src_b=(r_k[i], r_k[i + 1]),
                                    src_c=(sum_hk_b, sum_hk_b2),
                                )
                                sum_hk_c, sum_hk_c2 = cute.arch.fma_packed_f32x2(
                                    src_a=(r_h[2, i], r_h[2, i + 1]),
                                    src_b=(r_k[i], r_k[i + 1]),
                                    src_c=(sum_hk_c, sum_hk_c2),
                                )
                                sum_hk_d, sum_hk_d2 = cute.arch.fma_packed_f32x2(
                                    src_a=(r_h[3, i], r_h[3, i + 1]),
                                    src_b=(r_k[i], r_k[i + 1]),
                                    src_c=(sum_hk_d, sum_hk_d2),
                                )
                            else:
                                sum_hk_a, sum_hk_a2 = fma_pair(
                                    r_h[0, i], r_h[0, i + 1], r_k[i], r_k[i + 1], sum_hk_a, sum_hk_a2
                                )
                                sum_hk_b, sum_hk_b2 = fma_pair(
                                    r_h[1, i], r_h[1, i + 1], r_k[i], r_k[i + 1], sum_hk_b, sum_hk_b2
                                )
                                sum_hk_c, sum_hk_c2 = fma_pair(
                                    r_h[2, i], r_h[2, i + 1], r_k[i], r_k[i + 1], sum_hk_c, sum_hk_c2
                                )
                                sum_hk_d, sum_hk_d2 = fma_pair(
                                    r_h[3, i], r_h[3, i + 1], r_k[i], r_k[i + 1], sum_hk_d, sum_hk_d2
                                )
                        sum_hk_a = sum_hk_a + sum_hk_a2
                        sum_hk_b = sum_hk_b + sum_hk_b2
                        sum_hk_c = sum_hk_c + sum_hk_c2
                        sum_hk_d = sum_hk_d + sum_hk_d2

                        # Full-warp reduction for all 4 h@k dot products.
                        for offset in [16, 8, 4, 2, 1]:
                            sum_hk_a += cute.arch.shuffle_sync_bfly(
                                sum_hk_a, offset=offset, mask=-1, mask_and_clamp=31
                            )
                            sum_hk_b += cute.arch.shuffle_sync_bfly(
                                sum_hk_b, offset=offset, mask=-1, mask_and_clamp=31
                            )
                            sum_hk_c += cute.arch.shuffle_sync_bfly(
                                sum_hk_c, offset=offset, mask=-1, mask_and_clamp=31
                            )
                            sum_hk_d += cute.arch.shuffle_sync_bfly(
                                sum_hk_d, offset=offset, mask=-1, mask_and_clamp=31
                            )

                        # Step 3: delta rule for all 4 rows. v from SMEM or GMEM.
                        if cutlass.const_expr(use_smem_v):
                            v_local_a = v_idx_a - i_v * tile_v
                            r_v_a = sVdata[(i_t, v_local_a)]
                            r_v_b = sVdata[(i_t, v_local_a + 1)]
                            r_v_c = sVdata[(i_t, v_local_a + 2)]
                            r_v_d = sVdata[(i_t, v_local_a + 3)]
                        else:
                            r_v_a = cutlass.Float32(v[i_n, i_t, i_hv, v_idx_a])
                            r_v_b = cutlass.Float32(v[i_n, i_t, i_hv, v_idx_b])
                            r_v_c = cutlass.Float32(v[i_n, i_t, i_hv, v_idx_c])
                            r_v_d = cutlass.Float32(v[i_n, i_t, i_hv, v_idx_d])
                        v_new_a = (r_v_a - sum_hk_a) * r_beta
                        v_new_b = (r_v_b - sum_hk_b) * r_beta
                        v_new_c = (r_v_c - sum_hk_c) * r_beta
                        v_new_d = (r_v_d - sum_hk_d) * r_beta

                        # Steps 4+5 FUSED: rank-1 update with raw k (step 4) then
                        # h@q (step 5), per row. Double accumulators again.
                        sum_hq_a = cutlass.Float32(0.0)
                        sum_hq_a2 = cutlass.Float32(0.0)
                        sum_hq_b = cutlass.Float32(0.0)
                        sum_hq_b2 = cutlass.Float32(0.0)
                        sum_hq_c = cutlass.Float32(0.0)
                        sum_hq_c2 = cutlass.Float32(0.0)
                        sum_hq_d = cutlass.Float32(0.0)
                        sum_hq_d2 = cutlass.Float32(0.0)
                        for i in cutlass.range_constexpr(0, vec_size, 2):
                            if cutlass.const_expr(use_packed_fma):
                                r_h[0, i], r_h[0, i + 1] = cute.arch.fma_packed_f32x2(
                                    src_a=(r_k[i], r_k[i + 1]),
                                    src_b=(v_new_a, v_new_a),
                                    src_c=(r_h[0, i], r_h[0, i + 1]),
                                )
                                r_h[1, i], r_h[1, i + 1] = cute.arch.fma_packed_f32x2(
                                    src_a=(r_k[i], r_k[i + 1]),
                                    src_b=(v_new_b, v_new_b),
                                    src_c=(r_h[1, i], r_h[1, i + 1]),
                                )
                                r_h[2, i], r_h[2, i + 1] = cute.arch.fma_packed_f32x2(
                                    src_a=(r_k[i], r_k[i + 1]),
                                    src_b=(v_new_c, v_new_c),
                                    src_c=(r_h[2, i], r_h[2, i + 1]),
                                )
                                r_h[3, i], r_h[3, i + 1] = cute.arch.fma_packed_f32x2(
                                    src_a=(r_k[i], r_k[i + 1]),
                                    src_b=(v_new_d, v_new_d),
                                    src_c=(r_h[3, i], r_h[3, i + 1]),
                                )
                                sum_hq_a, sum_hq_a2 = cute.arch.fma_packed_f32x2(
                                    src_a=(r_h[0, i], r_h[0, i + 1]),
                                    src_b=(r_q[i], r_q[i + 1]),
                                    src_c=(sum_hq_a, sum_hq_a2),
                                )
                                sum_hq_b, sum_hq_b2 = cute.arch.fma_packed_f32x2(
                                    src_a=(r_h[1, i], r_h[1, i + 1]),
                                    src_b=(r_q[i], r_q[i + 1]),
                                    src_c=(sum_hq_b, sum_hq_b2),
                                )
                                sum_hq_c, sum_hq_c2 = cute.arch.fma_packed_f32x2(
                                    src_a=(r_h[2, i], r_h[2, i + 1]),
                                    src_b=(r_q[i], r_q[i + 1]),
                                    src_c=(sum_hq_c, sum_hq_c2),
                                )
                                sum_hq_d, sum_hq_d2 = cute.arch.fma_packed_f32x2(
                                    src_a=(r_h[3, i], r_h[3, i + 1]),
                                    src_b=(r_q[i], r_q[i + 1]),
                                    src_c=(sum_hq_d, sum_hq_d2),
                                )
                            else:
                                r_h[0, i], r_h[0, i + 1] = fma_pair(
                                    r_k[i], r_k[i + 1], v_new_a, v_new_a, r_h[0, i], r_h[0, i + 1]
                                )
                                r_h[1, i], r_h[1, i + 1] = fma_pair(
                                    r_k[i], r_k[i + 1], v_new_b, v_new_b, r_h[1, i], r_h[1, i + 1]
                                )
                                r_h[2, i], r_h[2, i + 1] = fma_pair(
                                    r_k[i], r_k[i + 1], v_new_c, v_new_c, r_h[2, i], r_h[2, i + 1]
                                )
                                r_h[3, i], r_h[3, i + 1] = fma_pair(
                                    r_k[i], r_k[i + 1], v_new_d, v_new_d, r_h[3, i], r_h[3, i + 1]
                                )
                                sum_hq_a, sum_hq_a2 = fma_pair(
                                    r_h[0, i], r_h[0, i + 1], r_q[i], r_q[i + 1], sum_hq_a, sum_hq_a2
                                )
                                sum_hq_b, sum_hq_b2 = fma_pair(
                                    r_h[1, i], r_h[1, i + 1], r_q[i], r_q[i + 1], sum_hq_b, sum_hq_b2
                                )
                                sum_hq_c, sum_hq_c2 = fma_pair(
                                    r_h[2, i], r_h[2, i + 1], r_q[i], r_q[i + 1], sum_hq_c, sum_hq_c2
                                )
                                sum_hq_d, sum_hq_d2 = fma_pair(
                                    r_h[3, i], r_h[3, i + 1], r_q[i], r_q[i + 1], sum_hq_d, sum_hq_d2
                                )
                        sum_hq_a = sum_hq_a + sum_hq_a2
                        sum_hq_b = sum_hq_b + sum_hq_b2
                        sum_hq_c = sum_hq_c + sum_hq_c2
                        sum_hq_d = sum_hq_d + sum_hq_d2

                        # Full-warp reduction for all 4 h@q dot products.
                        for offset in [16, 8, 4, 2, 1]:
                            sum_hq_a += cute.arch.shuffle_sync_bfly(
                                sum_hq_a, offset=offset, mask=-1, mask_and_clamp=31
                            )
                            sum_hq_b += cute.arch.shuffle_sync_bfly(
                                sum_hq_b, offset=offset, mask=-1, mask_and_clamp=31
                            )
                            sum_hq_c += cute.arch.shuffle_sync_bfly(
                                sum_hq_c, offset=offset, mask=-1, mask_and_clamp=31
                            )
                            sum_hq_d += cute.arch.shuffle_sync_bfly(
                                sum_hq_d, offset=offset, mask=-1, mask_and_clamp=31
                            )

                        # Reduction result is identical on all lanes -> lane 0
                        # writes. To SMEM (merged flush at kernel end) or GMEM.
                        if lane_in_group == 0:
                            if cutlass.const_expr(use_smem_v):
                                vla = v_idx_a - i_v * tile_v
                                sOutput[(i_t, vla)] = cutlass.BFloat16(sum_hq_a)
                                sOutput[(i_t, vla + 1)] = cutlass.BFloat16(sum_hq_b)
                                sOutput[(i_t, vla + 2)] = cutlass.BFloat16(sum_hq_c)
                                sOutput[(i_t, vla + 3)] = cutlass.BFloat16(sum_hq_d)
                            else:
                                o[(i_n, i_t, i_hv, v_idx_a)] = cutlass.BFloat16(sum_hq_a)
                                o[(i_n, i_t, i_hv, v_idx_b)] = cutlass.BFloat16(sum_hq_b)
                                o[(i_n, i_t, i_hv, v_idx_c)] = cutlass.BFloat16(sum_hq_c)
                                o[(i_n, i_t, i_hv, v_idx_d)] = cutlass.BFloat16(sum_hq_d)

                        # Stage D: snapshot the post-token state. Steps 4+5 are
                        # fused here, so r_h reaches its final (post-token i_t)
                        # value only after the loop above — hence the snapshot is
                        # LAST in the timestep (after the output write), all lanes
                        # participating (each writes its vec_size K-channels). Same
                        # sequence-indexed flat_idx and race-free fire-and-forget
                        # stores as the ilp=2 path (mirrors gdn_decode_mtp.py:1135-1160).
                        if cutlass.const_expr(cache_intermediate_states):
                            flat_idx = i_n * T * HV + i_t * HV + i_hv
                            inter_a = cute.local_tile(
                                intermediate_states,
                                (1, 1, vec_size),
                                (flat_idx, v_idx_a, lane_in_group),
                            )
                            cute.autovec_copy(cute.slice_(r_h, (0, None)), inter_a)
                            inter_b = cute.local_tile(
                                intermediate_states,
                                (1, 1, vec_size),
                                (flat_idx, v_idx_b, lane_in_group),
                            )
                            cute.autovec_copy(cute.slice_(r_h, (1, None)), inter_b)
                            inter_c = cute.local_tile(
                                intermediate_states,
                                (1, 1, vec_size),
                                (flat_idx, v_idx_c, lane_in_group),
                            )
                            cute.autovec_copy(cute.slice_(r_h, (2, None)), inter_c)
                            inter_d = cute.local_tile(
                                intermediate_states,
                                (1, 1, vec_size),
                                (flat_idx, v_idx_d, lane_in_group),
                            )
                            cute.autovec_copy(cute.slice_(r_h, (3, None)), inter_d)

                    # Write final state for all 4 rows back to the pool (once).
                    if cutlass.const_expr(not disable_state_update):
                        h_tile_out_a = cute.local_tile(
                            h0_source,
                            (1, 1, vec_size),
                            (flat_state_idx, v_idx_a, lane_in_group),
                        )
                        cute.autovec_copy(cute.slice_(r_h, (0, None)), h_tile_out_a)
                        h_tile_out_b = cute.local_tile(
                            h0_source,
                            (1, 1, vec_size),
                            (flat_state_idx, v_idx_b, lane_in_group),
                        )
                        cute.autovec_copy(cute.slice_(r_h, (1, None)), h_tile_out_b)
                        h_tile_out_c = cute.local_tile(
                            h0_source,
                            (1, 1, vec_size),
                            (flat_state_idx, v_idx_c, lane_in_group),
                        )
                        cute.autovec_copy(cute.slice_(r_h, (2, None)), h_tile_out_c)
                        h_tile_out_d = cute.local_tile(
                            h0_source,
                            (1, 1, vec_size),
                            (flat_state_idx, v_idx_d, lane_in_group),
                        )
                        cute.autovec_copy(cute.slice_(r_h, (3, None)), h_tile_out_d)

        # ============ Merged output writeback (use_smem_v only) ============
        # Each group wrote its own disjoint tile-local columns of sOutput (lane 0
        # only), so there is no write-write race; the barrier publishes all of
        # those before any thread reads. Then all 128 threads cooperatively flush
        # sOutput -> o, one tile-local column per thread (tidx<tile_v) across all
        # T tokens — consecutive threads hit consecutive v_global, so the GMEM o
        # writes coalesce (vs the per-token lane-0 scatter the non-smem path does).
        # Outside the ilp branches but inside `cache_idx >= 0` (uniform across the
        # CTA), so the barrier never deadlocks. Mirrors FlashInfer
        # gdn_decode_mtp.py:1325-1334.
        if cutlass.const_expr(use_smem_v):
            cute.arch.barrier()
            v_tile_base = i_v * tile_v
            for t_idx in cutlass.range_constexpr(T):
                if tidx < tile_v:
                    v_global = v_tile_base + tidx
                    if v_global < V:
                        o[(i_n, t_idx, i_hv, v_global)] = sOutput[(t_idx, tidx)]


@cute.jit
def run_kda_verify_kernel_mtp_ws(
    h0_source: cute.Tensor,
    intermediate_states: cute.Tensor,
    A_log: cute.Tensor,
    a: cute.Tensor,
    dt_bias: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    b: cute.Tensor,
    o: cute.Tensor,
    h0_indices: cute.Tensor,
    softplus_beta: cutlass.Constexpr[float],
    softplus_threshold: cutlass.Constexpr[float],
    scale: cutlass.Constexpr[float],
    HV: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    tile_v: cutlass.Constexpr[int],
    vec_size: cutlass.Constexpr[int],
    use_qk_l2norm: cutlass.Constexpr[bool],
    disable_state_update: cutlass.Constexpr[bool],
    ilp_rows: cutlass.Constexpr[int],
    use_packed_fma: cutlass.Constexpr[bool],
    use_smem_v: cutlass.Constexpr[bool],
    cache_intermediate_states: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    """Host-side launcher: grid = N * HV * num_v_tiles, block = 128 (4 warps)."""
    n_indices = h0_indices.layout.shape[0]
    v_dim = h0_source.layout.shape[1]
    k_dim = h0_source.layout.shape[2]

    num_v_tiles = cute.ceil_div(v_dim, tile_v)
    grid_size = n_indices * HV * num_v_tiles

    # sQ + sK + sG (all [T, K+8] fp32) + sBeta ([T] fp32) + alignment slack.
    # When use_smem_v, add sVdata ([T, tile_v] fp32) + sOutput ([T, tile_v] bf16),
    # matching the kernel's conditional allocation. All ws buffers use 16B
    # alignment (not Route-2's 128B; see cdf6a89), so the flat +128 slack covers
    # the cumulative per-tensor padding (every fp32 row here is a 16B multiple;
    # only sBeta can need <16B of padding) — no per-tensor 128B rounding needed.
    smem_bytes = (
        4 * T * (k_dim + 8)  # sQ
        + 4 * T * (k_dim + 8)  # sK
        + 4 * T * (k_dim + 8)  # sG (KDA: per-channel; GDN had 4*T scalar)
        + 4 * T  # sBeta
        + 128  # alignment slack
    )
    if cutlass.const_expr(use_smem_v):
        smem_bytes += 4 * T * tile_v  # sVdata (fp32)
        smem_bytes += 2 * T * tile_v  # sOutput (bf16)

    kda_verify_kernel_mtp_ws(
        h0_source,
        intermediate_states,
        vec_size,
        num_v_tiles,
        tile_v,
        A_log,
        a,
        dt_bias,
        q,
        k,
        v,
        b,
        o,
        h0_indices,
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
        ilp_rows,
        use_packed_fma,
        use_smem_v,
        cache_intermediate_states,
    ).launch(
        grid=(grid_size, 1, 1),
        block=[NUM_THREADS, 1, 1],
        smem=smem_bytes,
        stream=stream,
    )


def _get_compiled_mtp_ws_kernel(
    N,
    T,
    H,
    HV,
    K,
    V,
    pool_size,
    scale,
    use_qk_l2norm,
    disable_state_update,
    softplus_beta,
    softplus_threshold,
    tile_v,
    ilp_rows,
    use_packed_fma,
    use_smem_v,
    cache_intermediate_states,
):
    """Get or lazily compile the warp-spec MTP kernel for one shape/config."""
    key = (
        N,
        T,
        H,
        HV,
        K,
        V,
        pool_size,
        scale,
        use_qk_l2norm,
        disable_state_update,
        softplus_beta,
        softplus_threshold,
        tile_v,
        ilp_rows,
        use_packed_fma,
        use_smem_v,
        cache_intermediate_states,
    )
    if key in _compiled_mtp_ws_kernels:
        return _compiled_mtp_ws_kernels[key]

    q = torch.zeros(N, T, H, K, dtype=torch.bfloat16, device="cuda")
    k = torch.zeros(N, T, H, K, dtype=torch.bfloat16, device="cuda")
    v = torch.zeros(N, T, HV, V, dtype=torch.bfloat16, device="cuda")
    a = torch.zeros(N, T, HV, K, dtype=torch.bfloat16, device="cuda")
    b = torch.zeros(N, T, HV, dtype=torch.bfloat16, device="cuda")
    o = torch.zeros(N, T, HV, V, dtype=torch.bfloat16, device="cuda")
    A_log = torch.zeros(HV, dtype=torch.float32, device="cuda")
    dt_bias = torch.zeros(HV, K, dtype=torch.float32, device="cuda")
    # Warp-spec kernel uses the flat 3D state view [pool*HV, V, K] (VK layout).
    h0_source = torch.zeros(pool_size * HV, V, K, dtype=torch.float32, device="cuda")
    h0_indices = torch.zeros(N, dtype=torch.int32, device="cuda")
    # Stage D intermediate-state cache, indexed by SEQUENCE: flattened
    # [N*T*HV, V, K] (= [N, T, HV, V, K] vk). When not caching, a tiny dummy
    # (the kernel never touches it; cache_intermediate_states=False compiles the
    # snapshot stores out). Same first-dim convention as the real runtime tensor.
    if cache_intermediate_states:
        intermediate_states = torch.zeros(
            N * T * HV, V, K, dtype=torch.float32, device="cuda"
        )
    else:
        intermediate_states = torch.zeros(1, 1, 1, dtype=torch.float32, device="cuda")

    q_tensor = from_dlpack(q, assumed_align=16)
    k_tensor = from_dlpack(k, assumed_align=16)
    v_tensor = from_dlpack(v, assumed_align=16)
    a_tensor = from_dlpack(a, assumed_align=16)
    b_tensor = from_dlpack(b, assumed_align=16)
    A_log_tensor = from_dlpack(A_log, assumed_align=16)
    dt_bias_tensor = from_dlpack(dt_bias, assumed_align=16)
    h0_source_tensor = from_dlpack(h0_source, assumed_align=16)
    h0_indices_tensor = from_dlpack(h0_indices, assumed_align=16)
    o_tensor = from_dlpack(o, assumed_align=16)
    intermediate_states_tensor = from_dlpack(intermediate_states, assumed_align=16)

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    compiled_kernel = cute.compile(
        run_kda_verify_kernel_mtp_ws,
        h0_source_tensor,
        intermediate_states_tensor,
        A_log_tensor,
        a_tensor,
        dt_bias_tensor,
        q_tensor,
        k_tensor,
        v_tensor,
        b_tensor,
        o_tensor,
        h0_indices_tensor,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        scale=scale,
        HV=HV,
        T=T,
        H=H,
        K=K,
        V=V,
        tile_v=tile_v,
        vec_size=VEC_SIZE_MTP,
        use_qk_l2norm=use_qk_l2norm,
        disable_state_update=disable_state_update,
        ilp_rows=ilp_rows,
        use_packed_fma=use_packed_fma,
        use_smem_v=use_smem_v,
        cache_intermediate_states=cache_intermediate_states,
        stream=stream,
        options="--enable-tvm-ffi --opt-level 1",
    )

    _compiled_mtp_ws_kernels[key] = compiled_kernel
    logger.info(
        "CuTe DSL KDA MTP warp-spec kernel compiled: "
        f"N={N}, T={T}, H={H}, HV={HV}, K={K}, V={V}, pool_size={pool_size}, "
        f"tile_v={tile_v}, ilp_rows={ilp_rows}, use_packed_fma={use_packed_fma}, "
        f"use_smem_v={use_smem_v}, cache_intermediate_states={cache_intermediate_states}"
    )
    return compiled_kernel


def kda_decode_mtp_ws(
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
    tile_v: int | None = None,
    ilp_rows: int | None = None,
    disable_state_update: bool = False,
    use_packed_fma: bool | None = None,
    use_smem_v: bool | None = None,
    intermediate_states_buffer: torch.Tensor | None = None,
) -> torch.Tensor:
    """KDA MTP decode — warp-specialized variant (Route 1).

    Drop-in alternative to ``kda_decode_mtp`` with the same public contract,
    using FlashInfer's warp-specialized organization (grid = N*HV*num_v_tiles,
    register-resident state, full-warp shuffle reduction) and per-channel KDA
    decay. Kept separate so both routes can be benchmarked head-to-head.

    Dense MTP shapes (same as ``kda_decode_mtp``):
        q/k: (N, T, H, K)   v: (N, T, HV, V)
        a:   (N, T, HV, K)  b: (N, T, HV)   out: (N, T, HV, V)

    ``ilp_rows`` selects how many V-rows a warp processes together: 2 (any valid
    ``tile_v``) or 4 (requires ``tile_v % 16 == 0``; steps 1+2 and 4+5 are fused
    with double accumulators + packed F32x2 FMA on SM100). ``ilp_rows=None``
    (default) picks it from the ``work_units=N*HV`` heuristic
    (:func:`_select_mtp_config`), mirroring ``tile_v=None``; an explicit value
    overrides. If an explicit ``tile_v`` makes the heuristic's ilp=4 illegal
    (``tile_v % 16 != 0``) the auto path falls back to ilp=2 (an explicit
    ``ilp_rows=4`` with such a ``tile_v`` still asserts, by design).
    ``use_packed_fma=None`` auto-detects SM100+ (Blackwell); pass False to force
    the scalar fallback.

    ``use_smem_v`` (Stage C) preloads the CTA's v-tile into SMEM (one cooperative
    load instead of a per-token GMEM read) and accumulates outputs in SMEM for a
    single coalesced merged writeback at kernel end (instead of the per-token
    lane-0 scatter). ``use_smem_v=None`` (default) takes it from the
    ``work_units=N*HV`` heuristic (:func:`_select_mtp_config`), which enables it
    only for the large-batch (tile_v=64) bucket; an explicit bool overrides. It
    is independent of ``ilp_rows`` and works with any ``tile_v``.

    ``intermediate_states_buffer`` (Stage D, speculative-decoding support): an
    optional fp32, contiguous tensor of shape ``[N, T, HV, V, K]`` (vk / K-last).
    When given, the kernel snapshots the post-token state after EVERY token to
    ``buffer[i_n, i_t, i_hv]`` (indexed by SEQUENCE position, so a serving layer
    can roll back to the snapshot at the last accepted token); ``buffer[:, T-1]``
    equals the final state written to the pool. When ``None`` (default) no
    snapshots are taken. Produce-only — cuLA does NOT implement the rollback;
    the caller owns the buffer and the rollback policy. The return value is
    always just ``o`` (the buffer is filled in place).

    Constraints: ``state_layout='vk'`` only; ``ilp_rows in {2, 4}``.
    """
    N, T, H, K = q.shape
    HV = v.shape[2]
    V = v.shape[3]

    if scale is None:
        scale = K**-0.5
    else:
        assert scale > 0, f"scale must be positive, got {scale}"

    assert K == TILE_K, f"KDA MTP (ws) kernel requires K={TILE_K}, got {K}"

    # Resolve tile_v / ilp_rows / use_smem_v from the work_units=N*HV heuristic
    # where not given explicitly (mirrors kda_decode_mtp). An explicit tile_v can
    # make the heuristic's ilp=4 illegal (needs tile_v % 16 == 0); in the auto
    # path we fall back to the universally-legal ilp=2 rather than tripping the
    # rows_per_group assert below. (When tile_v also came from the heuristic,
    # _select_mtp_config already applied this backstop, so the guard is a no-op.)
    # use_smem_v is independent of tile_v legality (preloading v works for any
    # tile_v); the heuristic turns it on only for the large-batch (tile_v=64)
    # bucket. An explicit use_smem_v overrides.
    if tile_v is None or ilp_rows is None or use_smem_v is None:
        sel_tile_v, sel_ilp_rows, sel_use_smem_v = _select_mtp_config(
            N, HV, V, T, disable_state_update=disable_state_update
        )
        if tile_v is None:
            tile_v = sel_tile_v
        if ilp_rows is None:
            ilp_rows = sel_ilp_rows
            if ilp_rows == 4 and tile_v % 16 != 0:
                ilp_rows = 2
        if use_smem_v is None:
            use_smem_v = sel_use_smem_v

    if ilp_rows not in (2, 4):
        raise NotImplementedError(
            f"kda_decode_mtp_ws implements ilp_rows in {{2, 4}}, got {ilp_rows}"
        )

    # packed F32x2 FMA exists only on SM100+ (Blackwell); fall back to scalar
    # fma_pair elsewhere. None = auto-detect, matching FlashInfer's run_mtp_decode.
    if use_packed_fma is None:
        major, _ = torch.cuda.get_device_capability(q.device)
        use_packed_fma = major >= 10
    # The packed path only exists in the ilp=4 kernel branch; ilp=2 is scalar.
    if ilp_rows != 4:
        use_packed_fma = False

    state_layout = _canonicalize_state_layout(state_layout)
    if state_layout != "vk":
        raise NotImplementedError(
            "kda_decode_mtp_ws only supports state_layout='vk' "
            f"(FlashInfer is vk-only); got {state_layout!r}"
        )

    assert tile_v % 4 == 0, f"KDA MTP (ws) requires tile_v % 4 == 0, got tile_v={tile_v}"
    assert V % tile_v == 0, f"KDA MTP (ws) requires V % tile_v == 0, got V={V}, tile_v={tile_v}"
    # Each warp owns rows_per_group = tile_v/4 V-rows and steps through ilp_rows of
    # them per iteration, so rows_per_group must be a multiple of ilp_rows — else
    # the row_pair/row_quad loop count truncates and trailing rows are silently
    # skipped. ilp=2 -> tile_v % 8 == 0; ilp=4 -> tile_v % 16 == 0.
    rows_per_group = tile_v // 4
    assert rows_per_group % ilp_rows == 0, (
        f"ilp_rows={ilp_rows} requires (tile_v//4) divisible by {ilp_rows}, "
        f"got tile_v={tile_v} (tile_v//4={rows_per_group})"
    )

    # State is token-independent: reuse the single-token normalizer/validator.
    h0_source, pool_size, state_layout_is_kv = _normalize_state_source(
        initial_state_source,
        N=N,
        HV=HV,
        K=K,
        V=V,
        device=q.device,
        state_layout=state_layout,
    )
    assert not state_layout_is_kv  # guaranteed by the vk-only guard above

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

    # Flatten the VK state pool [pool, HV, V, K] -> [pool*HV, V, K]; the kernel
    # indexes flat_state_idx = cache_idx*HV + i_hv. .view() shares storage so the
    # kernel's in-place state writeback lands in the caller's tensor; it raises
    # (loudly, not a silent copy) if the pool is not contiguous, which would
    # break the in-place contract anyway.
    h0_source_flat = h0_source.view(pool_size * HV, V, K)

    # Stage D: resolve the intermediate-state snapshot cache. A buffer turns
    # snapshots on (constexpr); flatten [N, T, HV, V, K] -> [N*T*HV, V, K] so the
    # kernel's flat_idx = i_n*T*HV + i_t*HV + i_hv (indexed by SEQUENCE i_n, NOT
    # the pool slot cache_idx used for the state writeback) lands in the caller's
    # tensor. .view() shares storage + raises loudly on a non-contiguous buffer,
    # keeping the in-place fill contract (cf. h0_source_flat above). Unlike
    # FlashInfer (which .to(fp32).reshape().contiguous() — silently dropping the
    # snapshots if the buffer was not already contiguous fp32) we require fp32 +
    # contiguous up front so the fill is always visible. None -> 1-elem dummy and
    # the snapshot stores compile out (cache_intermediate_states=False).
    cache_intermediate_states = intermediate_states_buffer is not None
    if cache_intermediate_states:
        if intermediate_states_buffer.dtype != torch.float32:
            raise ValueError(
                "intermediate_states_buffer must be float32, got "
                f"{intermediate_states_buffer.dtype}"
            )
        expected_buf_shape = (N, T, HV, V, K)
        if tuple(intermediate_states_buffer.shape) != expected_buf_shape:
            raise ValueError(
                f"intermediate_states_buffer shape {tuple(intermediate_states_buffer.shape)} "
                f"!= expected {expected_buf_shape} ([N, T, HV, V, K] vk / K-last)"
            )
        intermediate_states_flat = intermediate_states_buffer.view(N * T * HV, V, K)
    else:
        intermediate_states_flat = torch.zeros(
            1, 1, 1, dtype=torch.float32, device=q.device
        )

    stream = _get_cached_stream(q.device)
    compiled_kernel = _get_compiled_mtp_ws_kernel(
        N,
        T,
        H,
        HV,
        K,
        V,
        pool_size,
        scale=scale,
        use_qk_l2norm=use_qk_l2norm_in_kernel,
        disable_state_update=disable_state_update,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        tile_v=tile_v,
        ilp_rows=ilp_rows,
        use_packed_fma=use_packed_fma,
        use_smem_v=use_smem_v,
        cache_intermediate_states=cache_intermediate_states,
    )

    compiled_kernel(
        h0_source_flat,
        intermediate_states_flat,
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
# Inline variant (small batch, work_units <= 128).
#
# Ported from FlashInfer ``gdn_verify_kernel_mtp_inline``
# (``gdn_decode_mtp.py:1438-2110``), Apache-2.0. FlashInfer dispatches to the
# inline kernel when ``B*HV <= 128`` (BS<=2 at HV=64) and to the warp-spec kernel
# above when ``> 128`` (``run_mtp_decode:2309``). Both share the exact same
# grid/CTA decomposition (``N*HV*num_v_tiles``, one (i_n, i_hv, i_v) tile per CTA,
# vec_size=4 full-warp K-reduce, 4 warps x rows_per_group V-rows) and the same
# ilp 2/4 axis. The inline variant differs only in how q/k/g/beta are staged:
#   - NO warp-0 staging / SMEM broadcast (no sQ/sK/sG/sBeta) and NO Phase-1
#     barrier. Each of the 4 warps computes q/k/g/beta inline, register-resident
#     — 4x redundant g/beta transcendentals, but at small batch the kernel is
#     latency-bound, so the warp-spec staging+barrier fixed cost (which has no T
#     amortization or grid occupancy to hide it there) is the thing to remove.
#   - DEFERRED L2 norm: q/k stay RAW in registers; the 1/||q||, 1/||k|| factors
#     are computed inside step 2 / step 5 (sum_sq_k, sum_sq_q piggyback the h@k /
#     h@q reductions) and applied as SCALARS after the reduction, never
#     materialized onto the q/k registers. (The warp-spec kernel L2-norms eagerly
#     in Phase 1.) This is orthogonal to the per-channel decay — the norm acts on
#     the q/k vectors, the decay on the state S — so it is copied verbatim.
#   - register-pipelined q/k: the ilp=4 path loads q[0]/k[0] in a prologue and
#     prefetches q[t+1]/k[t+1] at the end of each timestep; the ilp=2 path
#     batch-loads all T q/k (and all L2 norms) up front. State is register-
#     resident across T either way.
#
# The ONLY KDA change vs GDN is, again, the decay: GDN's per-(head, token) SCALAR
# gate register array ``r_g_arr[T]`` becomes the per-channel ``r_g_arr[T, vec_size]``
# (each lane precomputes g for its own vec_size K-channels, exactly as warp 0
# stages sG[T, K] in the warp-spec kernel), and the decay step ``r_h[row,i] *= r_g``
# becomes ``r_h[row,i] *= r_g_arr[i_t, i]``. beta stays a scalar array. The inline
# ilp=4 path uses plain scalar FMA (no packed-FMA / double-accumulator — matches
# FlashInfer); ``use_packed_fma`` is accepted for launcher/signature parity but
# unused in the body. All transcendentals drop ``fastmath`` to match the warp-spec
# kernel / ``kda_decode.py`` (keeps the inline g/softplus bit-comparable to ws).
# ============================================================================


@cute.kernel
def kda_verify_kernel_mtp_inline(
    h0_source: cute.Tensor,  # [pool_size * HV, V, K] fp32, K-last (VK layout)
    intermediate_states: cute.Tensor,  # [N*T*HV, V, K] fp32 snapshot cache (or dummy)
    vec_size: cutlass.Constexpr[int],
    num_v_tiles: cutlass.Constexpr[int],
    tile_v: cutlass.Constexpr[int],
    A_log: cute.Tensor,  # [HV] fp32
    a: cute.Tensor,  # [N, T, HV, K] (KDA: per-channel decay input)
    dt_bias: cute.Tensor,  # [HV, K] (KDA: per-channel decay bias)
    q: cute.Tensor,  # [N, T, H, K]
    k: cute.Tensor,  # [N, T, H, K]
    v: cute.Tensor,  # [N, T, HV, V]
    b: cute.Tensor,  # [N, T, HV]
    o: cute.Tensor,  # [N, T, HV, V] output
    h0_indices: cute.Tensor,  # [N] int32 (state-pool slot per sequence; <0 = pad)
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
    ilp_rows: cutlass.Constexpr[int],
    use_packed_fma: cutlass.Constexpr[bool],  # signature parity; unused (scalar FMA)
    use_smem_v: cutlass.Constexpr[bool],
    cache_intermediate_states: cutlass.Constexpr[bool],
):
    """Inline KDA MTP kernel. One CTA owns one (i_n, i_hv, i_v) tile.

    No warp specialization: all 4 warps run the T-step recurrence with state
    register-resident, each computing q/k/g/beta inline (deferred L2 norm). g is
    precomputed per-channel into a register array shared across this lane's V-rows.
    Best at small batch (work_units = N*HV <= 128), where removing the warp-spec
    staging + Phase-1 barrier wins over reusing it. ``use_smem_v`` (preload v-tile
    + coalesced merged output writeback) and ``cache_intermediate_states`` (Stage-D
    fire-and-forget post-token snapshots, sequence-indexed) work exactly as in the
    warp-spec kernel.
    """
    tidx, _, _ = cute.arch.thread_idx()
    lane_id = tidx % 32
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)

    # vec_size=4 -> threads_per_group=32 (full warp), 4 groups (one per warp).
    threads_per_group: cutlass.Constexpr[int] = K // vec_size  # 32
    num_groups: cutlass.Constexpr[int] = 4
    lane_in_group = lane_id % threads_per_group
    group_idx = warp_idx

    batch_idx, _, _ = cute.arch.block_idx()

    # Decode the flat CTA index into (i_n sequence, i_hv value-head, i_v V-tile).
    i_v = batch_idx % num_v_tiles
    tmp = batch_idx // num_v_tiles
    i_hv = tmp % HV
    i_n = tmp // HV
    i_h = i_hv // (HV // H)  # GVA: HV//H value-heads share one q/k head

    cache_idx = h0_indices[i_n]

    # exp(A_log) is per-head, shared across all K channels (KDA), hoisted once.
    r_A_log = cutlass.Float32(A_log[i_hv])
    r_exp_A = cute.exp(r_A_log)

    # Inline: NO sQ/sK/sG/sBeta. Only sVdata/sOutput, and only under use_smem_v
    # (allocated conditionally so the off-path footprint is just alignment slack).
    smem = cutlass.utils.SmemAllocator()
    if cutlass.const_expr(use_smem_v):
        sVdata = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((T, tile_v), stride=(tile_v, 1)), 16
        )
        sOutput = smem.allocate_tensor(
            cutlass.BFloat16, cute.make_layout((T, tile_v), stride=(tile_v, 1)), 16
        )

    # Per-lane registers (raw q/k under deferred norm). r_h holds up to 8 V-rows
    # of state (only ilp_rows used); each r_h[row] spans the warp's 32 lanes to
    # cover all K=128 channels.
    r_q = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    r_k = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    r_h = cute.make_rmem_tensor(
        cute.make_layout((8, vec_size), stride=(vec_size, 1)), cutlass.Float32
    )
    r_q_bf16 = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16)
    r_k_bf16 = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16)

    if cache_idx >= 0:
        k_start = lane_in_group * vec_size  # this lane's first K channel
        rows_per_group: cutlass.Constexpr[int] = tile_v // num_groups
        flat_state_idx = cache_idx * HV + i_hv  # row in [pool*HV, V, K]

        # use_smem_v: preload this CTA's v-tile into SMEM (all 128 threads, each
        # writing its tile-local column once: tidx<tile_v). No warp-spec split
        # here — one cooperative loop, published by the barrier below.
        if cutlass.const_expr(use_smem_v):
            for i_t in cutlass.range_constexpr(T):
                if tidx < tile_v:
                    v_global_idx = i_v * tile_v + tidx
                    if v_global_idx < V:
                        sVdata[(i_t, tidx)] = cutlass.Float32(
                            v[i_n, i_t, i_hv, v_global_idx]
                        )
            cute.arch.barrier()

        # KDA per-channel decay gate, precomputed for ALL tokens into a register
        # array shared by this lane's V-rows (avoids recomputing softplus/sigmoid
        # in every V-row iteration). Unlike GDN's scalar r_g_arr[T], r_g_arr is
        # [T, vec_size]: each lane fills g for its own vec_size K-channels (k_start
        # .. k_start+vec_size), so the warp's 32 lanes together cover all K=128
        # — and the decay step reads r_g_arr[i_t, i] for the channel r_h[row,i]
        # holds. beta is a per-(head, token) scalar.
        r_g_arr = cute.make_rmem_tensor(
            cute.make_layout((T, vec_size), stride=(vec_size, 1)), cutlass.Float32
        )
        r_beta_arr = cute.make_rmem_tensor(
            cute.make_layout((T,), stride=(1,)), cutlass.Float32
        )
        for i_t in cutlass.range_constexpr(T):
            r_b_val = cutlass.Float32(b[i_n, i_t, i_hv])
            r_beta_arr[i_t] = cutlass.Float32(1.0) / (
                cutlass.Float32(1.0) + cute.exp(-r_b_val)
            )
            for i in cutlass.range_constexpr(vec_size):
                kk = k_start + i
                x = cutlass.Float32(a[i_n, i_t, i_hv, kk]) + cutlass.Float32(
                    dt_bias[i_hv, kk]
                )
                beta_x = softplus_beta * x
                exp_beta_x = cute.exp(beta_x)
                softplus_val = (cutlass.Float32(1.0) / softplus_beta) * cute.log(
                    cutlass.Float32(1.0) + exp_beta_x
                )
                use_softplus = (
                    cutlass.Float32(1.0)
                    if beta_x <= softplus_threshold
                    else cutlass.Float32(0.0)
                )
                softplus_x = (
                    use_softplus * softplus_val
                    + (cutlass.Float32(1.0) - use_softplus) * x
                )
                r_g_arr[(i_t, i)] = cute.exp(-r_exp_A * softplus_x)

        # ============ Recurrence: ilp_rows == 4 (process 4 V-rows together) ===
        # Plain scalar FMA + deferred L2 norm + register-pipelined q/k. KDA change:
        # per-channel decay r_g_arr[i_t, i] (no scalar r_g). Mirrors FlashInfer
        # gdn_decode_mtp.py:1594-1902.
        if cutlass.const_expr(ilp_rows == 4):
            quarter_rows: cutlass.Constexpr[int] = rows_per_group // 4

            for row_quad in cutlass.range_constexpr(quarter_rows):
                v_idx_a = i_v * tile_v + group_idx * rows_per_group + row_quad * 4
                v_idx_b = v_idx_a + 1
                v_idx_c = v_idx_a + 2
                v_idx_d = v_idx_a + 3

                if v_idx_d < V:
                    # Load state for 4 rows.
                    h_tile_a = cute.local_tile(
                        h0_source, (1, 1, vec_size), (flat_state_idx, v_idx_a, lane_in_group)
                    )
                    h_tile_b = cute.local_tile(
                        h0_source, (1, 1, vec_size), (flat_state_idx, v_idx_b, lane_in_group)
                    )
                    h_tile_c = cute.local_tile(
                        h0_source, (1, 1, vec_size), (flat_state_idx, v_idx_c, lane_in_group)
                    )
                    h_tile_d = cute.local_tile(
                        h0_source, (1, 1, vec_size), (flat_state_idx, v_idx_d, lane_in_group)
                    )
                    cute.autovec_copy(h_tile_a, cute.slice_(r_h, (0, None)))
                    cute.autovec_copy(h_tile_b, cute.slice_(r_h, (1, None)))
                    cute.autovec_copy(h_tile_c, cute.slice_(r_h, (2, None)))
                    cute.autovec_copy(h_tile_d, cute.slice_(r_h, (3, None)))

                    # Prologue: load q[0]/k[0] raw (deferred norm). q gets *scale
                    # only in the no-l2norm path; otherwise scale folds into the
                    # deferred inv_norm_q.
                    q_tile = cute.local_tile(q, (1, 1, 1, vec_size), (i_n, 0, i_h, lane_in_group))
                    k_tile = cute.local_tile(k, (1, 1, 1, vec_size), (i_n, 0, i_h, lane_in_group))
                    cute.autovec_copy(q_tile, r_q_bf16)
                    cute.autovec_copy(k_tile, r_k_bf16)
                    for i in cutlass.range_constexpr(vec_size):
                        r_q[i] = cutlass.Float32(r_q_bf16[i])
                        r_k[i] = cutlass.Float32(r_k_bf16[i])
                    if cutlass.const_expr(not use_qk_l2norm):
                        for i in cutlass.range_constexpr(vec_size):
                            r_q[i] = r_q[i] * scale

                    for i_t in cutlass.range_constexpr(T):
                        r_beta = r_beta_arr[i_t]

                        # Step 1: per-channel decay of all 4 rows (KDA: r_g_arr[i_t, i]).
                        for i in cutlass.range_constexpr(vec_size):
                            r_h[0, i] = r_h[0, i] * r_g_arr[(i_t, i)]
                            r_h[1, i] = r_h[1, i] * r_g_arr[(i_t, i)]
                            r_h[2, i] = r_h[2, i] * r_g_arr[(i_t, i)]
                            r_h[3, i] = r_h[3, i] * r_g_arr[(i_t, i)]

                        # Step 2: h@k with deferred L2 norm (sum_sq_k piggybacks).
                        sum_hk_a = 0.0
                        sum_hk_b = 0.0
                        sum_hk_c = 0.0
                        sum_hk_d = 0.0
                        if cutlass.const_expr(use_qk_l2norm):
                            sum_sq_k = 0.0
                            for i in cutlass.range_constexpr(vec_size):
                                sum_hk_a += r_h[0, i] * r_k[i]
                                sum_hk_b += r_h[1, i] * r_k[i]
                                sum_hk_c += r_h[2, i] * r_k[i]
                                sum_hk_d += r_h[3, i] * r_k[i]
                                sum_sq_k += r_k[i] * r_k[i]
                            for offset in [16, 8, 4, 2, 1]:
                                sum_hk_a += cute.arch.shuffle_sync_bfly(
                                    sum_hk_a, offset=offset, mask=-1, mask_and_clamp=31
                                )
                                sum_hk_b += cute.arch.shuffle_sync_bfly(
                                    sum_hk_b, offset=offset, mask=-1, mask_and_clamp=31
                                )
                                sum_hk_c += cute.arch.shuffle_sync_bfly(
                                    sum_hk_c, offset=offset, mask=-1, mask_and_clamp=31
                                )
                                sum_hk_d += cute.arch.shuffle_sync_bfly(
                                    sum_hk_d, offset=offset, mask=-1, mask_and_clamp=31
                                )
                                sum_sq_k += cute.arch.shuffle_sync_bfly(
                                    sum_sq_k, offset=offset, mask=-1, mask_and_clamp=31
                                )
                            inv_norm_k = cute.rsqrt(sum_sq_k + 1e-6)
                            sum_hk_a = sum_hk_a * inv_norm_k
                            sum_hk_b = sum_hk_b * inv_norm_k
                            sum_hk_c = sum_hk_c * inv_norm_k
                            sum_hk_d = sum_hk_d * inv_norm_k
                        else:
                            for i in cutlass.range_constexpr(vec_size):
                                sum_hk_a += r_h[0, i] * r_k[i]
                                sum_hk_b += r_h[1, i] * r_k[i]
                                sum_hk_c += r_h[2, i] * r_k[i]
                                sum_hk_d += r_h[3, i] * r_k[i]
                            for offset in [16, 8, 4, 2, 1]:
                                sum_hk_a += cute.arch.shuffle_sync_bfly(
                                    sum_hk_a, offset=offset, mask=-1, mask_and_clamp=31
                                )
                                sum_hk_b += cute.arch.shuffle_sync_bfly(
                                    sum_hk_b, offset=offset, mask=-1, mask_and_clamp=31
                                )
                                sum_hk_c += cute.arch.shuffle_sync_bfly(
                                    sum_hk_c, offset=offset, mask=-1, mask_and_clamp=31
                                )
                                sum_hk_d += cute.arch.shuffle_sync_bfly(
                                    sum_hk_d, offset=offset, mask=-1, mask_and_clamp=31
                                )

                        # Step 3: delta rule. v from SMEM (preloaded) or GMEM.
                        if cutlass.const_expr(use_smem_v):
                            v_local_a = v_idx_a - i_v * tile_v
                            r_v_a = sVdata[(i_t, v_local_a)]
                            r_v_b = sVdata[(i_t, v_local_a + 1)]
                            r_v_c = sVdata[(i_t, v_local_a + 2)]
                            r_v_d = sVdata[(i_t, v_local_a + 3)]
                        else:
                            r_v_a = cutlass.Float32(v[i_n, i_t, i_hv, v_idx_a])
                            r_v_b = cutlass.Float32(v[i_n, i_t, i_hv, v_idx_b])
                            r_v_c = cutlass.Float32(v[i_n, i_t, i_hv, v_idx_c])
                            r_v_d = cutlass.Float32(v[i_n, i_t, i_hv, v_idx_d])
                        v_new_a = (r_v_a - sum_hk_a) * r_beta
                        v_new_b = (r_v_b - sum_hk_b) * r_beta
                        v_new_c = (r_v_c - sum_hk_c) * r_beta
                        v_new_d = (r_v_d - sum_hk_d) * r_beta

                        # Step 4: rank-1 update with raw k (deferred: ks = inv_norm_k * v_new).
                        if cutlass.const_expr(use_qk_l2norm):
                            ks_a = inv_norm_k * v_new_a
                            ks_b = inv_norm_k * v_new_b
                            ks_c = inv_norm_k * v_new_c
                            ks_d = inv_norm_k * v_new_d
                            for i in cutlass.range_constexpr(vec_size):
                                r_h[0, i] += r_k[i] * ks_a
                                r_h[1, i] += r_k[i] * ks_b
                                r_h[2, i] += r_k[i] * ks_c
                                r_h[3, i] += r_k[i] * ks_d
                        else:
                            for i in cutlass.range_constexpr(vec_size):
                                r_h[0, i] += r_k[i] * v_new_a
                                r_h[1, i] += r_k[i] * v_new_b
                                r_h[2, i] += r_k[i] * v_new_c
                                r_h[3, i] += r_k[i] * v_new_d

                        # Stage D: snapshot post-token state (sequence-indexed),
                        # fire-and-forget, before step 5 (mirrors gdn_decode_mtp.py:1749).
                        if cutlass.const_expr(cache_intermediate_states):
                            flat_idx = i_n * T * HV + i_t * HV + i_hv
                            inter_a = cute.local_tile(
                                intermediate_states, (1, 1, vec_size), (flat_idx, v_idx_a, lane_in_group)
                            )
                            cute.autovec_copy(cute.slice_(r_h, (0, None)), inter_a)
                            inter_b = cute.local_tile(
                                intermediate_states, (1, 1, vec_size), (flat_idx, v_idx_b, lane_in_group)
                            )
                            cute.autovec_copy(cute.slice_(r_h, (1, None)), inter_b)
                            inter_c = cute.local_tile(
                                intermediate_states, (1, 1, vec_size), (flat_idx, v_idx_c, lane_in_group)
                            )
                            cute.autovec_copy(cute.slice_(r_h, (2, None)), inter_c)
                            inter_d = cute.local_tile(
                                intermediate_states, (1, 1, vec_size), (flat_idx, v_idx_d, lane_in_group)
                            )
                            cute.autovec_copy(cute.slice_(r_h, (3, None)), inter_d)

                        # Step 5: h@q with deferred L2 norm (sum_sq_q piggybacks; scale folds in).
                        sum_hq_a = 0.0
                        sum_hq_b = 0.0
                        sum_hq_c = 0.0
                        sum_hq_d = 0.0
                        if cutlass.const_expr(use_qk_l2norm):
                            sum_sq_q = 0.0
                            for i in cutlass.range_constexpr(vec_size):
                                sum_hq_a += r_h[0, i] * r_q[i]
                                sum_hq_b += r_h[1, i] * r_q[i]
                                sum_hq_c += r_h[2, i] * r_q[i]
                                sum_hq_d += r_h[3, i] * r_q[i]
                                sum_sq_q += r_q[i] * r_q[i]
                            for offset in [16, 8, 4, 2, 1]:
                                sum_hq_a += cute.arch.shuffle_sync_bfly(
                                    sum_hq_a, offset=offset, mask=-1, mask_and_clamp=31
                                )
                                sum_hq_b += cute.arch.shuffle_sync_bfly(
                                    sum_hq_b, offset=offset, mask=-1, mask_and_clamp=31
                                )
                                sum_hq_c += cute.arch.shuffle_sync_bfly(
                                    sum_hq_c, offset=offset, mask=-1, mask_and_clamp=31
                                )
                                sum_hq_d += cute.arch.shuffle_sync_bfly(
                                    sum_hq_d, offset=offset, mask=-1, mask_and_clamp=31
                                )
                                sum_sq_q += cute.arch.shuffle_sync_bfly(
                                    sum_sq_q, offset=offset, mask=-1, mask_and_clamp=31
                                )
                            inv_norm_q_scaled = cute.rsqrt(sum_sq_q + 1e-6) * scale
                            sum_hq_a = sum_hq_a * inv_norm_q_scaled
                            sum_hq_b = sum_hq_b * inv_norm_q_scaled
                            sum_hq_c = sum_hq_c * inv_norm_q_scaled
                            sum_hq_d = sum_hq_d * inv_norm_q_scaled
                        else:
                            for i in cutlass.range_constexpr(vec_size):
                                sum_hq_a += r_h[0, i] * r_q[i]
                                sum_hq_b += r_h[1, i] * r_q[i]
                                sum_hq_c += r_h[2, i] * r_q[i]
                                sum_hq_d += r_h[3, i] * r_q[i]
                            for offset in [16, 8, 4, 2, 1]:
                                sum_hq_a += cute.arch.shuffle_sync_bfly(
                                    sum_hq_a, offset=offset, mask=-1, mask_and_clamp=31
                                )
                                sum_hq_b += cute.arch.shuffle_sync_bfly(
                                    sum_hq_b, offset=offset, mask=-1, mask_and_clamp=31
                                )
                                sum_hq_c += cute.arch.shuffle_sync_bfly(
                                    sum_hq_c, offset=offset, mask=-1, mask_and_clamp=31
                                )
                                sum_hq_d += cute.arch.shuffle_sync_bfly(
                                    sum_hq_d, offset=offset, mask=-1, mask_and_clamp=31
                                )

                        # Write output (lane 0): to SMEM (merged flush) or GMEM.
                        if lane_in_group == 0:
                            if cutlass.const_expr(use_smem_v):
                                vla = v_idx_a - i_v * tile_v
                                sOutput[(i_t, vla)] = cutlass.BFloat16(sum_hq_a)
                                sOutput[(i_t, vla + 1)] = cutlass.BFloat16(sum_hq_b)
                                sOutput[(i_t, vla + 2)] = cutlass.BFloat16(sum_hq_c)
                                sOutput[(i_t, vla + 3)] = cutlass.BFloat16(sum_hq_d)
                            else:
                                o[(i_n, i_t, i_hv, v_idx_a)] = cutlass.BFloat16(sum_hq_a)
                                o[(i_n, i_t, i_hv, v_idx_b)] = cutlass.BFloat16(sum_hq_b)
                                o[(i_n, i_t, i_hv, v_idx_c)] = cutlass.BFloat16(sum_hq_c)
                                o[(i_n, i_t, i_hv, v_idx_d)] = cutlass.BFloat16(sum_hq_d)

                        # Prefetch q/k for the next timestep (raw; g/beta are already
                        # in the register arrays). State stays register-resident.
                        if cutlass.const_expr(i_t + 1 < T):
                            q_tile = cute.local_tile(
                                q, (1, 1, 1, vec_size), (i_n, i_t + 1, i_h, lane_in_group)
                            )
                            k_tile = cute.local_tile(
                                k, (1, 1, 1, vec_size), (i_n, i_t + 1, i_h, lane_in_group)
                            )
                            cute.autovec_copy(q_tile, r_q_bf16)
                            cute.autovec_copy(k_tile, r_k_bf16)
                            for i in cutlass.range_constexpr(vec_size):
                                r_q[i] = cutlass.Float32(r_q_bf16[i])
                                r_k[i] = cutlass.Float32(r_k_bf16[i])
                            if cutlass.const_expr(not use_qk_l2norm):
                                for i in cutlass.range_constexpr(vec_size):
                                    r_q[i] = r_q[i] * scale

                    # Write final state for all 4 rows back (once).
                    if cutlass.const_expr(not disable_state_update):
                        h_tile_out_a = cute.local_tile(
                            h0_source, (1, 1, vec_size), (flat_state_idx, v_idx_a, lane_in_group)
                        )
                        cute.autovec_copy(cute.slice_(r_h, (0, None)), h_tile_out_a)
                        h_tile_out_b = cute.local_tile(
                            h0_source, (1, 1, vec_size), (flat_state_idx, v_idx_b, lane_in_group)
                        )
                        cute.autovec_copy(cute.slice_(r_h, (1, None)), h_tile_out_b)
                        h_tile_out_c = cute.local_tile(
                            h0_source, (1, 1, vec_size), (flat_state_idx, v_idx_c, lane_in_group)
                        )
                        cute.autovec_copy(cute.slice_(r_h, (2, None)), h_tile_out_c)
                        h_tile_out_d = cute.local_tile(
                            h0_source, (1, 1, vec_size), (flat_state_idx, v_idx_d, lane_in_group)
                        )
                        cute.autovec_copy(cute.slice_(r_h, (3, None)), h_tile_out_d)

        # ============ Recurrence: ilp_rows == 2 (process 2 V-rows together) ===
        # Batch-load all T q/k + precompute all deferred L2 norms up front, then a
        # fused T-loop (decay+h@k, update+h@q). KDA change: per-channel decay
        # r_g_arr[i_t, i]. Mirrors FlashInfer gdn_decode_mtp.py:1903-2098.
        elif cutlass.const_expr(ilp_rows == 2):
            half_rows: cutlass.Constexpr[int] = rows_per_group // 2

            r_q_all = cute.make_rmem_tensor(
                cute.make_layout((T, vec_size), stride=(vec_size, 1)), cutlass.Float32
            )
            r_k_all = cute.make_rmem_tensor(
                cute.make_layout((T, vec_size), stride=(vec_size, 1)), cutlass.Float32
            )
            inv_nk_arr = cute.make_rmem_tensor(
                cute.make_layout((T,), stride=(1,)), cutlass.Float32
            )
            inv_nq_arr = cute.make_rmem_tensor(
                cute.make_layout((T,), stride=(1,)), cutlass.Float32
            )

            for row_pair in cutlass.range_constexpr(half_rows):
                v_idx_a = i_v * tile_v + group_idx * rows_per_group + row_pair * 2
                v_idx_b = v_idx_a + 1

                if v_idx_b < V:
                    h_tile_a = cute.local_tile(
                        h0_source, (1, 1, vec_size), (flat_state_idx, v_idx_a, lane_in_group)
                    )
                    h_tile_b = cute.local_tile(
                        h0_source, (1, 1, vec_size), (flat_state_idx, v_idx_b, lane_in_group)
                    )
                    cute.autovec_copy(h_tile_a, cute.slice_(r_h, (0, None)))
                    cute.autovec_copy(h_tile_b, cute.slice_(r_h, (1, None)))

                    # Batch load ALL q/k (raw) + precompute ALL deferred L2 norms.
                    for i_t in cutlass.range_constexpr(T):
                        q_tile = cute.local_tile(
                            q, (1, 1, 1, vec_size), (i_n, i_t, i_h, lane_in_group)
                        )
                        k_tile = cute.local_tile(
                            k, (1, 1, 1, vec_size), (i_n, i_t, i_h, lane_in_group)
                        )
                        cute.autovec_copy(q_tile, r_q_bf16)
                        cute.autovec_copy(k_tile, r_k_bf16)
                        for i in cutlass.range_constexpr(vec_size):
                            r_q_all[(i_t, i)] = cutlass.Float32(r_q_bf16[i])
                            r_k_all[(i_t, i)] = cutlass.Float32(r_k_bf16[i])

                        if cutlass.const_expr(use_qk_l2norm):
                            sum_sq_q = 0.0
                            sum_sq_k = 0.0
                            for i in cutlass.range_constexpr(vec_size):
                                sum_sq_q += r_q_all[(i_t, i)] * r_q_all[(i_t, i)]
                                sum_sq_k += r_k_all[(i_t, i)] * r_k_all[(i_t, i)]
                            for offset in [16, 8, 4, 2, 1]:
                                sum_sq_q += cute.arch.shuffle_sync_bfly(
                                    sum_sq_q, offset=offset, mask=-1, mask_and_clamp=31
                                )
                                sum_sq_k += cute.arch.shuffle_sync_bfly(
                                    sum_sq_k, offset=offset, mask=-1, mask_and_clamp=31
                                )
                            inv_nk_arr[i_t] = cute.rsqrt(sum_sq_k + 1e-6)
                            inv_nq_arr[i_t] = cute.rsqrt(sum_sq_q + 1e-6) * scale
                        else:
                            for i in cutlass.range_constexpr(vec_size):
                                r_q_all[(i_t, i)] = r_q_all[(i_t, i)] * scale

                    # Preload all v values (GMEM path).
                    r_va_all = cute.make_rmem_tensor(
                        cute.make_layout((T,), stride=(1,)), cutlass.Float32
                    )
                    r_vb_all = cute.make_rmem_tensor(
                        cute.make_layout((T,), stride=(1,)), cutlass.Float32
                    )
                    if cutlass.const_expr(not use_smem_v):
                        for i_t in cutlass.range_constexpr(T):
                            r_va_all[i_t] = cutlass.Float32(v[i_n, i_t, i_hv, v_idx_a])
                            r_vb_all[i_t] = cutlass.Float32(v[i_n, i_t, i_hv, v_idx_b])

                    for i_t in cutlass.range_constexpr(T):
                        r_beta = r_beta_arr[i_t]

                        # Fused per-channel decay (KDA: r_g_arr[i_t, i]) + h@k.
                        sum_hk_a = 0.0
                        sum_hk_b = 0.0
                        for i in cutlass.range_constexpr(vec_size):
                            r_h[0, i] = r_h[0, i] * r_g_arr[(i_t, i)]
                            r_h[1, i] = r_h[1, i] * r_g_arr[(i_t, i)]
                            sum_hk_a += r_h[0, i] * r_k_all[(i_t, i)]
                            sum_hk_b += r_h[1, i] * r_k_all[(i_t, i)]
                        for offset in [16, 8, 4, 2, 1]:
                            sum_hk_a += cute.arch.shuffle_sync_bfly(
                                sum_hk_a, offset=offset, mask=-1, mask_and_clamp=31
                            )
                            sum_hk_b += cute.arch.shuffle_sync_bfly(
                                sum_hk_b, offset=offset, mask=-1, mask_and_clamp=31
                            )
                        if cutlass.const_expr(use_qk_l2norm):
                            inv_nk = inv_nk_arr[i_t]
                            sum_hk_a = sum_hk_a * inv_nk
                            sum_hk_b = sum_hk_b * inv_nk

                        # v from SMEM or pre-loaded registers.
                        if cutlass.const_expr(use_smem_v):
                            v_local_a = v_idx_a - i_v * tile_v
                            r_v_a = sVdata[(i_t, v_local_a)]
                            r_v_b = sVdata[(i_t, v_local_a + 1)]
                        else:
                            r_v_a = r_va_all[i_t]
                            r_v_b = r_vb_all[i_t]
                        v_new_a = (r_v_a - sum_hk_a) * r_beta
                        v_new_b = (r_v_b - sum_hk_b) * r_beta

                        # Fused rank-1 update (deferred: ks = inv_nk * v_new) + h@q.
                        if cutlass.const_expr(use_qk_l2norm):
                            ks_a = inv_nk * v_new_a
                            ks_b = inv_nk * v_new_b
                            sum_hq_a = 0.0
                            sum_hq_b = 0.0
                            for i in cutlass.range_constexpr(vec_size):
                                r_h[0, i] += r_k_all[(i_t, i)] * ks_a
                                r_h[1, i] += r_k_all[(i_t, i)] * ks_b
                                sum_hq_a += r_h[0, i] * r_q_all[(i_t, i)]
                                sum_hq_b += r_h[1, i] * r_q_all[(i_t, i)]
                        else:
                            sum_hq_a = 0.0
                            sum_hq_b = 0.0
                            for i in cutlass.range_constexpr(vec_size):
                                r_h[0, i] += r_k_all[(i_t, i)] * v_new_a
                                r_h[1, i] += r_k_all[(i_t, i)] * v_new_b
                                sum_hq_a += r_h[0, i] * r_q_all[(i_t, i)]
                                sum_hq_b += r_h[1, i] * r_q_all[(i_t, i)]

                        # Stage D: snapshot post-token state (sequence-indexed).
                        if cutlass.const_expr(cache_intermediate_states):
                            flat_idx = i_n * T * HV + i_t * HV + i_hv
                            inter_a = cute.local_tile(
                                intermediate_states, (1, 1, vec_size), (flat_idx, v_idx_a, lane_in_group)
                            )
                            cute.autovec_copy(cute.slice_(r_h, (0, None)), inter_a)
                            inter_b = cute.local_tile(
                                intermediate_states, (1, 1, vec_size), (flat_idx, v_idx_b, lane_in_group)
                            )
                            cute.autovec_copy(cute.slice_(r_h, (1, None)), inter_b)

                        # h@q reduction + deferred q-norm (scale already folded in).
                        for offset in [16, 8, 4, 2, 1]:
                            sum_hq_a += cute.arch.shuffle_sync_bfly(
                                sum_hq_a, offset=offset, mask=-1, mask_and_clamp=31
                            )
                            sum_hq_b += cute.arch.shuffle_sync_bfly(
                                sum_hq_b, offset=offset, mask=-1, mask_and_clamp=31
                            )
                        if cutlass.const_expr(use_qk_l2norm):
                            inv_nq = inv_nq_arr[i_t]
                            sum_hq_a = sum_hq_a * inv_nq
                            sum_hq_b = sum_hq_b * inv_nq

                        # Write output (lane 0): SMEM (merged flush) or GMEM.
                        if lane_in_group == 0:
                            if cutlass.const_expr(use_smem_v):
                                vla = v_idx_a - i_v * tile_v
                                sOutput[(i_t, vla)] = cutlass.BFloat16(sum_hq_a)
                                sOutput[(i_t, vla + 1)] = cutlass.BFloat16(sum_hq_b)
                            else:
                                o[(i_n, i_t, i_hv, v_idx_a)] = cutlass.BFloat16(sum_hq_a)
                                o[(i_n, i_t, i_hv, v_idx_b)] = cutlass.BFloat16(sum_hq_b)

                    # Write final state for both rows back (once).
                    if cutlass.const_expr(not disable_state_update):
                        h_tile_out_a = cute.local_tile(
                            h0_source, (1, 1, vec_size), (flat_state_idx, v_idx_a, lane_in_group)
                        )
                        cute.autovec_copy(cute.slice_(r_h, (0, None)), h_tile_out_a)
                        h_tile_out_b = cute.local_tile(
                            h0_source, (1, 1, vec_size), (flat_state_idx, v_idx_b, lane_in_group)
                        )
                        cute.autovec_copy(cute.slice_(r_h, (1, None)), h_tile_out_b)

        # ============ Merged output writeback (use_smem_v only) ============
        # Same coalesced sOutput -> o flush as the warp-spec kernel: barrier
        # publishes all groups' lane-0 writes, then all 128 threads flush one
        # tile-local column each across all T tokens.
        if cutlass.const_expr(use_smem_v):
            cute.arch.barrier()
            v_tile_base = i_v * tile_v
            for t_idx in cutlass.range_constexpr(T):
                if tidx < tile_v:
                    v_global = v_tile_base + tidx
                    if v_global < V:
                        o[(i_n, t_idx, i_hv, v_global)] = sOutput[(t_idx, tidx)]


@cute.jit
def run_kda_verify_kernel_mtp_inline(
    h0_source: cute.Tensor,
    intermediate_states: cute.Tensor,
    A_log: cute.Tensor,
    a: cute.Tensor,
    dt_bias: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    b: cute.Tensor,
    o: cute.Tensor,
    h0_indices: cute.Tensor,
    softplus_beta: cutlass.Constexpr[float],
    softplus_threshold: cutlass.Constexpr[float],
    scale: cutlass.Constexpr[float],
    HV: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    tile_v: cutlass.Constexpr[int],
    vec_size: cutlass.Constexpr[int],
    use_qk_l2norm: cutlass.Constexpr[bool],
    disable_state_update: cutlass.Constexpr[bool],
    ilp_rows: cutlass.Constexpr[int],
    use_packed_fma: cutlass.Constexpr[bool],
    use_smem_v: cutlass.Constexpr[bool],
    cache_intermediate_states: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    """Host-side launcher (inline): grid = N * HV * num_v_tiles, block = 128."""
    n_indices = h0_indices.layout.shape[0]
    v_dim = h0_source.layout.shape[1]

    num_v_tiles = cute.ceil_div(v_dim, tile_v)
    grid_size = n_indices * HV * num_v_tiles

    # Inline: no sQ/sK/sG/sBeta broadcast buffers. Just alignment slack, plus
    # sVdata ([T, tile_v] fp32) + sOutput ([T, tile_v] bf16) when use_smem_v
    # (matches the kernel's conditional allocation).
    smem_bytes = 128  # alignment slack
    if cutlass.const_expr(use_smem_v):
        smem_bytes += 4 * T * tile_v  # sVdata (fp32)
        smem_bytes += 2 * T * tile_v  # sOutput (bf16)

    kda_verify_kernel_mtp_inline(
        h0_source,
        intermediate_states,
        vec_size,
        num_v_tiles,
        tile_v,
        A_log,
        a,
        dt_bias,
        q,
        k,
        v,
        b,
        o,
        h0_indices,
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
        ilp_rows,
        use_packed_fma,
        use_smem_v,
        cache_intermediate_states,
    ).launch(
        grid=(grid_size, 1, 1),
        block=[NUM_THREADS, 1, 1],
        smem=smem_bytes,
        stream=stream,
    )


def _get_compiled_mtp_inline_kernel(
    N,
    T,
    H,
    HV,
    K,
    V,
    pool_size,
    scale,
    use_qk_l2norm,
    disable_state_update,
    softplus_beta,
    softplus_threshold,
    tile_v,
    ilp_rows,
    use_packed_fma,
    use_smem_v,
    cache_intermediate_states,
):
    """Get or lazily compile the inline MTP kernel for one shape/config."""
    key = (
        N,
        T,
        H,
        HV,
        K,
        V,
        pool_size,
        scale,
        use_qk_l2norm,
        disable_state_update,
        softplus_beta,
        softplus_threshold,
        tile_v,
        ilp_rows,
        use_packed_fma,
        use_smem_v,
        cache_intermediate_states,
    )
    if key in _compiled_mtp_inline_kernels:
        return _compiled_mtp_inline_kernels[key]

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
    if cache_intermediate_states:
        intermediate_states = torch.zeros(
            N * T * HV, V, K, dtype=torch.float32, device="cuda"
        )
    else:
        intermediate_states = torch.zeros(1, 1, 1, dtype=torch.float32, device="cuda")

    q_tensor = from_dlpack(q, assumed_align=16)
    k_tensor = from_dlpack(k, assumed_align=16)
    v_tensor = from_dlpack(v, assumed_align=16)
    a_tensor = from_dlpack(a, assumed_align=16)
    b_tensor = from_dlpack(b, assumed_align=16)
    A_log_tensor = from_dlpack(A_log, assumed_align=16)
    dt_bias_tensor = from_dlpack(dt_bias, assumed_align=16)
    h0_source_tensor = from_dlpack(h0_source, assumed_align=16)
    h0_indices_tensor = from_dlpack(h0_indices, assumed_align=16)
    o_tensor = from_dlpack(o, assumed_align=16)
    intermediate_states_tensor = from_dlpack(intermediate_states, assumed_align=16)

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    compiled_kernel = cute.compile(
        run_kda_verify_kernel_mtp_inline,
        h0_source_tensor,
        intermediate_states_tensor,
        A_log_tensor,
        a_tensor,
        dt_bias_tensor,
        q_tensor,
        k_tensor,
        v_tensor,
        b_tensor,
        o_tensor,
        h0_indices_tensor,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        scale=scale,
        HV=HV,
        T=T,
        H=H,
        K=K,
        V=V,
        tile_v=tile_v,
        vec_size=VEC_SIZE_MTP,
        use_qk_l2norm=use_qk_l2norm,
        disable_state_update=disable_state_update,
        ilp_rows=ilp_rows,
        use_packed_fma=use_packed_fma,
        use_smem_v=use_smem_v,
        cache_intermediate_states=cache_intermediate_states,
        stream=stream,
        options="--enable-tvm-ffi --opt-level 1",
    )

    _compiled_mtp_inline_kernels[key] = compiled_kernel
    logger.info(
        "CuTe DSL KDA MTP inline kernel compiled: "
        f"N={N}, T={T}, H={H}, HV={HV}, K={K}, V={V}, pool_size={pool_size}, "
        f"tile_v={tile_v}, ilp_rows={ilp_rows}, use_packed_fma={use_packed_fma}, "
        f"use_smem_v={use_smem_v}, cache_intermediate_states={cache_intermediate_states}"
    )
    return compiled_kernel


def kda_decode_mtp_ws_inline(
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
    tile_v: int | None = None,
    ilp_rows: int | None = None,
    disable_state_update: bool = False,
    use_packed_fma: bool | None = None,
    use_smem_v: bool | None = None,
    intermediate_states_buffer: torch.Tensor | None = None,
) -> torch.Tensor:
    """KDA MTP decode — INLINE variant (Route 1, small-batch path).

    Same public contract and arguments as :func:`kda_decode_mtp_ws`, but runs the
    inline kernel (no warp-0 staging / Phase-1 barrier, deferred L2 norm,
    register-resident q/k/g/beta). FlashInfer routes here at small ``work_units =
    N*HV`` (<= 128); this entry point is currently standalone (explicit call) for
    head-to-head validation — the auto inline-vs-warp-spec dispatch is wired in a
    later step. See :func:`kda_decode_mtp_ws` for the full argument semantics
    (``ilp_rows``/``tile_v``/``use_smem_v`` resolution from the heuristic,
    ``intermediate_states_buffer`` snapshots, vk-only state layout).

    The inline ilp=4 path uses scalar FMA (no packed-FMA / double-accumulator), so
    ``use_packed_fma`` does not affect numerics here; it is accepted only to keep
    the signature identical to the warp-spec entry point.
    """
    N, T, H, K = q.shape
    HV = v.shape[2]
    V = v.shape[3]

    if scale is None:
        scale = K**-0.5
    else:
        assert scale > 0, f"scale must be positive, got {scale}"

    assert K == TILE_K, f"KDA MTP (inline) kernel requires K={TILE_K}, got {K}"

    # Resolve tile_v / ilp_rows / use_smem_v from the work_units=N*HV heuristic
    # where not given explicitly (identical handling to kda_decode_mtp_ws).
    if tile_v is None or ilp_rows is None or use_smem_v is None:
        sel_tile_v, sel_ilp_rows, sel_use_smem_v = _select_mtp_config(
            N, HV, V, T, disable_state_update=disable_state_update
        )
        if tile_v is None:
            tile_v = sel_tile_v
        if ilp_rows is None:
            ilp_rows = sel_ilp_rows
            if ilp_rows == 4 and tile_v % 16 != 0:
                ilp_rows = 2
        if use_smem_v is None:
            use_smem_v = sel_use_smem_v

    if ilp_rows not in (2, 4):
        raise NotImplementedError(
            f"kda_decode_mtp_ws_inline implements ilp_rows in {{2, 4}}, got {ilp_rows}"
        )

    # The inline kernel is scalar-FMA only; use_packed_fma is inert. Resolve it
    # the same way for signature parity but it never reaches a packed instruction.
    if use_packed_fma is None:
        major, _ = torch.cuda.get_device_capability(q.device)
        use_packed_fma = major >= 10
    if ilp_rows != 4:
        use_packed_fma = False

    state_layout = _canonicalize_state_layout(state_layout)
    if state_layout != "vk":
        raise NotImplementedError(
            "kda_decode_mtp_ws_inline only supports state_layout='vk' "
            f"(FlashInfer is vk-only); got {state_layout!r}"
        )

    assert tile_v % 4 == 0, f"KDA MTP (inline) requires tile_v % 4 == 0, got tile_v={tile_v}"
    assert V % tile_v == 0, f"KDA MTP (inline) requires V % tile_v == 0, got V={V}, tile_v={tile_v}"
    rows_per_group = tile_v // 4
    assert rows_per_group % ilp_rows == 0, (
        f"ilp_rows={ilp_rows} requires (tile_v//4) divisible by {ilp_rows}, "
        f"got tile_v={tile_v} (tile_v//4={rows_per_group})"
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
    assert not state_layout_is_kv  # guaranteed by the vk-only guard above

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

    cache_intermediate_states = intermediate_states_buffer is not None
    if cache_intermediate_states:
        if intermediate_states_buffer.dtype != torch.float32:
            raise ValueError(
                "intermediate_states_buffer must be float32, got "
                f"{intermediate_states_buffer.dtype}"
            )
        expected_buf_shape = (N, T, HV, V, K)
        if tuple(intermediate_states_buffer.shape) != expected_buf_shape:
            raise ValueError(
                f"intermediate_states_buffer shape {tuple(intermediate_states_buffer.shape)} "
                f"!= expected {expected_buf_shape} ([N, T, HV, V, K] vk / K-last)"
            )
        intermediate_states_flat = intermediate_states_buffer.view(N * T * HV, V, K)
    else:
        intermediate_states_flat = torch.zeros(
            1, 1, 1, dtype=torch.float32, device=q.device
        )

    stream = _get_cached_stream(q.device)
    compiled_kernel = _get_compiled_mtp_inline_kernel(
        N,
        T,
        H,
        HV,
        K,
        V,
        pool_size,
        scale=scale,
        use_qk_l2norm=use_qk_l2norm_in_kernel,
        disable_state_update=disable_state_update,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        tile_v=tile_v,
        ilp_rows=ilp_rows,
        use_packed_fma=use_packed_fma,
        use_smem_v=use_smem_v,
        cache_intermediate_states=cache_intermediate_states,
    )

    compiled_kernel(
        h0_source_flat,
        intermediate_states_flat,
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
