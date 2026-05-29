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
  ``tile_v % 16 == 0`` (so {16,32,64}). ilp=8 + use_smem_v come next.
- ``vk`` state layout only (FlashInfer is vk-only; kv is a later add-back).
- No ``use_smem_v`` / ``sOutput`` merge, no intermediate-state snapshots.
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
):
    """Warp-specialized KDA MTP kernel. One CTA owns one (i_n, i_hv, i_v) tile.

    Phase 1: warp 0 computes q/k (L2-normed) + per-channel g + scalar beta for
    all T tokens and writes them to SMEM; warps 1-3 prefetch the first ILP set
    of state rows from GMEM into registers. A barrier publishes the SMEM. Then
    all 4 warps run the T-step recurrence with state register-resident, one CTA
    covering ``tile_v`` V-rows (4 warps x rows_per_group), each lane owning
    ``vec_size`` K-channels of a V-row and reducing over K via full-warp shuffle.
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

        # Publish warp 0's SMEM writes to all warps before the recurrence reads.
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

                        # Step 3: delta rule.
                        r_v_a = cutlass.Float32(v[i_n, i_t, i_hv, v_idx_a])
                        r_v_b = cutlass.Float32(v[i_n, i_t, i_hv, v_idx_b])
                        v_new_a = (r_v_a - sum_hk_a) * r_beta
                        v_new_b = (r_v_b - sum_hk_b) * r_beta

                        # Step 4: rank-1 update with raw k (decay already applied).
                        for i in cutlass.range_constexpr(vec_size):
                            r_h[0, i] += r_k[i] * v_new_a
                            r_h[1, i] += r_k[i] * v_new_b

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

                        # Reduction result is identical on all lanes -> lane 0 writes.
                        if lane_in_group == 0:
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
        # GDN. use_smem_v / sOutput / intermediate-state snapshots are later stages
        # and intentionally omitted (GMEM v read + direct o write only).
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

                        # Step 3: delta rule for all 4 rows (GMEM v read).
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

                        # Reduction result is identical on all lanes -> lane 0 writes.
                        if lane_in_group == 0:
                            o[(i_n, i_t, i_hv, v_idx_a)] = cutlass.BFloat16(sum_hq_a)
                            o[(i_n, i_t, i_hv, v_idx_b)] = cutlass.BFloat16(sum_hq_b)
                            o[(i_n, i_t, i_hv, v_idx_c)] = cutlass.BFloat16(sum_hq_c)
                            o[(i_n, i_t, i_hv, v_idx_d)] = cutlass.BFloat16(sum_hq_d)

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


@cute.jit
def run_kda_verify_kernel_mtp_ws(
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
    stream: cuda.CUstream,
):
    """Host-side launcher: grid = N * HV * num_v_tiles, block = 128 (4 warps)."""
    n_indices = h0_indices.layout.shape[0]
    v_dim = h0_source.layout.shape[1]
    k_dim = h0_source.layout.shape[2]

    num_v_tiles = cute.ceil_div(v_dim, tile_v)
    grid_size = n_indices * HV * num_v_tiles

    # sQ + sK + sG (all [T, K+8] fp32) + sBeta ([T] fp32) + alignment slack.
    smem_bytes = (
        4 * T * (k_dim + 8)  # sQ
        + 4 * T * (k_dim + 8)  # sK
        + 4 * T * (k_dim + 8)  # sG (KDA: per-channel; GDN had 4*T scalar)
        + 4 * T  # sBeta
        + 128  # alignment slack
    )

    kda_verify_kernel_mtp_ws(
        h0_source,
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

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    compiled_kernel = cute.compile(
        run_kda_verify_kernel_mtp_ws,
        h0_source_tensor,
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
        stream=stream,
        options="--enable-tvm-ffi --opt-level 1",
    )

    _compiled_mtp_ws_kernels[key] = compiled_kernel
    logger.info(
        "CuTe DSL KDA MTP warp-spec kernel compiled: "
        f"N={N}, T={T}, H={H}, HV={HV}, K={K}, V={V}, pool_size={pool_size}, "
        f"tile_v={tile_v}, ilp_rows={ilp_rows}, use_packed_fma={use_packed_fma}"
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

    # Resolve tile_v / ilp_rows from the work_units=N*HV heuristic where not
    # given explicitly (mirrors kda_decode_mtp). The heuristic's use_smem_v is
    # produced but ignored here (Stage C). An explicit tile_v can make the
    # heuristic's ilp=4 illegal (needs tile_v % 16 == 0); in the auto path we
    # fall back to the universally-legal ilp=2 rather than tripping the
    # rows_per_group assert below. (When tile_v also came from the heuristic,
    # _select_mtp_config already applied this backstop, so the guard is a no-op.)
    if tile_v is None or ilp_rows is None:
        sel_tile_v, sel_ilp_rows, _sel_use_smem_v = _select_mtp_config(
            N, HV, V, T, disable_state_update=disable_state_update
        )
        if tile_v is None:
            tile_v = sel_tile_v
        if ilp_rows is None:
            ilp_rows = sel_ilp_rows
            if ilp_rows == 4 and tile_v % 16 != 0:
                ilp_rows = 2

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
