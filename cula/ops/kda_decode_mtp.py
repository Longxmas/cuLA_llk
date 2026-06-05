"""CuTe DSL KDA MTP decode

Production KDA MTP decode kernel. Public entry point: ``kda_decode_mtp_ws``
(warp-spec). The defining feature is KDA's per-K-channel decay gate ``g_t in R^K``
(``beta`` stays a per-(head, token) scalar); the whole kernel is built around that
channel axis.

Grid = N*HV*num_v_tiles, one CTA per (i_n, i_hv, i_v V-tile). State is
register-resident across the T tokens; the K-reduce is a full-warp shuffle. The
recurrence uses the DECAY-FIRST order (decay the whole state, then dot with raw k);
bf16 rounding differs slightly from the single-token ``kda_decode`` (accumulation
order), both validated against the fp32 torch oracle at atol 3e-2 / rtol 2e-2.

Scope (this file):
- Warp-spec variant. ``ilp_rows in {2, 4}``: ilp=2 covers every
  tile_v in {8,16,32,64}; ilp=4 fuses steps 1+2 and 4+5 with double accumulators +
  packed F32x2 FMA on SM100 (scalar ``fma_pair`` fallback elsewhere) and requires
  ``tile_v % 16 == 0`` (so {16,32,64}).
- ``vk`` state layout only.
- ``use_smem_v`` (Stage C): preload the v-tile into SMEM + coalesced merged output
  writeback. Constexpr, off unless the heuristic / an explicit arg turns it on.
- ``cache_intermediate_states`` (Stage D): when an ``intermediate_states_buffer``
  ([N, T, HV, V, K] vk) is passed, snapshot every token's post-state to GMEM
  (sequence-indexed) for speculative-decoding rollback. Produce-only.
- ``disable_state_update`` supported (default False = always write back).

Math per token t (decay-first, per-channel g):
    g_t   = exp(-exp(A_log) * softplus(a_t + dt_bias))       # (K,) per-channel
    S    <- S * diag(g_t)                                     # step 1 (per channel)
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

logger = logging.getLogger(__name__)

# vec_size = 4 -> 32 threads/group = a full warp, 4 groups (warps) per block.
# Full-warp shuffle reduction over K is independent of tile_v, so the kernel body
# is NOT specialized per tile_v (tile_v stays a plain constexpr, not a body shape).
VEC_SIZE_MTP = 4

# Warp-spec MTP kernels are compiled once per shape/config (including T, tile_v,
# ilp_rows, use_packed_fma) and cached.
_compiled_mtp_ws_kernels: dict[tuple, object] = {}


# ---------------------------------------------------------------------------
# Host-side config helpers (pure-Python; shared by the ws entry point,
# the benchmark, and the tests).
# ---------------------------------------------------------------------------
def _normalize_mtp_a(a: torch.Tensor, *, N: int, T: int, HV: int, K: int) -> torch.Tensor:
    """Normalize `a` to the compile-time dense MTP shape (N, T, HV, K)."""
    if a.dim() == 4 and tuple(a.shape) == (N, T, HV, K):
        return a
    if a.dim() == 3 and tuple(a.shape) == (N, T, HV * K):
        return a.view(N, T, HV, K)
    raise ValueError(f"Unexpected a shape for MTP dense: {tuple(a.shape)}; expected {(N, T, HV, K)}")


# Valid V-tile sizes {8,16,32,64}: each a multiple of NUM_WARPS (4) so V_PER_WARP
# is integral; the heuristic picks from these.
_MTP_TILE_V_CHOICES = (8, 16, 32, 64)


def _select_mtp_config(
    N: int,
    HV: int,
    V: int,
    T: int,
    *,
    disable_state_update: bool = False,
) -> tuple[int, int, bool]:
    work_units = N * HV

    if work_units <= 64:
        tile_v, ilp_rows, use_smem_v = 8, 2, False
    elif work_units <= 128:
        tile_v, ilp_rows, use_smem_v = 16, 4, False
    elif work_units <= 448:
        if T <= 2:
            tile_v, ilp_rows, use_smem_v = 16, 2, False
        else:
            tile_v, ilp_rows, use_smem_v = 32, 4, False
    elif work_units <= 1024:
        tile_v, ilp_rows, use_smem_v = 32, 4, False
    else:
        # Large batches: ilp capped at 4, so (64, 4, True) uniformly.
        tile_v, ilp_rows, use_smem_v = 64, 4, True

    # Clamp to V and back off to a divisor of V (V is a multiple of 16 in
    # practice, so this is a no-op for the common V=128 case).
    tile_v = min(tile_v, V)
    while tile_v > _MTP_TILE_V_CHOICES[0] and V % tile_v != 0:
        tile_v //= 2

    # Legality backstop: ilp=4 requires (tile_v//4) % 4 == 0, i.e. tile_v % 16 == 0
    if ilp_rows == 4 and tile_v % 16 != 0:
        ilp_rows = 2

    return tile_v, ilp_rows, use_smem_v


def _select_mtp_tile_v(N: int, HV: int, V: int, T: int) -> int:
    # Pick tile_v from work_units = N*HV (the tile_v axis of _select_mtp_config).

    return _select_mtp_config(N, HV, V, T)[0]


@cute.jit
def fma_pair(a1, a2, b1, b2, c1, c2):
    # FMA two pairs: (a1*b1+c1, a2*b2+c2).
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
    A_log: cute.Tensor,  # [HV] fp32 (per-channel decay)
    a: cute.Tensor,  # [N, T, HV, K] (per-channel decay input)
    dt_bias: cute.Tensor,  # [HV, K] (per-channel decay bias)
    q: cute.Tensor,  # [N, T, H, K]
    k: cute.Tensor,  # [N, T, H, K]
    v: cute.Tensor,  # [N, T, HV, V]
    b: cute.Tensor,  # [N, T, HV] (update-gate logit)
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
    fast_math: cutlass.Constexpr[bool],
):
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

    # exp(A_log) is per-head, shared across all K channels — hoist once.
    r_A_log = cutlass.Float32(A_log[i_hv])
    r_exp_A = cute.exp(r_A_log, fastmath=fast_math)

    # SMEM broadcast buffers (warp 0 -> all warps). sG is [T, K] (per-channel);
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

    # use_smem_v (Stage C): preload the v-tile into SMEM + accumulate outputs for a
    # coalesced merged writeback. Allocated last/conditionally so off-path offsets stay put.
    if cutlass.const_expr(use_smem_v):
        sVdata = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((T, tile_v), stride=(tile_v, 1)), 16
        )
        sOutput = smem.allocate_tensor(
            cutlass.BFloat16, cute.make_layout((T, tile_v), stride=(tile_v, 1)), 16
        )

    # Per-lane registers: r_g = this lane's vec_size channels of g; r_h = up to 8
    # V-rows of state (only ilp_rows used), each row spanning 32 lanes over K=128.
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
                    inv_norm_q_scaled = cute.rsqrt(sum_q + 1e-6, fastmath=fast_math) * scale
                    inv_norm_k = cute.rsqrt(sum_k + 1e-6, fastmath=fast_math)
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
                    exp_beta_x = cute.exp(beta_x, fastmath=fast_math)
                    softplus_val = (cutlass.Float32(1.0) / softplus_beta) * cute.log(
                        cutlass.Float32(1.0) + exp_beta_x, fastmath=fast_math
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
                    sG[(i_t, kk)] = cute.exp(-r_exp_A * softplus_x, fastmath=fast_math)

                # Update gate beta is a per-(head, token) scalar (warp-uniform).
                r_b = cutlass.Float32(b[i_n, i_t, i_hv])
                r_beta = cutlass.Float32(1.0) / (
                    cutlass.Float32(1.0) + cute.exp(-r_b, fastmath=fast_math)
                )
                sBeta[i_t] = r_beta

                # Preload the v-tile into SMEM: warp 0 covers tile-local cols 0..31,
                # warps 1-3 the rest (tidx<tile_v guard -> each col written once).
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

            # Warps 1-3 cover the tile-local v columns warp 0 can't reach
            # (tidx 32..127); same tidx<tile_v guard -> each column written once.
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

                        # Stage D: snapshot post-token state, sequence-indexed
                        # (flat_idx = i_n*T*HV + i_t*HV + i_hv), race-free before step 5.
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
        # Steps 1+2 fused (decay then h@k) and 4+5 fused (rank-1 then h@q), with
        # double accumulators (halve the K-reduce FFMA chain) + packed F32x2 FMA on
        # SM100. Per-channel decay r_g[i]/r_g[i+1] loaded from sG.
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

                        # Steps 1+2 fused: per-channel decay then h@k. Two accumulators
                        # (a/a2) split even/odd K so the FFMA chain is half-length.
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

                        # Stage D: snapshot post-token state (sequence-indexed),
                        # last here since fused 4+5 means r_h is final only now.
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
        # Barrier publishes all groups' disjoint lane-0 sOutput writes, then all 128
        # threads flush sOutput -> o (one tile-local column each, all T tokens) so the
        # GMEM writes coalesce. Inside `cache_idx >= 0` so the barrier never deadlocks.
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
    fast_math: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    """Host-side launcher: grid = N * HV * num_v_tiles, block = 128 (4 warps)."""
    n_indices = h0_indices.layout.shape[0]
    v_dim = h0_source.layout.shape[1]
    k_dim = h0_source.layout.shape[2]

    num_v_tiles = cute.ceil_div(v_dim, tile_v)
    grid_size = n_indices * HV * num_v_tiles

    smem_bytes = (
        4 * T * (k_dim + 8)  # sQ
        + 4 * T * (k_dim + 8)  # sK
        + 4 * T * (k_dim + 8)  # sG (per-channel)
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
        fast_math,
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
    opt_level=3,
    fast_math=True,
):
    """Get or lazily compile the warp-spec MTP kernel for one shape/config.

    ``opt_level`` (``--opt-level``) and ``fast_math`` are part of the cache key.
    """
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
        opt_level,
        fast_math,
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
        fast_math=fast_math,
        stream=stream,
        options=f"--enable-tvm-ffi --opt-level {opt_level}",
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
    N, T, H, K = q.shape
    HV = v.shape[2]
    V = v.shape[3]

    if scale is None:
        scale = K**-0.5
    else:
        assert scale > 0, f"scale must be positive, got {scale}"

    assert K == TILE_K, f"KDA MTP (ws) kernel requires K={TILE_K}, got {K}"

    # Resolve tile_v / ilp_rows / use_smem_v from the work_units=N*HV heuristic
    # where not given explicitly. An explicit tile_v can make the heuristic's ilp=4
    # illegal (needs tile_v % 16 == 0); the auto path then falls back to ilp=2.
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

    # packed F32x2 FMA exists only on SM100+ (Blackwell); scalar fma_pair
    # elsewhere. None = auto-detect by compute capability.
    if use_packed_fma is None:
        major, _ = torch.cuda.get_device_capability(q.device)
        use_packed_fma = major >= 10
    # The packed path only exists in the ilp=4 kernel branch; ilp=2 is scalar.
    if ilp_rows != 4:
        use_packed_fma = False

    state_layout = _canonicalize_state_layout(state_layout)
    if state_layout != "vk":
        raise NotImplementedError(
            "kda_decode_mtp_ws only supports state_layout='vk'; "
            f"got {state_layout!r}"
        )

    assert tile_v % 4 == 0, f"KDA MTP (ws) requires tile_v % 4 == 0, got tile_v={tile_v}"
    assert V % tile_v == 0, f"KDA MTP (ws) requires V % tile_v == 0, got V={V}, tile_v={tile_v}"

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

    # Flatten the VK state pool [pool, HV, V, K] -> [pool*HV, V, K]
    h0_source_flat = h0_source.view(pool_size * HV, V, K)

    # Stage D: resolve the snapshot cache. 
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
# small_batch 算子(1-warp/program):kv 布局(lane=V)+ vk 布局(lane=K)
# 从 kda_decode_mtp_small_batch.py 合并。公开函数 kda_decode_mtp_small_batch(variant=)。
# _normalize_mtp_a 用本文件本地定义(已在上方)。
# ============================================================================


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
    disable_state_update: bool = False,
    variant: str = "kv",
    bv: int = WARP_BV,
    k_split: int = 1,
    opt_level: int = 3,
    fast_math: bool = True,
) -> torch.Tensor:
    """KDA MTP decode(KDA/topk=1/decode-only),1 warp/program。布局由 variant 决定:
    variant='kv':lane=V + kv 布局(V 连续 coalesced)+ 线程内零-shuffle reduce;k_split 可调(1/2/4 或 <=0 auto)。
    variant='vk':lane=K + vk 布局 + float4 连续块 load + 2-stage 软件流水;bv 可调(8/16/32 或 <=0 auto,k_split 忽略)。"""
    assert variant in ("kv", "vk"), f"variant 仅支持 'kv'/'vk',got {variant!r}"
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

    # variant 决定 state 布局与 config 旋钮(kv→k_split 可调/bv 固定;vk→bv 可调/不 split)。
    if variant == "kv":
        state_layout = "kv"
        assert bv == WARP_BV, f"small_batch(kv) 固定 1 warp,bv 必须为 {WARP_BV},got {bv}"
        if k_split <= 0:  # auto:按 work_units(N*HV)的 wave 适配挑 k_split
            num_sms = torch.cuda.get_device_properties(q.device).multi_processor_count
            k_split = _select_k_split(N * HV, V, num_sms)
        assert k_split in (1, 2, 4), f"k_split 仅支持 1/2/4 或 <=0(auto),got {k_split}"
        assert bv % k_split == 0 and K % k_split == 0, (
            f"需 bv%k_split==0 且 K%k_split==0,got bv={bv}, K={K}, k_split={k_split}"
        )
        vcols = bv // k_split  # 每 program 的 V 列数(= kernel 内 BV)
        assert V % vcols == 0, f"small_batch(kv) requires V % (bv//k_split) == 0, got V={V}, vcols={vcols}"
    else:  # vk
        state_layout = "vk"
        if bv <= 0:  # auto:按 work_units(N*HV)的 wave 占用挑 BV(小批降 BV 填 grid)
            num_sms = torch.cuda.get_device_properties(q.device).multi_processor_count
            bv = _select_vk_bv(N * HV, V, num_sms)
        assert bv in (8, 16, 32), f"vk bv 仅支持 8/16/32 或 <=0(auto),got {bv}"
        assert V % bv == 0, f"vk requires V % bv == 0, got V={V}, bv={bv}"

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

    stream = _get_cached_stream(q.device)

    # variant 决定 state view 形状与编译 kernel。
    if variant == "kv":
        h0_source_flat = h0_source.view(pool_size * HV, K, V)  # kv
        compiled_kernel = _get_compiled_mtp_small_batch_kernel(
            N, T, H, HV, K, V, pool_size, vcols, k_split,
            scale=scale, use_qk_l2norm=use_qk_l2norm_in_kernel,
            disable_state_update=disable_state_update,
            softplus_beta=softplus_beta, softplus_threshold=softplus_threshold,
            opt_level=opt_level, fast_math=fast_math,
        )
    else:  # vk
        h0_source_flat = h0_source.view(pool_size * HV, V, K)  # vk
        compiled_kernel = _get_compiled_mtp_vk_kernel(
            N, T, H, HV, K, V, pool_size, bv,
            scale=scale, use_qk_l2norm=use_qk_l2norm_in_kernel,
            disable_state_update=disable_state_update,
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
        stream,
    )

    return o


# ============================================================================
# vk:lane=K + warp-shuffle reduce —— lane t 沿 K 连续块映射(仅 vk、单 warp、不 split)。
# lane t 沿 K 连续块持 vec_size 个 K × 全 BV 个 V 列(float4 load,coalesced+向量化);
# reduce-over-K = 线程内 vec_size + 32-lane 5 步 butterfly;o 用 32 lane 同址幂等写落盘。
# 数学与 kv 布局变体完全一致(对齐 fp32 torch 参考)。
# ============================================================================
@cute.kernel
def kda_mtp_small_batch_vk_kernel(
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
def run_kda_mtp_small_batch_vk_kernel(
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
    """lane=K vk launcher:grid = N*HV*(V//BV),block = 32(1 warp)。无 SMEM。"""
    n_indices = h0_indices.layout.shape[0]
    num_v_tiles = cute.ceil_div(V, BV)
    grid_size = n_indices * HV * num_v_tiles

    kda_mtp_small_batch_vk_kernel(
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


_compiled_mtp_vk_kernels: dict[tuple, object] = {}


def _get_compiled_mtp_vk_kernel(
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
    if key in _compiled_mtp_vk_kernels:
        return _compiled_mtp_vk_kernels[key]

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
        run_kda_mtp_small_batch_vk_kernel,
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

    _compiled_mtp_vk_kernels[key] = compiled_kernel
    logger.info(
        "CuTe DSL KDA MTP small-batch VK(lane=K) kernel compiled: "
        f"N={N}, T={T}, H={H}, HV={HV}, K={K}, V={V}, pool_size={pool_size}, BV={BV}, "
        f"opt_level={opt_level}, fast_math={fast_math}"
    )
    return compiled_kernel


# B200 ncu:vk BV=32 时 168 reg → Block Limit Registers ≈12。
def _select_vk_bv(work_units, V, num_sms):
    """vk split 轴是 V(BV)= 每 program 处理的 V 列数:降 BV → 寄存器↓ occupancy↑ grid↑,小批填 wave、藏 shuffle 延迟。
    B200 实测(N≤16,T≤8)BV=8 全程碾压 BV=32(0.92–1.15x vs 0.64–0.99x);N≥32 未测,保守回 BV=32。"""
    waves32 = work_units * (V // 32) / (num_sms * 12)  # BV=32 的波数(Block Limit Reg≈12)
    # waves32: N1=0.14 N2=0.28 N4=0.56 N8=1.12 N16=2.25 N32=4.5 → 阈值 3.0 覆盖 N≤16。
    if V % 8 == 0 and waves32 < 3.0:
        return 8
    return 32


# ============================================================================
# 自适应 dispatch 入口:按 work_units 在 small_batch(小批)与 ws(大批)间选择
# ============================================================================
def kda_decode_mtp(
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
) -> torch.Tensor:
    """KDA MTP decode 自适应 dispatch 入口(KDA/topk=1/decode-only)。

    state_layout='kv' → small_batch kv kernel(lane=V / kv 布局)。
    state_layout='vk'(默认): work_units = N*HV <= 512 (8*64) → small_batch vk kernel,
    否则 → ws warp-spec kernel。variant 专属 config(tile_v/ilp/bv/k_split)由各底层算子
    按 work_units 自动选(目前用简单阈值,细 config 后续评测再调)。
    """
    common = dict(
        A_log=A_log, dt_bias=dt_bias, q=q, k=k, v=v, a=a, b=b,
        initial_state_source=initial_state_source,
        initial_state_indices=initial_state_indices,
        scale=scale, use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        softplus_beta=softplus_beta, softplus_threshold=softplus_threshold,
        out=out, disable_state_update=disable_state_update,
    )
    if state_layout == "kv":
        return kda_decode_mtp_small_batch(**common, variant="kv")
    work_units = q.shape[0] * v.shape[2]  # N * HV
    if work_units <= 512:
        return kda_decode_mtp_small_batch(**common, variant="vk")
    return kda_decode_mtp_ws(**common, state_layout="vk")
