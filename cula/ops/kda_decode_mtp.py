"""CuTe DSL KDA MTP (Multi-Token Prediction) decode kernel — issue #17.

This extends the single-token small-batch KDA decode kernel
(`cula/ops/kda_decode.py::kda_kernel_small_batch`) to process T > 1 tokens per
(batch, value-head) in one launch, threading the recurrent state across the T
tokens. It is the "verify" primitive for speculative / multi-token decoding.

Scope (P1+P2, correctness-first):
- Dense layout only:  q/k (N, T, H, K), v/a (N, T, HV, V/K), b (N, T, HV).
- Single parameterized CTA organization (one CTA per (token, q-head), loops its
  V-heads and V-tiles). Covers small AND mid/large batch via a configurable
  tile_v and a work_units=N*HV heuristic (flashinfer-style), instead of a
  separate small/large kernel split.
- State kept in shared memory across the T tokens (register-resident state is a
  later optimization). State written back once after all T tokens.
- VK and KV state layouts. Optional Q/K L2 norm. Softplus gate + sigmoid beta.

Deferred (see ISSUE_17_PLAN.md):
- P3: precompute per-token q/k/gate once (instead of per V-tile), register
  state, ILP rows, smem-v, inline-vs-warp-specialized variants, SM100
  packed-FMA, full get_mtp_config (ilp/smem_v) tuning.
- P4: varlen, intermediate-state snapshots for rollback, disable_state_update.

The math per token t mirrors kda_decode exactly (channel-wise decay):
    gate  = exp(-exp(A_log) * softplus(a_t + dt_bias))
    v_new = sigmoid(b_t) * (v_t - H @ (gate * k_norm))
    H     = diag(gate) @ H + v_new @ k_norm^T          # carried to t+1
    o_t   = H @ (l2norm(q_t) * scale)
"""

import logging

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

from cula.ops.kda_decode import (
    NUM_STAGES,
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

# MTP kernels are compiled once per shape/config (including T) and cached.
_compiled_mtp_kernels: dict[tuple, object] = {}


def _define_mtp_kernels(tile_v: int):
    """Define the CuTe DSL MTP decode kernel specialized for a given V tile size.

    tile_v is a P2 config parameter: the kernel is specialized per tile_v and
    cached. The thread/iteration geometry below assumes 4 warps (128 threads)
    and TILE_K=128, matching the single-token small-batch kernel. With those
    fixed, NUM_K_ITERS_SMALL == tile_v and the state load/store covers exactly
    TILE_K * tile_v elements, so the only tile_v-dependent control flow is the
    butterfly reduction width.
    """

    NUM_WARPS_SMALL = 4
    TILE_V = tile_v
    TILE_V_PADDED = tile_v + 4
    V_PER_WARP_SMALL = TILE_V // NUM_WARPS_SMALL
    ROWS_PER_ITER_SMALL = 32 // V_PER_WARP_SMALL
    NUM_K_ITERS_SMALL = TILE_K // ROWS_PER_ITER_SMALL
    # Butterfly all-reduce offsets (in k_local units) to sum the K partials over
    # the ROWS_PER_ITER lanes that share a v_local. Generalizes the hardcoded
    # [4, 2, 1] (ROWS_PER_ITER=8 for tile_v=16) to any power-of-two ROWS_PER_ITER.
    K_REDUCE_OFFSETS = []
    _red = ROWS_PER_ITER_SMALL // 2
    while _red >= 1:
        K_REDUCE_OFFSETS.append(_red)
        _red //= 2

    @cute.kernel
    def kda_kernel_small_batch_mtp(
        h0_source: cute.Tensor,
        smem_layout_staged: cute.Layout,
        num_v_tiles: cutlass.Constexpr[int],
        num_blocks_per_state_small: cutlass.Constexpr[int],
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        a: cute.Tensor,
        b: cute.Tensor,
        A_log: cute.Tensor,
        dt_bias: cute.Tensor,
        o: cute.Tensor,
        h0_indices: cute.Tensor,
        softplus_beta: cutlass.Constexpr[float],
        softplus_threshold: cutlass.Constexpr[float],
        scale: cutlass.Constexpr[float],
        T: cutlass.Constexpr[int],
        H: cutlass.Constexpr[int],
        HV: cutlass.Constexpr[int],
        use_qk_l2norm: cutlass.Constexpr[bool],
        state_layout_is_kv: cutlass.Constexpr[bool],
    ):
        """Small-batch dense KDA MTP kernel for q/k/v shaped as (N, T, ...).

        Each CTA owns one (token-batch, q-head) pair and loops over its
        value-heads and V tiles. For each V tile it loads the recurrent state
        once, runs the delta-rule recurrence over all T tokens (updating the
        state in shared memory and writing one output per token), then writes
        the final state back once.
        """
        tidx, _, _ = cute.arch.thread_idx()
        in_warp_tid = tidx % 32
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        block_idx, _, _ = cute.arch.block_idx()

        batch_idx = block_idx // num_blocks_per_state_small
        batch_inner = block_idx % num_blocks_per_state_small
        num_v_tiles_per_block = num_v_tiles // num_blocks_per_state_small
        start_v_tile = batch_inner * num_v_tiles_per_block

        num_value_heads_per_q = HV // H
        i_n = batch_idx // H
        i_h = batch_idx % H
        i_hv_base = i_h * num_value_heads_per_q
        num_hv_iters = num_value_heads_per_q

        pool_idx = h0_indices[i_n]

        if pool_idx >= 0:
            k_local = in_warp_tid // V_PER_WARP_SMALL
            v_local = in_warp_tid % V_PER_WARP_SMALL
            v_base = warp_idx * V_PER_WARP_SMALL
            v_idx = v_base + v_local

            smem = cutlass.utils.SmemAllocator()
            sData = smem.allocate_tensor(cutlass.Float32, smem_layout_staged, 128)
            smem_o_layout = cute.make_layout((TILE_V,), stride=(1,))
            smem_o = smem.allocate_tensor(cutlass.Float32, smem_o_layout, 128)
            smem_k_layout = cute.make_layout((TILE_K,), stride=(1,))
            smem_q_layout = cute.make_layout((TILE_K,), stride=(1,))
            smem_g_layout = cute.make_layout((TILE_K,), stride=(1,))
            smem_gk_layout = cute.make_layout((TILE_K,), stride=(1,))
            sK = smem.allocate_tensor(cutlass.Float32, smem_k_layout, 128)
            sQ = smem.allocate_tensor(cutlass.Float32, smem_q_layout, 128)
            sG = smem.allocate_tensor(cutlass.Float32, smem_g_layout, 128)
            sGK = smem.allocate_tensor(cutlass.Float32, smem_gk_layout, 128)

            # State load/store index setup (token-independent).
            kv_v_load = 0
            kv_k_load_base = 0
            kv_k_load_step = 0
            vk_k_load = 0
            vk_v_load_base = 0
            vk_v_load_step = 0
            if state_layout_is_kv:
                kv_v_load = tidx % TILE_V
                kv_k_load_base = tidx // TILE_V
                kv_k_load_step = NUM_THREADS // TILE_V
            else:
                vk_k_load = tidx % TILE_K
                vk_v_load_base = tidx // TILE_K
                vk_v_load_step = NUM_THREADS // TILE_K

            for hv_offset in range(num_hv_iters):
                i_hv = i_hv_base + hv_offset

                for v_tile_offset in range(num_v_tiles_per_block):
                    stage = v_tile_offset % NUM_STAGES
                    v_tile = start_v_tile + v_tile_offset
                    v_global_base = v_tile * TILE_V
                    v_global = v_tile * TILE_V + v_idx

                    # --- Load recurrent state for this V tile (once) ---
                    for k_iter in range(NUM_K_ITERS_SMALL):
                        k_load = 0
                        v_load = 0
                        if state_layout_is_kv:
                            k_load = kv_k_load_base + k_iter * kv_k_load_step
                            v_load = kv_v_load
                        else:
                            k_load = vk_k_load
                            v_load = vk_v_load_base + k_iter * vk_v_load_step
                        v_global_load = v_global_base + v_load
                        h_val = 0.0
                        if v_global_load < v.shape[3]:
                            if state_layout_is_kv:
                                h_val = cutlass.Float32(h0_source[(pool_idx, i_hv, k_load, v_global_load)])
                            else:
                                h_val = cutlass.Float32(h0_source[(pool_idx, i_hv, v_global_load, k_load)])
                        sData[(k_load, v_load, stage)] = h_val
                    cute.arch.barrier()

                    # --- Recurrence over the T tokens (state carried in sData) ---
                    for t in range(T):
                        # Load q_t, k_t for this token. The end-of-iteration
                        # barrier (or the state-load barrier on t=0) guarantees
                        # prior readers of sQ/sK are done before we overwrite.
                        if tidx < TILE_K:
                            sK[tidx] = cutlass.Float32(k[i_n, t, i_h, tidx])
                            sQ[tidx] = cutlass.Float32(q[i_n, t, i_h, tidx])
                        cute.arch.barrier()

                        if use_qk_l2norm:
                            sum_q_partial = 0.0
                            sum_k_partial = 0.0
                            if warp_idx == 0:
                                for norm_iter in range(4):
                                    norm_idx = in_warp_tid + norm_iter * 32
                                    q_val = sQ[norm_idx]
                                    k_val = sK[norm_idx]
                                    sum_q_partial += q_val * q_val
                                    sum_k_partial += k_val * k_val
                                for offset in [16, 8, 4, 2, 1]:
                                    sum_q_partial += cute.arch.shuffle_sync_bfly(sum_q_partial, offset=offset, mask=-1, mask_and_clamp=31)
                                    sum_k_partial += cute.arch.shuffle_sync_bfly(sum_k_partial, offset=offset, mask=-1, mask_and_clamp=31)
                                if in_warp_tid == 0:
                                    smem_o[0] = cute.rsqrt(sum_q_partial + 1e-6)
                                    smem_o[1] = cute.rsqrt(sum_k_partial + 1e-6)
                            cute.arch.barrier()

                            inv_norm_q = smem_o[0]
                            inv_norm_k = smem_o[1]
                            if tidx < TILE_K:
                                sK[tidx] = sK[tidx] * inv_norm_k
                                sQ[tidx] = sQ[tidx] * scale * inv_norm_q
                            cute.arch.barrier()
                        else:
                            if tidx < TILE_K:
                                sQ[tidx] = sQ[tidx] * scale
                            cute.arch.barrier()

                        # Channel-wise decay gate g_t and update gate beta_t.
                        r_exp_A = 0.0
                        if in_warp_tid == 0:
                            r_exp_A = cute.exp(cutlass.Float32(A_log[i_hv]))
                        r_exp_A = cute.arch.shuffle_sync(r_exp_A, 0)
                        if tidx < TILE_K:
                            r_a_k = cutlass.Float32(a[i_n, t, i_hv, tidx])
                            r_dt_bias_k = cutlass.Float32(dt_bias[i_hv, tidx])
                            x = r_a_k + r_dt_bias_k
                            beta_x = softplus_beta * x
                            softplus_x = 0.0
                            if beta_x <= softplus_threshold:
                                exp_beta_x = cute.exp(beta_x)
                                log_input = cutlass.Float32(1.0 + exp_beta_x)
                                log_result = cutlass.Float32(cute.log(log_input))
                                softplus_x = cutlass.Float32((cutlass.Float32(1.0) / softplus_beta) * log_result)
                            else:
                                softplus_x = x
                            sG[tidx] = cute.exp(-r_exp_A * softplus_x)

                        r_beta = 0.0
                        if in_warp_tid == 0:
                            r_b = cutlass.Float32(b[i_n, t, i_hv])
                            r_beta = 1.0 / (1.0 + cute.exp(-r_b))
                        r_beta = cute.arch.shuffle_sync(r_beta, 0)

                        if tidx < TILE_K:
                            sGK[tidx] = sG[tidx] * sK[tidx]
                        cute.arch.barrier()

                        r_v = 0.0
                        if v_global < v.shape[3]:
                            r_v = cutlass.Float32(v[i_n, t, i_hv, v_global])

                        # sum_hk = H @ (g_t * k_t)  (reduce over K)
                        sum_hk = 0.0
                        for k_iter in range(NUM_K_ITERS_SMALL):
                            k_base = k_iter * ROWS_PER_ITER_SMALL
                            k_idx = k_base + k_local
                            sum_hk += sData[(k_idx, v_idx, stage)] * sGK[k_idx]
                        for offset in K_REDUCE_OFFSETS:
                            sum_hk += cute.arch.shuffle_sync_bfly(
                                sum_hk, offset=offset * V_PER_WARP_SMALL, mask=-1, mask_and_clamp=31
                            )

                        v_new = (r_v - sum_hk) * r_beta
                        v_new = cute.arch.shuffle_sync(v_new, v_local)

                        # State update H = diag(g_t) H + k_t v_new^T, and
                        # sum_hq = H_new @ q_t  (reduce over K)
                        sum_hq = 0.0
                        for k_iter in range(NUM_K_ITERS_SMALL):
                            k_base = k_iter * ROWS_PER_ITER_SMALL
                            k_idx = k_base + k_local
                            h_old = sData[(k_idx, v_idx, stage)] * sG[k_idx]
                            h_new = h_old + sK[k_idx] * v_new
                            sData[(k_idx, v_idx, stage)] = h_new
                            sum_hq += h_new * sQ[k_idx]
                        for offset in K_REDUCE_OFFSETS:
                            sum_hq += cute.arch.shuffle_sync_bfly(
                                sum_hq, offset=offset * V_PER_WARP_SMALL, mask=-1, mask_and_clamp=31
                            )

                        if k_local == 0 and v_global < v.shape[3]:
                            o[(i_n, t, i_hv, v_global)] = cutlass.BFloat16(sum_hq)

                        # Ensure state writes are visible and all reads of
                        # sQ/sK/sG are done before the next token overwrites them.
                        cute.arch.barrier()

                    # --- Write final state back (once, after all T tokens) ---
                    for k_iter in cutlass.range(NUM_K_ITERS_SMALL, unroll=2):
                        k_write = 0
                        v_write = 0
                        if state_layout_is_kv:
                            k_write = kv_k_load_base + k_iter * kv_k_load_step
                            v_write = kv_v_load
                        else:
                            k_write = vk_k_load
                            v_write = vk_v_load_base + k_iter * vk_v_load_step
                        v_global_write = v_global_base + v_write
                        if v_global_write < v.shape[3]:
                            if state_layout_is_kv:
                                h0_source[(pool_idx, i_hv, k_write, v_global_write)] = sData[(k_write, v_write, stage)]
                            else:
                                h0_source[(pool_idx, i_hv, v_global_write, k_write)] = sData[(k_write, v_write, stage)]
                    cute.arch.barrier()

    return kda_kernel_small_batch_mtp


def _create_mtp_jit_functions(tile_v: int):
    """Create the JIT launcher for the MTP dense kernel at a given tile_v."""

    kda_small_mtp = _define_mtp_kernels(tile_v)
    TILE_V = tile_v
    TILE_V_PADDED = tile_v + 4

    @cute.jit
    def run_small_batch_mtp(
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        a: cute.Tensor,
        b: cute.Tensor,
        A_log: cute.Tensor,
        dt_bias: cute.Tensor,
        h0_source: cute.Tensor,
        h0_indices: cute.Tensor,
        o: cute.Tensor,
        softplus_beta: cutlass.Constexpr[float],
        softplus_threshold: cutlass.Constexpr[float],
        scale: cutlass.Constexpr[float],
        T: cutlass.Constexpr[int],
        H: cutlass.Constexpr[int],
        HV: cutlass.Constexpr[int],
        K: cutlass.Constexpr[int],
        V: cutlass.Constexpr[int],
        use_qk_l2norm: cutlass.Constexpr[bool],
        state_layout_is_kv: cutlass.Constexpr[bool],
        num_blocks_per_state_small: cutlass.Constexpr[int],
        stream: cuda.CUstream,
    ):
        del K
        n_indices = h0_indices.layout.shape[0]
        batch_size = n_indices * H

        num_v_tiles_small = cute.ceil_div(V, TILE_V)
        smem_layout_small = cute.make_layout(
            (TILE_K, TILE_V, NUM_STAGES),
            stride=(TILE_V_PADDED, 1, TILE_K * TILE_V_PADDED),
        )
        # Each SmemAllocator tensor is 128-byte aligned, so smem_o (4*TILE_V
        # bytes, < 128 for small tile_v) actually consumes a full 128-byte slot.
        # Round it up so the declared launch smem is never below real usage.
        smem_o_bytes = ((4 * TILE_V + 127) // 128) * 128
        smem_bytes_small = 4 * TILE_K * TILE_V_PADDED * NUM_STAGES + smem_o_bytes + 4 * TILE_K * 4 + 128

        kda_small_mtp(
            h0_source,
            smem_layout_small,
            num_v_tiles_small,
            num_blocks_per_state_small,
            q,
            k,
            v,
            a,
            b,
            A_log,
            dt_bias,
            o,
            h0_indices,
            softplus_beta,
            softplus_threshold,
            scale,
            T,
            H,
            HV,
            use_qk_l2norm,
            state_layout_is_kv,
        ).launch(
            grid=(batch_size * num_blocks_per_state_small, 1, 1),
            block=[NUM_THREADS, 1, 1],
            smem=smem_bytes_small,
            stream=stream,
        )

    return run_small_batch_mtp


_mtp_jit_functions: dict[int, object] = {}


def _get_mtp_jit_function(tile_v: int):
    if tile_v not in _mtp_jit_functions:
        _mtp_jit_functions[tile_v] = _create_mtp_jit_functions(tile_v)
    return _mtp_jit_functions[tile_v]


def _get_compiled_mtp_kernel(
    N,
    T,
    H,
    HV,
    K,
    V,
    pool_size,
    scale,
    use_qk_l2norm,
    state_layout_is_kv,
    num_blocks_per_state_small,
    softplus_beta,
    softplus_threshold,
    tile_v,
):
    """Get or lazily compile the MTP decode kernel for one shape/config (incl. T, tile_v)."""
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
        state_layout_is_kv,
        num_blocks_per_state_small,
        softplus_beta,
        softplus_threshold,
        tile_v,
    )
    if key in _compiled_mtp_kernels:
        return _compiled_mtp_kernels[key]

    q = torch.zeros(N, T, H, K, dtype=torch.bfloat16, device="cuda")
    k = torch.zeros(N, T, H, K, dtype=torch.bfloat16, device="cuda")
    v = torch.zeros(N, T, HV, V, dtype=torch.bfloat16, device="cuda")
    a = torch.zeros(N, T, HV, K, dtype=torch.bfloat16, device="cuda")
    b = torch.zeros(N, T, HV, dtype=torch.bfloat16, device="cuda")
    o = torch.zeros(N, T, HV, V, dtype=torch.bfloat16, device="cuda")
    A_log = torch.zeros(HV, dtype=torch.float32, device="cuda")
    dt_bias = torch.zeros(HV, K, dtype=torch.float32, device="cuda")
    if state_layout_is_kv:
        h0_source = torch.zeros(pool_size, HV, K, V, dtype=torch.float32, device="cuda")
    else:
        h0_source = torch.zeros(pool_size, HV, V, K, dtype=torch.float32, device="cuda")
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

    run_small_mtp = _get_mtp_jit_function(tile_v)
    compiled_kernel = cute.compile(
        run_small_mtp,
        q_tensor,
        k_tensor,
        v_tensor,
        a_tensor,
        b_tensor,
        A_log_tensor,
        dt_bias_tensor,
        h0_source_tensor,
        h0_indices_tensor,
        o_tensor,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        scale=scale,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        use_qk_l2norm=use_qk_l2norm,
        state_layout_is_kv=state_layout_is_kv,
        num_blocks_per_state_small=num_blocks_per_state_small,
        stream=stream,
        options="--enable-tvm-ffi --opt-level 1",
    )

    _compiled_mtp_kernels[key] = compiled_kernel
    logger.info(
        "CuTe DSL KDA MTP kernel compiled: "
        f"N={N}, T={T}, H={H}, HV={HV}, K={K}, V={V}, pool_size={pool_size}, tile_v={tile_v}"
    )
    return compiled_kernel


def _normalize_mtp_a(a: torch.Tensor, *, N: int, T: int, HV: int, K: int) -> torch.Tensor:
    """Normalize `a` to the compile-time dense MTP shape (N, T, HV, K)."""
    if a.dim() == 4 and tuple(a.shape) == (N, T, HV, K):
        return a
    if a.dim() == 3 and tuple(a.shape) == (N, T, HV * K):
        return a.view(N, T, HV, K)
    raise ValueError(f"Unexpected a shape for MTP dense: {tuple(a.shape)}; expected {(N, T, HV, K)}")


# Valid V-tile sizes: each must be a multiple of NUM_WARPS (4) so V_PER_WARP is
# integral, and the heuristic picks from these. (flashinfer's get_mtp_config uses
# {8,16,32,64}; we mirror the tile_v axis. ilp_rows is P3 (capped at 4 here);
# use_smem_v is produced but not yet consumed.)
_MTP_TILE_V_CHOICES = (8, 16, 32, 64)


def _select_mtp_config(
    N: int,
    HV: int,
    V: int,
    T: int,
    *,
    disable_state_update: bool = False,
) -> tuple[int, int, bool]:
    """Pick (tile_v, ilp_rows, use_smem_v) from work_units = N*HV.

    Mirrors FlashInfer's ``get_mtp_config`` (``gdn_decode_mtp.py:63-116``)
    thresholds, with one deliberate divergence: ``ilp_rows`` is **capped at 4**
    (the warp-spec kernel does not implement the ilp=8 path), so the >1024 bucket
    uses ilp=4 instead of FlashInfer's ilp=8 (which it picks there for
    state_update + T<=2). ``use_smem_v`` is produced for the large-batch bucket so
    Stage C can consume it; the current ``kda_decode_mtp_ws`` ignores it.

    Small work_units -> small tile_v -> more V-tiles -> more CTAs to fill the GPU;
    large work_units -> large tile_v -> fewer CTAs, better per-CTA efficiency.

    ``disable_state_update`` is accepted for parity with FlashInfer's signature
    (there it gates the ilp=8 choice); with ilp capped at 4 it does not change the
    selection here, but threading it keeps the call sites aligned with FlashInfer
    and ready for the ilp=8 path if it is ever added.
    """
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
        # Large batches. FlashInfer uses ilp=8 + use_smem_v=False here when
        # state_update is ON and T<=2; we cap ilp at 4, so use (64, 4, True)
        # uniformly. use_smem_v=True is produced for Stage C (use_smem_v/sOutput)
        # to consume — the current warp-spec kernel ignores the field.
        tile_v, ilp_rows, use_smem_v = 64, 4, True

    # Clamp to V and back off to a divisor of V (V is a multiple of 16 in
    # practice, so this is a no-op for the common V=128 case).
    tile_v = min(tile_v, V)
    while tile_v > _MTP_TILE_V_CHOICES[0] and V % tile_v != 0:
        tile_v //= 2

    # Legality backstop: ilp=4 requires (tile_v//4) % 4 == 0, i.e. tile_v % 16 == 0
    # (otherwise the warp-spec kernel's row_quad loop count truncates and trailing
    # V-rows are silently skipped). If clamping/back-off dropped tile_v below a
    # multiple of 16 (e.g. small V), fall back to the universally-legal ilp=2.
    if ilp_rows == 4 and tile_v % 16 != 0:
        ilp_rows = 2

    return tile_v, ilp_rows, use_smem_v


def _select_mtp_tile_v(N: int, HV: int, V: int, T: int) -> int:
    """Pick tile_v from work_units = N*HV (the tile_v axis of _select_mtp_config).

    Thin wrapper for callers that only need tile_v (the Route-2 kernel
    ``kda_decode_mtp`` and the benchmark); the full ``(tile_v, ilp_rows,
    use_smem_v)`` tuple lives in :func:`_select_mtp_config`.
    """
    return _select_mtp_config(N, HV, V, T)[0]


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
    tile_v: int | None = None,
) -> torch.Tensor:
    """CuTe DSL KDA multi-token-prediction (MTP) decode.

    Runs the KDA gated-delta-rule recurrence over T tokens per (batch, head) in
    one launch, threading the recurrent state across tokens and updating it
    in-place. Output is produced for every token position.

    State layout contract (token-independent):
        - "vk": (pool_size, HV, V, K), default
        - "kv": (pool_size, HV, K, V)

    Dense MTP shapes:
        q/k: (N, T, H, K)
        v:   (N, T, HV, V)
        a:   (N, T, HV, K)
        b:   (N, T, HV)
        out: (N, T, HV, V)

    P2 covers small AND mid/large batch through a single parameterized kernel:
    `tile_v` defaults to a work_units=N*HV heuristic, or can be overridden.
    Dense layout only; varlen is deferred to P4.
    """
    N, T, H, K = q.shape
    HV = v.shape[2]
    V = v.shape[3]

    if scale is None:
        scale = K**-0.5
    else:
        assert scale > 0, f"scale must be positive, got {scale}"

    assert K == TILE_K, f"KDA MTP kernel requires K={TILE_K}, got {K}"

    if tile_v is None:
        tile_v = _select_mtp_tile_v(N, HV, V, T)
    assert tile_v % 4 == 0, f"KDA MTP kernel requires tile_v % 4 == 0, got tile_v={tile_v}"
    assert V % tile_v == 0, f"KDA MTP kernel requires V % tile_v == 0, got V={V}, tile_v={tile_v}"

    state_layout = _canonicalize_state_layout(state_layout)

    # State is token-independent (pool, HV, V/K, K/V): reuse the single-token
    # normalizer / validator.
    h0_source, pool_size, state_layout_is_kv = _normalize_state_source(
        initial_state_source,
        N=N,
        HV=HV,
        K=K,
        V=V,
        device=q.device,
        state_layout=state_layout,
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

    num_blocks_per_state_small = 1

    stream = _get_cached_stream(q.device)
    compiled_kernel = _get_compiled_mtp_kernel(
        N,
        T,
        H,
        HV,
        K,
        V,
        pool_size,
        scale=scale,
        use_qk_l2norm=use_qk_l2norm_in_kernel,
        state_layout_is_kv=state_layout_is_kv,
        num_blocks_per_state_small=num_blocks_per_state_small,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        tile_v=tile_v,
    )

    compiled_kernel(
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        h0_source,
        initial_state_indices,
        o,
        stream,
    )

    return o
