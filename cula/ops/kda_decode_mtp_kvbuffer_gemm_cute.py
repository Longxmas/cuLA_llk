"""KDA MTP decode — GEMM/tensor-core KVBuffer chunkwise verify (CuTe DSL, sm_90).

CuTe port of the Triton reference ``kda_decode_mtp_kvbuffer_gemm.py`` (same math,
same flat-in-T goal): every reduction runs on tensor cores via warp-level
``mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32`` (Hopper has no warp-level
tf32 atom in the DSL, so the instruction is wrapped with llvm.inline_asm, the same
extension mechanism as ``ptx_umma_ext.py``). One CTA (4 warps) per (n, hv) head,
tokens padded to BT=16; pad rows/cols are zeroed in SMEM so they fall out of every
GEMM.

Phases (CTA-wide barriers between):
  P1 token-parallel l2norm/gating (warp w owns tokens w, w+4, ...)
  P2 K-parallel prefix scan (thread = channel): kdec/kinv/qdec, b_last, u-bufs
  P3 MMA: A = kdec kinv^T -> L = -tril_strict(beta*A); P = qdec kinv^T (lower)
  P4 log-depth inverse: inv = (I+L)(I+L^2)(I+L^4)(I+L^8)  [3 doubling steps]
  P5 V blocks (BVBLK cols): stage S0 in SMEM, then Skdec = kdec S0^T,
     x = beta*(v - Skdec), u = inv x, o = qdec S0^T + P u,
     S_T = b_last*(S0 + u^T kinv); u/o/state stored from MMA fragments.

mma.sync m16n8k8 fragment mapping (PTX ISA), gid = lane>>2, tig = lane&3:
  A row-major [16,8]: a0=A[gid][tig] a1=A[gid+8][tig] a2=A[gid][tig+4] a3=A[gid+8][tig+4]
  B col-major [8,8]:  b0=B[tig][gid] b1=B[tig+4][gid]
  C/D [16,8] f32:     c0=C[gid][2tig] c1=C[gid][2tig+1] c2=C[gid+8][2tig] c3=C[gid+8][2tig+1]
"""

import logging

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass._mlir import ir
from cutlass._mlir.dialects import arith as _arith
from cutlass._mlir.dialects import llvm as _llvm
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T as _T
from cutlass.cutlass_dsl import dsl_user_op

from cula.ops.kda_decode import (
    TILE_K,
    _get_cached_stream,
    _normalize_A_log,
    _normalize_dt_bias,
    _normalize_state_indices,
    _normalize_state_source,
    _prepare_output_tensor,
)
from cula.ops.kda_decode_mtp import VEC_SIZE, _normalize_mtp_a

logger = logging.getLogger(__name__)

BT = 16  # token pad (mma M)


@dsl_user_op
def _mma_m16n8k8_tf32(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, *, loc=None, ip=None):
    """One mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32; returns (d0, d1, d2, d3).

    a*/b* are Float32 values reinterpreted as tf32 (raw f32 bits; HW ignores the low
    mantissa bits — same truncation semantics as Triton's tf32 dots)."""
    f32 = _T.f32()
    i32 = _T.i32()

    def _bits(v):
        vv = v.ir_value(loc=loc, ip=ip) if hasattr(v, "ir_value") else v
        return _arith.bitcast(i32, vv, loc=loc, ip=ip)

    def _f(v):
        return v.ir_value(loc=loc, ip=ip) if hasattr(v, "ir_value") else v

    res_ty = _llvm.StructType.get_literal([f32, f32, f32, f32])
    res = _llvm.inline_asm(
        res_ty,
        [_bits(a0), _bits(a1), _bits(a2), _bits(a3), _bits(b0), _bits(b1),
         _f(c0), _f(c1), _f(c2), _f(c3)],
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{$0,$1,$2,$3}, {$4,$5,$6,$7}, {$8,$9}, {$10,$11,$12,$13};",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    d0 = cutlass.Float32(_llvm.extractvalue(f32, res, [0], loc=loc, ip=ip))
    d1 = cutlass.Float32(_llvm.extractvalue(f32, res, [1], loc=loc, ip=ip))
    d2 = cutlass.Float32(_llvm.extractvalue(f32, res, [2], loc=loc, ip=ip))
    d3 = cutlass.Float32(_llvm.extractvalue(f32, res, [3], loc=loc, ip=ip))
    return d0, d1, d2, d3


@cute.kernel
def kda_mtp_gemm_kvbuffer_cute_kernel(
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
    BVBLK: cutlass.Constexpr[int],
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
):
    tidx, _, _ = cute.arch.thread_idx()
    lane_id = tidx % 32
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    gid = lane_id // 4   # mma fragment group id (0..7)
    tig = lane_id % 4    # thread-in-group (0..3)

    num_warps: cutlass.Constexpr[int] = 4
    bidx, _, _ = cute.arch.block_idx()
    i_hv = bidx % HV
    i_n = bidx // HV
    i_h = i_hv // (HV // H)

    cache_idx = h0_indices[i_n]
    r_exp_A = cute.exp(cutlass.Float32(A_log[i_hv]), fastmath=fast_math)

    smem = cutlass.utils.SmemAllocator()
    sKdec = smem.allocate_tensor(cutlass.Float32, cute.make_layout((BT, K), stride=(K + 8, 1)), 16)
    sKinv = smem.allocate_tensor(cutlass.Float32, cute.make_layout((BT, K), stride=(K + 8, 1)), 16)
    sQdec = smem.allocate_tensor(cutlass.Float32, cute.make_layout((BT, K), stride=(K + 8, 1)), 16)
    sBeta = smem.allocate_tensor(cutlass.Float32, cute.make_layout((BT,)), 16)
    sBlast = smem.allocate_tensor(cutlass.Float32, cute.make_layout((K,)), 16)
    sL = smem.allocate_tensor(cutlass.Float32, cute.make_layout((BT, BT), stride=(BT + 1, 1)), 16)
    sP = smem.allocate_tensor(cutlass.Float32, cute.make_layout((BT, BT), stride=(BT + 1, 1)), 16)
    sInv = smem.allocate_tensor(cutlass.Float32, cute.make_layout((BT, BT), stride=(BT + 1, 1)), 16)
    sLp = smem.allocate_tensor(cutlass.Float32, cute.make_layout((BT, BT), stride=(BT + 1, 1)), 16)
    sX = smem.allocate_tensor(cutlass.Float32, cute.make_layout((BT, BVBLK), stride=(BVBLK + 1, 1)), 16)
    sU = smem.allocate_tensor(cutlass.Float32, cute.make_layout((BT, BVBLK), stride=(BVBLK + 1, 1)), 16)
    sS0 = smem.allocate_tensor(cutlass.Float32, cute.make_layout((BVBLK, K), stride=(K + 8, 1)), 16)

    r_qbf = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16)
    r_kbf = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16)
    r_qf = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    r_kf = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    r_s4 = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)

    if cache_idx >= 0:
        k_start = lane_id * vec_size
        flat_state_idx = cache_idx * HV + i_hv

        # ---- P1: token-parallel l2norm + staging (warp w owns tokens w, w+4, ...) ----
        tokens_per_warp: cutlass.Constexpr[int] = (T + num_warps - 1) // num_warps
        for tt in cutlass.range_constexpr(tokens_per_warp):
            t_tok = tt * num_warps + warp_idx
            if t_tok < T:
                q_tile = cute.local_tile(q, (1, 1, 1, vec_size), (i_n, t_tok, i_h, lane_id))
                k_tile = cute.local_tile(k, (1, 1, 1, vec_size), (i_n, t_tok, i_h, lane_id))
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
                    sKdec[t_tok, k_start + c] = r_kf[c]  # k_norm staged; decay applied in P2
                    sQdec[t_tok, k_start + c] = r_qf[c]  # q_scaled staged
                if lane_id == 0:
                    sBeta[t_tok] = cutlass.Float32(1.0) / (
                        cutlass.Float32(1.0)
                        + cute.exp(-cutlass.Float32(b[i_n, t_tok, i_hv]), fastmath=fast_math)
                    )
        # zero the BT pad rows (so they vanish in every GEMM) + pad beta
        for rp in cutlass.range_constexpr(BT - T):
            sKdec[T + rp, tidx] = cutlass.Float32(0.0)
            sKinv[T + rp, tidx] = cutlass.Float32(0.0)
            sQdec[T + rp, tidx] = cutlass.Float32(0.0)
        if tidx >= T:
            if tidx < BT:
                sBeta[tidx] = cutlass.Float32(0.0)
        cute.arch.barrier()

        # ---- P2: K-parallel scan (thread = channel kc): gate, prefix product, feature maps ----
        kc = tidx  # requires K == 128 == block size
        dtb_c = cutlass.Float32(dt_bias[i_hv, kc])
        lb = cutlass.Float32(0.0)
        for i_t in cutlass.range_constexpr(T):
            xg = cutlass.Float32(a[i_n, i_t, i_hv, kc]) + dtb_c
            beta_x = softplus_beta * xg
            exp_bx = cute.exp(beta_x, fastmath=fast_math)
            sp_val = (cutlass.Float32(1.0) / softplus_beta) * cute.log(
                cutlass.Float32(1.0) + exp_bx, fastmath=fast_math
            )
            use_sp = cutlass.Float32(1.0) if beta_x <= softplus_threshold else cutlass.Float32(0.0)
            sp_x = use_sp * sp_val + (cutlass.Float32(1.0) - use_sp) * xg
            lb = lb - r_exp_A * sp_x  # log cumulative decay
            bcum = cute.exp(lb, fastmath=fast_math)
            binv = cute.exp(-lb, fastmath=fast_math)
            kn = sKdec[i_t, kc]
            kinv_v = kn * binv
            sKdec[i_t, kc] = kn * bcum
            sKinv[i_t, kc] = kinv_v
            sQdec[i_t, kc] = sQdec[i_t, kc] * bcum
            if cutlass.const_expr(write_ubuf):
                kinv_buf[i_n, i_t, i_hv, kc] = kinv_v
                b_buf[i_n, i_t, i_hv, kc] = bcum
        sBlast[kc] = cute.exp(lb, fastmath=fast_math)
        cute.arch.barrier()

        # ---- P3: MMA A/P. warps 0,1 -> L n-halves; warps 2,3 -> P n-halves ----
        c0 = cutlass.Float32(0.0)
        c1 = cutlass.Float32(0.0)
        c2 = cutlass.Float32(0.0)
        c3 = cutlass.Float32(0.0)
        n_base3 = (warp_idx % 2) * 8
        for ks in cutlass.range_constexpr(K // 8):
            kb = ks * 8
            # DSL runtime-if: vars must pre-exist to survive the branch (scf.if yields)
            a0 = cutlass.Float32(0.0)
            a1 = cutlass.Float32(0.0)
            a2 = cutlass.Float32(0.0)
            a3 = cutlass.Float32(0.0)
            if warp_idx < 2:
                a0 = sKdec[gid, kb + tig]
                a1 = sKdec[gid + 8, kb + tig]
                a2 = sKdec[gid, kb + tig + 4]
                a3 = sKdec[gid + 8, kb + tig + 4]
            else:
                a0 = sQdec[gid, kb + tig]
                a1 = sQdec[gid + 8, kb + tig]
                a2 = sQdec[gid, kb + tig + 4]
                a3 = sQdec[gid + 8, kb + tig + 4]
            b0 = sKinv[n_base3 + gid, kb + tig]
            b1 = sKinv[n_base3 + gid, kb + tig + 4]
            c0, c1, c2, c3 = _mma_m16n8k8_tf32(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3)
        # store with causal masks; fragment positions: rows gid/gid+8, cols 2tig/2tig+1
        for fi in cutlass.range_constexpr(4):
            row = gid + (fi // 2) * 8
            col = n_base3 + 2 * tig + (fi % 2)
            cv = c0
            if cutlass.const_expr(fi == 1):
                cv = c1
            if cutlass.const_expr(fi == 2):
                cv = c2
            if cutlass.const_expr(fi == 3):
                cv = c3
            if warp_idx < 2:
                keep = cutlass.Float32(1.0) if row > col else cutlass.Float32(0.0)
                sL[row, col] = -sBeta[row] * cv * keep
            else:
                keep = cutlass.Float32(1.0) if row >= col else cutlass.Float32(0.0)
                sP[row, col] = cv * keep
        cute.arch.barrier()

        # ---- P4: inv = (I+L)(I+L^2)(I+L^4)(I+L^8), 3 doubling steps ----
        for j in cutlass.range_constexpr(2):
            flat = j * 128 + tidx
            rr = flat // BT
            cc = flat % BT
            one = cutlass.Float32(1.0) if rr == cc else cutlass.Float32(0.0)
            sInv[rr, cc] = one + sL[rr, cc]
            sLp[rr, cc] = sL[rr, cc]  # Lp starts as L; squared at each step
        cute.arch.barrier()
        # each doubling step MUST square Lp first, THEN increment inv with the squared
        # Lp (incrementing with the old Lp double-counts low powers — caught by the
        # pure-python oracle). warps 0,1 own the two n-halves; warps 2,3 just barrier.
        for step in cutlass.range_constexpr(3):
            nb = (warp_idx % 2) * 8
            d0 = cutlass.Float32(0.0)
            d1 = cutlass.Float32(0.0)
            d2 = cutlass.Float32(0.0)
            d3 = cutlass.Float32(0.0)
            if warp_idx < 2:  # (1) Lp <- Lp @ Lp
                for ks in cutlass.range_constexpr(2):
                    kb = ks * 8
                    a0 = sLp[gid, kb + tig]
                    a1 = sLp[gid + 8, kb + tig]
                    a2 = sLp[gid, kb + tig + 4]
                    a3 = sLp[gid + 8, kb + tig + 4]
                    b0 = sLp[kb + tig, nb + gid]
                    b1 = sLp[kb + tig + 4, nb + gid]
                    d0, d1, d2, d3 = _mma_m16n8k8_tf32(a0, a1, a2, a3, b0, b1, d0, d1, d2, d3)
            cute.arch.barrier()  # reads of old sLp done
            if warp_idx < 2:
                for fi in cutlass.range_constexpr(4):
                    row = gid + (fi // 2) * 8
                    col = nb + 2 * tig + (fi % 2)
                    dv = d0
                    if cutlass.const_expr(fi == 1):
                        dv = d1
                    if cutlass.const_expr(fi == 2):
                        dv = d2
                    if cutlass.const_expr(fi == 3):
                        dv = d3
                    sLp[row, col] = dv
            cute.arch.barrier()
            d0 = cutlass.Float32(0.0)
            d1 = cutlass.Float32(0.0)
            d2 = cutlass.Float32(0.0)
            d3 = cutlass.Float32(0.0)
            if warp_idx < 2:  # (2) inv += inv @ Lp(new)
                for ks in cutlass.range_constexpr(2):
                    kb = ks * 8
                    a0 = sInv[gid, kb + tig]
                    a1 = sInv[gid + 8, kb + tig]
                    a2 = sInv[gid, kb + tig + 4]
                    a3 = sInv[gid + 8, kb + tig + 4]
                    b0 = sLp[kb + tig, nb + gid]
                    b1 = sLp[kb + tig + 4, nb + gid]
                    d0, d1, d2, d3 = _mma_m16n8k8_tf32(a0, a1, a2, a3, b0, b1, d0, d1, d2, d3)
            cute.arch.barrier()  # reads of sInv done
            if warp_idx < 2:
                for fi in cutlass.range_constexpr(4):
                    row = gid + (fi // 2) * 8
                    col = nb + 2 * tig + (fi % 2)
                    dv = d0
                    if cutlass.const_expr(fi == 1):
                        dv = d1
                    if cutlass.const_expr(fi == 2):
                        dv = d2
                    if cutlass.const_expr(fi == 3):
                        dv = d3
                    sInv[row, col] = sInv[row, col] + dv
            cute.arch.barrier()

        # ---- P5: V blocks — Skdec, x, u, o, state, all on tensor cores ----
        num_v_blocks: cutlass.Constexpr[int] = V // BVBLK
        n_tiles_blk: cutlass.Constexpr[int] = BVBLK // 8
        for vb in cutlass.range_constexpr(num_v_blocks):
            v_base = vb * BVBLK
            # stage S0 block [BVBLK, K] cooperatively (float4)
            for j in cutlass.range_constexpr(BVBLK * K // (128 * vec_size)):
                flat = j * 128 + tidx
                s_row = flat // (K // vec_size)
                s_col4 = flat % (K // vec_size)
                h_tile = cute.local_tile(
                    h0_source, (1, 1, vec_size), (flat_state_idx, v_base + s_row, s_col4)
                )
                cute.autovec_copy(h_tile, r_s4)
                for cc4 in cutlass.range_constexpr(vec_size):
                    sS0[s_row, s_col4 * vec_size + cc4] = r_s4[cc4]
            cute.arch.barrier()

            # Skdec = kdec @ S0^T -> x = beta*(v - Skdec) -> sX  (warp owns n-tiles strided)
            for nt in cutlass.range_constexpr((n_tiles_blk + num_warps - 1) // num_warps):
                n_tile = nt * num_warps + warp_idx
                if n_tile < n_tiles_blk:
                    nb5 = n_tile * 8
                    e0 = cutlass.Float32(0.0)
                    e1 = cutlass.Float32(0.0)
                    e2 = cutlass.Float32(0.0)
                    e3 = cutlass.Float32(0.0)
                    for ks in cutlass.range_constexpr(K // 8):
                        kb = ks * 8
                        a0 = sKdec[gid, kb + tig]
                        a1 = sKdec[gid + 8, kb + tig]
                        a2 = sKdec[gid, kb + tig + 4]
                        a3 = sKdec[gid + 8, kb + tig + 4]
                        b0 = sS0[nb5 + gid, kb + tig]
                        b1 = sS0[nb5 + gid, kb + tig + 4]
                        e0, e1, e2, e3 = _mma_m16n8k8_tf32(a0, a1, a2, a3, b0, b1, e0, e1, e2, e3)
                    for fi in cutlass.range_constexpr(4):
                        row = gid + (fi // 2) * 8
                        col = nb5 + 2 * tig + (fi % 2)
                        ev = e0
                        if cutlass.const_expr(fi == 1):
                            ev = e1
                        if cutlass.const_expr(fi == 2):
                            ev = e2
                        if cutlass.const_expr(fi == 3):
                            ev = e3
                        # pad rows: clamp the t index via modulo (in-bounds) and mask the value
                        vmask = cutlass.Float32(1.0) if row < T else cutlass.Float32(0.0)
                        vv = cutlass.Float32(v[i_n, row % T, i_hv, v_base + col]) * vmask
                        sX[row, col] = sBeta[row] * (vv - ev)
            cute.arch.barrier()

            # u = inv @ x  (k = BT tokens, 2 k-slabs)
            for nt in cutlass.range_constexpr((n_tiles_blk + num_warps - 1) // num_warps):
                n_tile = nt * num_warps + warp_idx
                if n_tile < n_tiles_blk:
                    nb5 = n_tile * 8
                    e0 = cutlass.Float32(0.0)
                    e1 = cutlass.Float32(0.0)
                    e2 = cutlass.Float32(0.0)
                    e3 = cutlass.Float32(0.0)
                    for ks in cutlass.range_constexpr(BT // 8):
                        kb = ks * 8
                        a0 = sInv[gid, kb + tig]
                        a1 = sInv[gid + 8, kb + tig]
                        a2 = sInv[gid, kb + tig + 4]
                        a3 = sInv[gid + 8, kb + tig + 4]
                        b0 = sX[kb + tig, nb5 + gid]
                        b1 = sX[kb + tig + 4, nb5 + gid]
                        e0, e1, e2, e3 = _mma_m16n8k8_tf32(a0, a1, a2, a3, b0, b1, e0, e1, e2, e3)
                    for fi in cutlass.range_constexpr(4):
                        row = gid + (fi // 2) * 8
                        col = nb5 + 2 * tig + (fi % 2)
                        ev = e0
                        if cutlass.const_expr(fi == 1):
                            ev = e1
                        if cutlass.const_expr(fi == 2):
                            ev = e2
                        if cutlass.const_expr(fi == 3):
                            ev = e3
                        sU[row, col] = ev
                        if cutlass.const_expr(write_ubuf):
                            if row < T:
                                u_buf[i_n, row, i_hv, v_base + col] = ev
            cute.arch.barrier()

            # o = qdec @ S0^T + P @ u   (both accumulated into the same fragments)
            if cutlass.const_expr(emit_output):
                for nt in cutlass.range_constexpr((n_tiles_blk + num_warps - 1) // num_warps):
                    n_tile = nt * num_warps + warp_idx
                    if n_tile < n_tiles_blk:
                        nb5 = n_tile * 8
                        e0 = cutlass.Float32(0.0)
                        e1 = cutlass.Float32(0.0)
                        e2 = cutlass.Float32(0.0)
                        e3 = cutlass.Float32(0.0)
                        for ks in cutlass.range_constexpr(K // 8):
                            kb = ks * 8
                            a0 = sQdec[gid, kb + tig]
                            a1 = sQdec[gid + 8, kb + tig]
                            a2 = sQdec[gid, kb + tig + 4]
                            a3 = sQdec[gid + 8, kb + tig + 4]
                            b0 = sS0[nb5 + gid, kb + tig]
                            b1 = sS0[nb5 + gid, kb + tig + 4]
                            e0, e1, e2, e3 = _mma_m16n8k8_tf32(a0, a1, a2, a3, b0, b1, e0, e1, e2, e3)
                        for ks in cutlass.range_constexpr(BT // 8):
                            kb = ks * 8
                            a0 = sP[gid, kb + tig]
                            a1 = sP[gid + 8, kb + tig]
                            a2 = sP[gid, kb + tig + 4]
                            a3 = sP[gid + 8, kb + tig + 4]
                            b0 = sU[kb + tig, nb5 + gid]
                            b1 = sU[kb + tig + 4, nb5 + gid]
                            e0, e1, e2, e3 = _mma_m16n8k8_tf32(a0, a1, a2, a3, b0, b1, e0, e1, e2, e3)
                        for fi in cutlass.range_constexpr(4):
                            row = gid + (fi // 2) * 8
                            col = nb5 + 2 * tig + (fi % 2)
                            ev = e0
                            if cutlass.const_expr(fi == 1):
                                ev = e1
                            if cutlass.const_expr(fi == 2):
                                ev = e2
                            if cutlass.const_expr(fi == 3):
                                ev = e3
                            if row < T:
                                o[(i_n, row, i_hv, v_base + col)] = cutlass.BFloat16(ev)

            # S_T = b_last * (S0 + u^T @ kinv): M = v rows (BVBLK), N = K channels
            if cutlass.const_expr(not disable_state_update):
                m_tiles: cutlass.Constexpr[int] = BVBLK // 16
                pairs: cutlass.Constexpr[int] = m_tiles * (K // 8)
                for pp in cutlass.range_constexpr((pairs + num_warps - 1) // num_warps):
                    pidx = pp * num_warps + warp_idx
                    if pidx < pairs:
                        m_t = pidx % m_tiles
                        n_t = pidx // m_tiles
                        mb = m_t * 16
                        nb5 = n_t * 8
                        e0 = cutlass.Float32(0.0)
                        e1 = cutlass.Float32(0.0)
                        e2 = cutlass.Float32(0.0)
                        e3 = cutlass.Float32(0.0)
                        for ks in cutlass.range_constexpr(BT // 8):
                            kb = ks * 8
                            a0 = sU[kb + tig, mb + gid]        # A = u^T: A[v][t] = u[t][v]
                            a1 = sU[kb + tig, mb + gid + 8]
                            a2 = sU[kb + tig + 4, mb + gid]
                            a3 = sU[kb + tig + 4, mb + gid + 8]
                            b0 = sKinv[kb + tig, nb5 + gid]
                            b1 = sKinv[kb + tig + 4, nb5 + gid]
                            e0, e1, e2, e3 = _mma_m16n8k8_tf32(a0, a1, a2, a3, b0, b1, e0, e1, e2, e3)
                        for fi in cutlass.range_constexpr(4):
                            vrow = mb + gid + (fi // 2) * 8
                            kcol = nb5 + 2 * tig + (fi % 2)
                            ev = e0
                            if cutlass.const_expr(fi == 1):
                                ev = e1
                            if cutlass.const_expr(fi == 2):
                                ev = e2
                            if cutlass.const_expr(fi == 3):
                                ev = e3
                            h0_source[(flat_state_idx, v_base + vrow, kcol)] = (
                                sBlast[kcol] * (sS0[vrow, kcol] + ev)
                            )
            cute.arch.barrier()  # before next block overwrites sS0/sX/sU


@cute.jit
def run_kda_mtp_gemm_kvbuffer_cute_kernel(
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
    BVBLK: cutlass.Constexpr[int],
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
    """cute-gemm-kvbuffer launcher: grid = N*HV (one CTA per head), block = 128."""
    n_indices = h0_indices.layout.shape[0]
    grid_size = n_indices * HV
    smem_bytes = (
        3 * 4 * BT * (K + 8)        # sKdec/sKinv/sQdec
        + 4 * BT + 4 * K            # sBeta + sBlast
        + 4 * 4 * BT * (BT + 1)     # sL/sP/sInv/sLp
        + 2 * 4 * BT * (BVBLK + 1)  # sX/sU
        + 4 * BVBLK * (K + 8)       # sS0
        + 512                       # alignment slack
    )
    kda_mtp_gemm_kvbuffer_cute_kernel(
        h0_source, A_log, a, dt_bias, q, k, v, b, o, h0_indices,
        u_buf, kinv_buf, b_buf,
        vec_size, BVBLK,
        softplus_beta, softplus_threshold, scale,
        HV, T, H, K, V,
        use_qk_l2norm, disable_state_update, emit_output, write_ubuf, fast_math,
    ).launch(grid=(grid_size, 1, 1), block=[128, 1, 1], smem=smem_bytes, stream=stream)


_compiled_gemm_kvbuffer_cute_kernels: dict[tuple, object] = {}


def _get_compiled_gemm_kvbuffer_cute_kernel(
    N, T, H, HV, K, V, pool_size, bvblk, scale, use_qk_l2norm,
    disable_state_update, emit_output, write_ubuf,
    softplus_beta, softplus_threshold, opt_level=3, fast_math=True,
):
    key = (
        N, T, H, HV, K, V, pool_size, bvblk, scale, use_qk_l2norm,
        disable_state_update, emit_output, write_ubuf,
        softplus_beta, softplus_threshold, opt_level, fast_math,
    )
    if key in _compiled_gemm_kvbuffer_cute_kernels:
        return _compiled_gemm_kvbuffer_cute_kernels[key]

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
        run_kda_mtp_gemm_kvbuffer_cute_kernel,
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
        BVBLK=bvblk,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        scale=scale,
        HV=HV, T=T, H=H, K=K, V=V,
        use_qk_l2norm=use_qk_l2norm,
        disable_state_update=disable_state_update,
        emit_output=emit_output,
        write_ubuf=write_ubuf,
        fast_math=fast_math,
        stream=cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        options=f"--enable-tvm-ffi --opt-level {opt_level}",
    )
    _compiled_gemm_kvbuffer_cute_kernels[key] = compiled_kernel
    logger.info(
        "CuTe DSL KDA MTP gemm-KVBuffer (sm90 mma) kernel compiled: "
        f"N={N}, T={T}, HV={HV}, K={K}, V={V}, BVBLK={bvblk}, opt_level={opt_level}"
    )
    return compiled_kernel


def kda_decode_mtp_gemm_kvbuffer_cute(
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
    bvblk: int = 32,
    opt_level: int = 3,
    fast_math: bool = True,
) -> torch.Tensor:
    """KDA MTP decode — CuTe sm_90 tensor-core kvbuffer VERIFY (port of the Triton gemm op)."""
    N, T, H, K = q.shape
    HV = v.shape[2]
    V = v.shape[3]
    write_ubuf = u_buffer is not None

    if scale is None:
        scale = K**-0.5
    assert K == TILE_K == 128, f"cute-gemm-kvbuffer requires K=128, got {K}"
    assert T <= BT, f"cute-gemm-kvbuffer pads tokens to BT={BT}, needs T<={BT}, got {T}"
    assert V % bvblk == 0 and bvblk % 16 == 0, f"bvblk must divide V and be 16-aligned, got {bvblk}"

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
    compiled_kernel = _get_compiled_gemm_kvbuffer_cute_kernel(
        N, T, H, HV, K, V, pool_size, bvblk,
        scale=scale, use_qk_l2norm=use_qk_l2norm_in_kernel,
        disable_state_update=disable_state_update, emit_output=emit_output,
        write_ubuf=write_ubuf,
        softplus_beta=softplus_beta, softplus_threshold=softplus_threshold,
        opt_level=opt_level, fast_math=fast_math,
    )
    compiled_kernel(
        h0_source_flat, A_log, a, dt_bias, q, k, v, b, o,
        initial_state_indices, u_buf, kinv_buf, b_buf, stream,
    )
    return o
