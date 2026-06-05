#!/usr/bin/env python3
# Copyright 2025-2026 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
P0 oracle for KDA MTP (Multi-Token Prediction) decode — issue #17.

MTP runs the KDA gated-delta-rule recurrence over T > 1 tokens sequentially
per (batch, head), threading the recurrent state forward across tokens. This
file establishes the ground-truth PyTorch reference (``torch_kda_mtp_ref``)
and validates it against the equivalent of running the existing single-token
``kda_decode`` kernel T times with the state carried over between calls.

Per-token recurrence (identical to test_kda_decode.py, applied for t=0..T-1):
    gate  = exp(-exp(A_log) * softplus(a_t + dt_bias))   # (K,) per-channel decay
    v_new = sigmoid(b_t) * (v_t - H @ (gate * k_norm))
    H     = diag(gate) @ H + v_new @ k_norm^T            # state carried to t+1
    o_t   = H @ (l2norm(q_t) * scale)

Once the fused MTP kernel lands (P1), it will be validated against
``torch_kda_mtp_ref`` directly. P0 only needs the existing kernel and runs
fully on SM90 / H200.
"""

import os
import pathlib
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # for sibling test import

from cula.kda import (
    kda_decode,
    kda_decode_mtp_ws,
)
from cula.ops.kda_decode_mtp_ws import _select_mtp_config, _select_mtp_tile_v

# Trusted single-token reference from the existing decode test. We cross-check
# our MTP reference against it (pure torch, no kernel) so the MTP oracle is
# provably the same recurrence as test_kda_decode.py, just threaded over T.
from test_kda_decode import torch_kda_decode_ref


# ---------------------------------------------------------------------------
# PyTorch reference (ground truth oracle for the fused MTP kernel in P1)
# ---------------------------------------------------------------------------
def torch_kda_mtp_ref(
    q,  # (N, T, H, K) float32
    k,  # (N, T, H, K) float32   (H = query/key head count)
    v,  # (N, T, HV, V) float32
    a,  # (N, T, HV, K) float32
    b,  # (N, T, HV) float32
    A_log,  # (HV,) float32
    dt_bias,  # (HV, K) float32
    state,  # (N, HV, V, K) float32   initial recurrent state
    scale,  # float
    use_l2norm=True,
    softplus_beta=1.0,
    softplus_threshold=20.0,
):
    """
    Pure PyTorch reference for multi-token KDA decode.

    Loops the single-token recurrence over the T token dimension, threading
    the state forward. Returns the per-token outputs and the final state.

    Returns:
        o:         (N, T, HV, V) float32   per-token outputs
        state_new: (N, HV, V, K) float32   state after all T tokens
    """
    N, T, HV, V = v.shape
    K = q.shape[-1]
    H = q.shape[2]
    heads_per_group = HV // H  # GQA ratio

    A = torch.exp(A_log)  # (HV,)

    state_cur = state.clone()
    o = torch.zeros(N, T, HV, V, dtype=torch.float32, device=q.device)

    for t in range(T):
        for n in range(N):
            for hv in range(HV):
                i_h = hv // heads_per_group

                # Gate: exp(-A * softplus(a + dt_bias))  — per-K-channel vector
                x = a[n, t, hv, :] + dt_bias[hv, :]  # (K,)
                sp = F.softplus(x, beta=softplus_beta, threshold=softplus_threshold)
                gate = torch.exp(-A[hv] * sp)  # (K,)

                # L2 normalize q and k
                if use_l2norm:
                    q_vec = F.normalize(q[n, t, i_h, :], dim=0) * scale
                    k_vec = F.normalize(k[n, t, i_h, :], dim=0)
                else:
                    q_vec = q[n, t, i_h, :] * scale
                    k_vec = k[n, t, i_h, :]

                # H @ (gate * k_norm) — sum over K dim; state is (V, K)
                Hk = state_cur[n, hv] @ (gate * k_vec)  # (V,)

                # Delta correction
                beta_val = torch.sigmoid(b[n, t, hv])  # scalar
                v_new = beta_val * (v[n, t, hv, :] - Hk)  # (V,)

                # State update: diag(gate) @ H + outer(v_new, k_norm)
                state_cur[n, hv] = (
                    gate[None, :] * state_cur[n, hv] + v_new[:, None] * k_vec[None, :]
                )

                # Output: H_new @ q_scaled
                o[n, t, hv, :] = state_cur[n, hv] @ q_vec  # (V,)

    return o, state_cur


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_inputs_mtp(N, T, H, HV, K, V, device="cuda", seed=42):
    """Generate random MTP inputs (with a T token dimension)."""
    torch.manual_seed(seed)
    q = torch.randn(N, T, H, K, device=device, dtype=torch.bfloat16)
    k = torch.randn(N, T, H, K, device=device, dtype=torch.bfloat16)
    v = torch.randn(N, T, HV, V, device=device, dtype=torch.bfloat16)
    a = (torch.randn(N, T, HV, K, device=device, dtype=torch.float32) * 0.1).to(torch.bfloat16)
    b = torch.randn(N, T, HV, device=device, dtype=torch.bfloat16)
    A_log = -torch.rand(HV, device=device, dtype=torch.float32) * 2  # negative → A < 1
    dt_bias = torch.randn(HV, K, device=device, dtype=torch.float32) * 0.1
    state = torch.randn(N, HV, V, K, device=device, dtype=torch.float32) * 0.01
    return q, k, v, a, b, A_log, dt_bias, state


def run_kda_decode_mtp_via_loop_dense(q, k, v, a, b, A_log, dt_bias, state, scale, opt_level=1):
    """
    MTP semantics via T sequential calls to the existing single-token
    ``kda_decode`` kernel (dense layout), carrying the state across tokens.

    This is the P0 stand-in for the fused MTP kernel: it exercises the exact
    recurrence the fused kernel must reproduce.

    Args use the MTP-shaped tensors:
        q, k: (N, T, H, K)   v: (N, T, HV, V)
        a: (N, T, HV, K)     b: (N, T, HV)     state: (N, HV, V, K)

    Returns:
        o_all:        (N, T, HV, V) bfloat16
        state_source: (N, HV, V, K) float32   (mutated in-place across the loop)
    """
    N, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[3]

    state_source = state.clone().contiguous()  # (N, HV, V, K), updated in-place
    indices = torch.arange(N, device=q.device, dtype=torch.int32)
    o_all = torch.empty(N, T, HV, V, device=q.device, dtype=torch.bfloat16)

    for t in range(T):
        q_t = q[:, t].unsqueeze(1).contiguous()  # (N, 1, H, K)
        k_t = k[:, t].unsqueeze(1).contiguous()  # (N, 1, H, K)
        v_t = v[:, t].unsqueeze(1).contiguous()  # (N, 1, HV, V)
        a_t = a[:, t].unsqueeze(1).contiguous()  # (N, 1, HV, K)
        b_t = b[:, t].unsqueeze(1).contiguous()  # (N, 1, HV)

        o_t = kda_decode(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q_t.to(torch.bfloat16),
            k=k_t.to(torch.bfloat16),
            v=v_t.to(torch.bfloat16),
            a=a_t.to(torch.bfloat16),
            b=b_t.to(torch.bfloat16),
            initial_state_source=state_source,  # carried + updated in-place
            initial_state_indices=indices,
            scale=scale,
            use_qk_l2norm_in_kernel=True,
            opt_level=opt_level,
        )
        o_all[:, t] = o_t.squeeze(1)  # (N, HV, V)

    return o_all, state_source


def _assert_close(name, ref, actual, atol=3e-2, rtol=2e-2):
    """
    Assert tensors are close, always reporting the observed margins.

    Tolerances match test_kda_decode.py (atol 3e-2 / rtol 2e-2) so this stays
    aligned with the single-token baseline. The margin is printed on every call
    (run pytest with -s to see it) so we can observe how bf16 error accumulates
    across the T tokens instead of hiding it behind a looser bound.
    """
    diff = (ref.float() - actual.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"    [{name}] max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f} (atol={atol}, rtol={rtol})")
    ok = torch.allclose(ref.float(), actual.float(), atol=atol, rtol=rtol)
    assert ok, f"{name}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, atol={atol}, rtol={rtol}"


# ---------------------------------------------------------------------------
# Test: MTP reference IS the single-token reference threaded over T tokens.
# Pure torch (no kernel) — this is the alignment guarantee with
# test_kda_decode.py: it proves torch_kda_mtp_ref reproduces the trusted
# torch_kda_decode_ref step-by-step. Must match to fp32 epsilon.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("T", [1, 2, 4, 8])
def test_mtp_ref_equals_threaded_single_token_ref(T):
    N, H, HV, K, V = 4, 8, 16, 128, 128
    scale = K**-0.5
    q, k, v, a, b, A_log, dt_bias, state = make_inputs_mtp(N, T, H, HV, K, V)

    # Our MTP reference (loops over T internally)
    o_mtp, state_mtp = torch_kda_mtp_ref(
        q.float(), k.float(), v.float(), a, b.float(), A_log, dt_bias, state.clone(), scale
    )

    # Trusted single-token reference, threaded manually over T (identical inputs)
    state_cur = state.clone()
    o_manual = torch.zeros(N, T, HV, V, dtype=torch.float32, device=q.device)
    for t in range(T):
        o_t, state_cur = torch_kda_decode_ref(
            q[:, t].float(),
            k[:, t].float(),
            v[:, t].float(),
            a[:, t],
            b[:, t].float(),
            A_log,
            dt_bias,
            state_cur,
            scale,
        )
        o_manual[:, t] = o_t

    # Both are pure fp32 torch with identical arithmetic -> tight match.
    torch.testing.assert_close(o_mtp, o_manual, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(state_mtp, state_cur, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Tests: MTP reference matches looped single-token kernel (dense layout)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("N", [1, 4, 16, 64])
@pytest.mark.parametrize("T", [2, 4, 8])
@pytest.mark.parametrize("H,HV", [(8, 16), (16, 32)])
def test_kda_mtp_ref_matches_looped_kernel_dense(N, T, H, HV):
    K, V = 128, 128
    scale = K**-0.5
    q, k, v, a, b, A_log, dt_bias, state = make_inputs_mtp(N, T, H, HV, K, V)

    # Ground-truth reference (fp32 recurrence over T tokens)
    o_ref, state_ref = torch_kda_mtp_ref(
        q.float(),
        k.float(),
        v.float(),
        a,
        b.float(),
        A_log,
        dt_bias,
        state.clone(),
        scale,
    )

    # MTP via T sequential single-token kernel calls (state carried over)
    o_loop, state_loop = run_kda_decode_mtp_via_loop_dense(
        q, k, v, a, b, A_log, dt_bias, state, scale
    )

    _assert_close("mtp output", o_ref, o_loop.float())
    _assert_close("mtp final state", state_ref, state_loop)


# ---------------------------------------------------------------------------
# Test: a single MTP step (T=1) must equal the existing single-token decode
# (sanity check that the MTP reference reduces to the established baseline)
# ---------------------------------------------------------------------------
def test_kda_mtp_ref_t1_equals_single_token():
    N, T, H, HV, K, V = 4, 1, 8, 16, 128, 128
    scale = K**-0.5
    q, k, v, a, b, A_log, dt_bias, state = make_inputs_mtp(N, T, H, HV, K, V)

    o_ref, state_ref = torch_kda_mtp_ref(
        q.float(), k.float(), v.float(), a, b.float(), A_log, dt_bias, state.clone(), scale
    )
    o_loop, state_loop = run_kda_decode_mtp_via_loop_dense(
        q, k, v, a, b, A_log, dt_bias, state, scale
    )

    _assert_close("t1 output", o_ref, o_loop.float())
    _assert_close("t1 final state", state_ref, state_loop)


# ---------------------------------------------------------------------------
# Test: zero initial state
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("T", [2, 4, 8])
def test_kda_mtp_zero_state(T):
    N, H, HV, K, V = 4, 8, 16, 128, 128
    scale = K**-0.5
    q, k, v, a, b, A_log, dt_bias, _ = make_inputs_mtp(N, T, H, HV, K, V)
    state = torch.zeros(N, HV, V, K, device="cuda", dtype=torch.float32)

    o_ref, state_ref = torch_kda_mtp_ref(
        q.float(), k.float(), v.float(), a, b.float(), A_log, dt_bias, state.clone(), scale
    )
    o_loop, state_loop = run_kda_decode_mtp_via_loop_dense(
        q, k, v, a, b, A_log, dt_bias, state, scale
    )

    _assert_close("zero-state output", o_ref, o_loop.float())
    _assert_close("zero-state final state", state_ref, state_loop)


# ===========================================================================
# Route 1 (warp-specialized port): kda_decode_mtp_ws
#
# kda_decode_mtp_ws is the FlashInfer-style warp-specialized variant (grid =
# N*HV*num_v_tiles, register-resident state, full-warp shuffle reduction,
# decay-first order with per-channel g), validated against the fp32 torch oracle.
# vk-only. bf16 rounding (accumulation order) differs from the looped single-token
# kernel, so we gate on the oracle, not on bit-for-bit equality with the loop.
# ===========================================================================
def run_kda_decode_mtp_ws_dense(
    q, k, v, a, b, A_log, dt_bias, state, scale, tile_v=None, ilp_rows=2,
    use_packed_fma=None, use_smem_v=None, opt_level=1, fast_math=False,
):
    """Run the warp-specialized fused MTP kernel (kda_decode_mtp_ws), dense vk."""
    N = q.shape[0]
    state_source = state.clone().contiguous()  # (N, HV, V, K), updated in-place
    indices = torch.arange(N, device=q.device, dtype=torch.int32)

    o = kda_decode_mtp_ws(
        A_log=A_log,
        dt_bias=dt_bias,
        q=q.to(torch.bfloat16),
        k=k.to(torch.bfloat16),
        v=v.to(torch.bfloat16),
        a=a.to(torch.bfloat16),
        b=b.to(torch.bfloat16),
        initial_state_source=state_source,
        initial_state_indices=indices,
        scale=scale,
        use_qk_l2norm_in_kernel=True,
        tile_v=tile_v,
        ilp_rows=ilp_rows,
        use_packed_fma=use_packed_fma,
        use_smem_v=use_smem_v,
        opt_level=opt_level,
        fast_math=fast_math,
    )
    return o, state_source  # (N, T, HV, V), (N, HV, V, K)


@pytest.mark.parametrize("N", [1, 4, 16, 64])
@pytest.mark.parametrize("T", [2, 4, 8])
@pytest.mark.parametrize("H,HV", [(8, 16), (16, 32)])
def test_kda_decode_mtp_ws_kernel_dense(N, T, H, HV):
    K, V = 128, 128
    scale = K**-0.5
    q, k, v, a, b, A_log, dt_bias, state = make_inputs_mtp(N, T, H, HV, K, V)

    o_ref, state_ref = torch_kda_mtp_ref(
        q.float(), k.float(), v.float(), a, b.float(), A_log, dt_bias, state.clone(), scale
    )
    o_kernel, state_kernel = run_kda_decode_mtp_ws_dense(
        q, k, v, a, b, A_log, dt_bias, state, scale
    )

    _assert_close("ws mtp kernel output", o_ref, o_kernel.float())
    _assert_close("ws mtp kernel final state", state_ref, state_kernel)


@pytest.mark.parametrize("tile_v", [8, 16, 32, 64])
@pytest.mark.parametrize("T", [2, 4])
def test_kda_decode_mtp_ws_kernel_tile_v(tile_v, T):
    N, H, HV, K, V = 4, 8, 16, 128, 128
    scale = K**-0.5
    q, k, v, a, b, A_log, dt_bias, state = make_inputs_mtp(N, T, H, HV, K, V)

    o_ref, state_ref = torch_kda_mtp_ref(
        q.float(), k.float(), v.float(), a, b.float(), A_log, dt_bias, state.clone(), scale
    )
    o_kernel, state_kernel = run_kda_decode_mtp_ws_dense(
        q, k, v, a, b, A_log, dt_bias, state, scale, tile_v=tile_v
    )

    _assert_close(f"ws mtp tile_v={tile_v} output", o_ref, o_kernel.float())
    _assert_close(f"ws mtp tile_v={tile_v} final state", state_ref, state_kernel)


@pytest.mark.parametrize("T", [2, 4, 8])
def test_kda_decode_mtp_ws_kernel_zero_state(T):
    N, H, HV, K, V = 4, 8, 16, 128, 128
    scale = K**-0.5
    q, k, v, a, b, A_log, dt_bias, _ = make_inputs_mtp(N, T, H, HV, K, V)
    state = torch.zeros(N, HV, V, K, device="cuda", dtype=torch.float32)

    o_ref, state_ref = torch_kda_mtp_ref(
        q.float(), k.float(), v.float(), a, b.float(), A_log, dt_bias, state.clone(), scale
    )
    o_kernel, state_kernel = run_kda_decode_mtp_ws_dense(
        q, k, v, a, b, A_log, dt_bias, state, scale
    )

    _assert_close("ws mtp zero-state output", o_ref, o_kernel.float())
    _assert_close("ws mtp zero-state final state", state_ref, state_kernel)


@pytest.mark.parametrize("N", [1024, 2048])
def test_kda_decode_mtp_ws_kernel_large_n(N):
    """Mid/large batch (work_units heuristic picks tile_v=64). Validate against
    the established single-token kernel looped over T (GPU-only, fast)."""
    T, H, HV, K, V = 2, 8, 16, 128, 128
    scale = K**-0.5
    q, k, v, a, b, A_log, dt_bias, state = make_inputs_mtp(N, T, H, HV, K, V)

    o_loop, state_loop = run_kda_decode_mtp_via_loop_dense(
        q, k, v, a, b, A_log, dt_bias, state, scale
    )
    o_kernel, state_kernel = run_kda_decode_mtp_ws_dense(
        q, k, v, a, b, A_log, dt_bias, state, scale
    )

    _assert_close(f"ws mtp large N={N} output", o_loop.float(), o_kernel.float())
    _assert_close(f"ws mtp large N={N} final state", state_loop, state_kernel)


def test_kda_decode_mtp_ws_disable_state_update():
    """disable_state_update=True must leave the state pool untouched while still
    producing correct per-token outputs."""
    N, T, H, HV, K, V = 4, 4, 8, 16, 128, 128
    scale = K**-0.5
    q, k, v, a, b, A_log, dt_bias, state = make_inputs_mtp(N, T, H, HV, K, V)

    o_ref, _ = torch_kda_mtp_ref(
        q.float(), k.float(), v.float(), a, b.float(), A_log, dt_bias, state.clone(), scale
    )

    state_source = state.clone().contiguous()
    state_before = state_source.clone()
    indices = torch.arange(N, device=q.device, dtype=torch.int32)
    o_kernel = kda_decode_mtp_ws(
        A_log=A_log,
        dt_bias=dt_bias,
        q=q.to(torch.bfloat16),
        k=k.to(torch.bfloat16),
        v=v.to(torch.bfloat16),
        a=a.to(torch.bfloat16),
        b=b.to(torch.bfloat16),
        initial_state_source=state_source,
        initial_state_indices=indices,
        scale=scale,
        use_qk_l2norm_in_kernel=True,
        disable_state_update=True,
    )

    _assert_close("ws disable_state_update output", o_ref, o_kernel.float())
    # State must be byte-for-byte unchanged.
    assert torch.equal(state_source, state_before), "state pool was modified despite disable_state_update=True"


# ===========================================================================
# Route 1 Stage 2: ilp_rows=4 path (fused steps 1+2 & 4+5, double accumulators,
# packed F32x2 FMA on SM100 / scalar fma_pair fallback). Requires tile_v % 16 == 0.
#
# The core dense test sweeps use_packed_fma ∈ {False, True} to validate BOTH FMA
# paths against the same fp32 oracle: the False instances exercise the portable
# scalar fusion (always compiles); the True instances exercise the SM100 packed
# intrinsic. The tile_v / large_n / disable tests pin use_packed_fma=False so the
# fusion+double-accumulator math is validated independent of the packed API.
# ===========================================================================
@pytest.mark.parametrize("use_packed_fma", [False, True])
@pytest.mark.parametrize("N,T,H,HV", [(1, 2, 8, 16), (4, 4, 8, 16), (16, 2, 16, 32)])
def test_kda_decode_mtp_ws_kernel_ilp4_dense(N, T, H, HV, use_packed_fma):
    K, V = 128, 128
    scale = K**-0.5
    q, k, v, a, b, A_log, dt_bias, state = make_inputs_mtp(N, T, H, HV, K, V)

    o_ref, state_ref = torch_kda_mtp_ref(
        q.float(), k.float(), v.float(), a, b.float(), A_log, dt_bias, state.clone(), scale
    )
    o_kernel, state_kernel = run_kda_decode_mtp_ws_dense(
        q, k, v, a, b, A_log, dt_bias, state, scale,
        tile_v=32, ilp_rows=4, use_packed_fma=use_packed_fma,
    )

    tag = f"ws ilp4 packed={use_packed_fma}"
    _assert_close(f"{tag} output", o_ref, o_kernel.float())
    _assert_close(f"{tag} final state", state_ref, state_kernel)


@pytest.mark.parametrize("tile_v", [16, 32, 64])
@pytest.mark.parametrize("T", [2, 4])
def test_kda_decode_mtp_ws_kernel_ilp4_tile_v(tile_v, T):
    # tile_v ∈ {16,32,64} -> quarter_rows = (tile_v//4)//4 ∈ {1,2,4}, so this also
    # exercises the row_quad loop running more than once. Scalar FMA (portable).
    N, H, HV, K, V = 4, 8, 16, 128, 128
    scale = K**-0.5
    q, k, v, a, b, A_log, dt_bias, state = make_inputs_mtp(N, T, H, HV, K, V)

    o_ref, state_ref = torch_kda_mtp_ref(
        q.float(), k.float(), v.float(), a, b.float(), A_log, dt_bias, state.clone(), scale
    )
    o_kernel, state_kernel = run_kda_decode_mtp_ws_dense(
        q, k, v, a, b, A_log, dt_bias, state, scale,
        tile_v=tile_v, ilp_rows=4, use_packed_fma=False,
    )

    _assert_close(f"ws ilp4 tile_v={tile_v} output", o_ref, o_kernel.float())
    _assert_close(f"ws ilp4 tile_v={tile_v} final state", state_ref, state_kernel)


@pytest.mark.parametrize("N", [1024, 2048])
def test_kda_decode_mtp_ws_kernel_ilp4_large_n(N):
    """Mid/large batch with ilp=4 (heuristic tile_v=64, valid for ilp=4). Validate
    against the single-token kernel looped over T (GPU-only, fast)."""
    T, H, HV, K, V = 2, 8, 16, 128, 128
    scale = K**-0.5
    q, k, v, a, b, A_log, dt_bias, state = make_inputs_mtp(N, T, H, HV, K, V)

    o_loop, state_loop = run_kda_decode_mtp_via_loop_dense(
        q, k, v, a, b, A_log, dt_bias, state, scale
    )
    o_kernel, state_kernel = run_kda_decode_mtp_ws_dense(
        q, k, v, a, b, A_log, dt_bias, state, scale, ilp_rows=4, use_packed_fma=False
    )

    _assert_close(f"ws ilp4 large N={N} output", o_loop.float(), o_kernel.float())
    _assert_close(f"ws ilp4 large N={N} final state", state_loop, state_kernel)


def test_kda_decode_mtp_ws_ilp4_disable_state_update():
    """ilp=4 disable_state_update=True: correct per-token output, pool untouched."""
    N, T, H, HV, K, V = 4, 4, 8, 16, 128, 128
    scale = K**-0.5
    q, k, v, a, b, A_log, dt_bias, state = make_inputs_mtp(N, T, H, HV, K, V)

    o_ref, _ = torch_kda_mtp_ref(
        q.float(), k.float(), v.float(), a, b.float(), A_log, dt_bias, state.clone(), scale
    )

    state_source = state.clone().contiguous()
    state_before = state_source.clone()
    indices = torch.arange(N, device=q.device, dtype=torch.int32)
    o_kernel = kda_decode_mtp_ws(
        A_log=A_log,
        dt_bias=dt_bias,
        q=q.to(torch.bfloat16),
        k=k.to(torch.bfloat16),
        v=v.to(torch.bfloat16),
        a=a.to(torch.bfloat16),
        b=b.to(torch.bfloat16),
        initial_state_source=state_source,
        initial_state_indices=indices,
        scale=scale,
        use_qk_l2norm_in_kernel=True,
        tile_v=32,
        ilp_rows=4,
        use_packed_fma=False,
        disable_state_update=True,
    )

    _assert_close("ws ilp4 disable_state_update output", o_ref, o_kernel.float())
    assert torch.equal(state_source, state_before), "state pool was modified despite disable_state_update=True"


def test_kda_decode_mtp_ws_ilp4_rejects_bad_tile_v():
    """ilp=4 requires tile_v % 16 == 0; tile_v=8 must raise, not silently skip rows."""
    N, T, H, HV, K, V = 4, 2, 8, 16, 128, 128
    scale = K**-0.5
    q, k, v, a, b, A_log, dt_bias, state = make_inputs_mtp(N, T, H, HV, K, V)
    with pytest.raises(AssertionError):
        run_kda_decode_mtp_ws_dense(
            q, k, v, a, b, A_log, dt_bias, state, scale,
            tile_v=8, ilp_rows=4, use_packed_fma=False,
        )


# ===========================================================================
# Stage 3-A: joint (tile_v, ilp_rows) heuristic. _select_mtp_config mirrors
# FlashInfer's get_mtp_config (ilp capped at 4); kda_decode_mtp_ws(ilp_rows=None)
# is the production default that consumes it (small work_units -> ilp=2, mid/large
# -> ilp=4). use_smem_v is produced for Stage C but not yet consumed.
# ===========================================================================
@pytest.mark.parametrize(
    "N,HV,V,T,expected",
    [
        # work_units = N*HV buckets at V=128 (no clamp / back-off).
        (1, 16, 128, 2, (8, 2, False)),    # wu=16    <=64
        (4, 16, 128, 4, (8, 2, False)),    # wu=64    <=64 (boundary)
        (1, 65, 128, 2, (16, 4, False)),   # wu=65    <=128
        (8, 16, 128, 2, (16, 4, False)),   # wu=128   <=128 (boundary)
        (16, 16, 128, 2, (16, 2, False)),  # wu=256   <=448, T<=2
        (16, 16, 128, 4, (32, 4, False)),  # wu=256   <=448, T>=3
        (7, 64, 128, 2, (16, 2, False)),   # wu=448   <=448 (boundary), T<=2
        (7, 64, 128, 8, (32, 4, False)),   # wu=448   <=448 (boundary), T>=3
        (16, 64, 128, 2, (32, 4, False)),  # wu=1024  <=1024 (boundary)
        (64, 16, 128, 8, (32, 4, False)),  # wu=1024  <=1024
        (17, 64, 128, 2, (64, 4, True)),   # wu=1088  >1024 (large; smem_v=True)
        (256, 64, 128, 8, (64, 4, True)),  # wu=16384 >1024
        # clamp to V + ilp legality back-off at small V.
        (8, 16, 8, 2, (8, 2, False)),      # wu=128 picks (16,4); clamp->8; 8%16!=0 -> ilp=2
        (8, 16, 16, 2, (16, 4, False)),    # wu=128 picks (16,4); clamp keeps 16; legal -> ilp=4
    ],
)
def test_select_mtp_config(N, HV, V, T, expected):
    """The joint heuristic returns the expected (tile_v, ilp_rows, use_smem_v),
    and _select_mtp_tile_v stays the tile_v projection of the same selection."""
    assert _select_mtp_config(N, HV, V, T) == expected
    assert _select_mtp_tile_v(N, HV, V, T) == expected[0]


def test_select_mtp_config_ilp_capped_at_4():
    """We cap ilp at 4 (no ilp=8 path): no bucket — including >1024 with
    state_update ON + T<=2, where FlashInfer would pick ilp=8 — returns ilp>4."""
    for N in (1, 8, 16, 64, 256, 4096):
        for HV in (16, 64):
            for T in (1, 2, 4, 8):
                for dsu in (False, True):
                    _, ilp, _ = _select_mtp_config(N, HV, 128, T, disable_state_update=dsu)
                    assert ilp in (2, 4), f"N={N},HV={HV},T={T},dsu={dsu} -> ilp={ilp}"


@pytest.mark.parametrize(
    "N,H,HV,T,expected_ilp",
    [
        (1, 8, 16, 2, 2),    # work_units=16   -> tile_v=8,  ilp=2
        (8, 8, 16, 2, 4),    # work_units=128  -> tile_v=16, ilp=4
        (16, 8, 16, 2, 2),   # work_units=256, T<=2 -> tile_v=16, ilp=2
        (16, 8, 16, 4, 4),   # work_units=256, T>=3 -> tile_v=32, ilp=4
        (64, 8, 16, 2, 4),   # work_units=1024 -> tile_v=32, ilp=4
        (128, 8, 16, 2, 4),  # work_units=2048 -> tile_v=64, ilp=4 (large bucket)
    ],
)
def test_kda_decode_mtp_ws_auto_config(N, H, HV, T, expected_ilp):
    """Production default: kda_decode_mtp_ws with NO tile_v/ilp_rows lets the
    work_units heuristic pick both. Confirm the picked ilp AND that the auto path
    is numerically correct vs the looped single-token kernel (GPU-only, any N)."""
    K, V = 128, 128
    scale = K**-0.5

    # Document the branch this case exercises (and guard the default selection).
    _, sel_ilp, _ = _select_mtp_config(N, HV, V, T)
    assert sel_ilp == expected_ilp, (
        f"work_units={N * HV}, T={T}: expected ilp={expected_ilp}, got {sel_ilp}"
    )

    q, k, v, a, b, A_log, dt_bias, state = make_inputs_mtp(N, T, H, HV, K, V)

    o_loop, state_loop = run_kda_decode_mtp_via_loop_dense(
        q, k, v, a, b, A_log, dt_bias, state, scale
    )

    # No tile_v, no ilp_rows -> both come from _select_mtp_config.
    state_source = state.clone().contiguous()
    indices = torch.arange(N, device=q.device, dtype=torch.int32)
    o_kernel = kda_decode_mtp_ws(
        A_log=A_log,
        dt_bias=dt_bias,
        q=q.to(torch.bfloat16),
        k=k.to(torch.bfloat16),
        v=v.to(torch.bfloat16),
        a=a.to(torch.bfloat16),
        b=b.to(torch.bfloat16),
        initial_state_source=state_source,
        initial_state_indices=indices,
        scale=scale,
        use_qk_l2norm_in_kernel=True,
    )

    tag = f"ws auto (wu={N * HV}, T={T}, ilp={sel_ilp})"
    _assert_close(f"{tag} output", o_loop.float(), o_kernel.float())
    _assert_close(f"{tag} final state", state_loop, state_source)


# ===========================================================================
# Stage 3-C: use_smem_v + sOutput merged writeback. Preloading v into SMEM and
# accumulating outputs in SMEM (flushed coalesced at kernel end) is a pure
# data-movement change: the v value read (Float32 of the same bf16) and the
# output written (BFloat16 of the same sum) are byte-identical to the GMEM path,
# so use_smem_v=True must match use_smem_v=False BIT-FOR-BIT — the strongest
# correctness check, and (since sOutput is a new cross-warp SMEM write) a race
# check too. We also gate against the fp32 oracle and the looped kernel.
# ===========================================================================
@pytest.mark.parametrize("use_smem_v", [False, True])
@pytest.mark.parametrize(
    "tile_v,ilp_rows",
    [(8, 2), (16, 2), (32, 2), (64, 2), (16, 4), (32, 4), (64, 4)],
)
def test_kda_decode_mtp_ws_smem_v_dense(use_smem_v, tile_v, ilp_rows):
    """use_smem_v on/off is correct vs the fp32 oracle across tile_v (incl 64) and
    both ilp paths. use_packed_fma=False so the fusion math is checked portably."""
    N, T, H, HV, K, V = 4, 4, 8, 16, 128, 128
    scale = K**-0.5
    q, k, v, a, b, A_log, dt_bias, state = make_inputs_mtp(N, T, H, HV, K, V)

    o_ref, state_ref = torch_kda_mtp_ref(
        q.float(), k.float(), v.float(), a, b.float(), A_log, dt_bias, state.clone(), scale
    )
    o_kernel, state_kernel = run_kda_decode_mtp_ws_dense(
        q, k, v, a, b, A_log, dt_bias, state, scale,
        tile_v=tile_v, ilp_rows=ilp_rows, use_packed_fma=False, use_smem_v=use_smem_v,
    )

    tag = f"ws smem_v={use_smem_v} tile_v={tile_v} ilp={ilp_rows}"
    _assert_close(f"{tag} output", o_ref, o_kernel.float())
    _assert_close(f"{tag} final state", state_ref, state_kernel)


@pytest.mark.parametrize(
    "tile_v,ilp_rows",
    [(8, 2), (16, 2), (32, 2), (64, 2), (16, 4), (32, 4), (64, 4)],
)
def test_kda_decode_mtp_ws_smem_v_bit_identical_to_gmem(tile_v, ilp_rows):
    """use_smem_v only relocates v reads (SMEM vs GMEM) and o writes (SMEM-then-
    flush vs direct); the arithmetic is untouched, so the output AND the state
    pool must be byte-for-byte identical to the use_smem_v=False path. A mismatch
    means either a relocation bug or a cross-warp sOutput race."""
    N, T, H, HV, K, V = 4, 4, 8, 16, 128, 128
    scale = K**-0.5
    q, k, v, a, b, A_log, dt_bias, state = make_inputs_mtp(N, T, H, HV, K, V)

    o_gmem, state_gmem = run_kda_decode_mtp_ws_dense(
        q, k, v, a, b, A_log, dt_bias, state, scale,
        tile_v=tile_v, ilp_rows=ilp_rows, use_packed_fma=False, use_smem_v=False,
    )
    o_smem, state_smem = run_kda_decode_mtp_ws_dense(
        q, k, v, a, b, A_log, dt_bias, state, scale,
        tile_v=tile_v, ilp_rows=ilp_rows, use_packed_fma=False, use_smem_v=True,
    )

    assert torch.equal(o_smem, o_gmem), (
        f"use_smem_v output diverged from GMEM path (tile_v={tile_v}, ilp={ilp_rows}): "
        f"max|diff|={(o_smem.float() - o_gmem.float()).abs().max().item()}"
    )
    assert torch.equal(state_smem, state_gmem), (
        f"use_smem_v state diverged from GMEM path (tile_v={tile_v}, ilp={ilp_rows})"
    )


@pytest.mark.parametrize("N", [1024, 2048])
def test_kda_decode_mtp_ws_smem_v_large_n(N):
    """Large batch is where use_smem_v earns its keep (the heuristic turns it on at
    tile_v=64). Validate the explicit use_smem_v=True path vs the looped single-
    token kernel (GPU-only, fast), ilp=4 (heuristic tile_v=64)."""
    T, H, HV, K, V = 2, 8, 16, 128, 128
    scale = K**-0.5
    q, k, v, a, b, A_log, dt_bias, state = make_inputs_mtp(N, T, H, HV, K, V)

    o_loop, state_loop = run_kda_decode_mtp_via_loop_dense(
        q, k, v, a, b, A_log, dt_bias, state, scale
    )
    o_kernel, state_kernel = run_kda_decode_mtp_ws_dense(
        q, k, v, a, b, A_log, dt_bias, state, scale,
        tile_v=64, ilp_rows=4, use_packed_fma=False, use_smem_v=True,
    )

    _assert_close(f"ws smem_v large N={N} output", o_loop.float(), o_kernel.float())
    _assert_close(f"ws smem_v large N={N} final state", state_loop, state_kernel)


def test_kda_decode_mtp_ws_smem_v_auto_heuristic_large_n():
    """Production default (no use_smem_v arg) auto-enables use_smem_v at the large-
    batch (tile_v=64) bucket; the auto path stays numerically correct vs the loop.
    work_units = N*HV = 256*16 = 4096 > 1024 -> (tile_v, ilp, smem_v)=(64,4,True)."""
    N, T, H, HV, K, V = 256, 2, 8, 16, 128, 128
    scale = K**-0.5
    assert _select_mtp_config(N, HV, V, T) == (64, 4, True)

    q, k, v, a, b, A_log, dt_bias, state = make_inputs_mtp(N, T, H, HV, K, V)
    o_loop, state_loop = run_kda_decode_mtp_via_loop_dense(
        q, k, v, a, b, A_log, dt_bias, state, scale
    )

    # No tile_v / ilp_rows / use_smem_v -> all three from _select_mtp_config.
    state_source = state.clone().contiguous()
    indices = torch.arange(N, device=q.device, dtype=torch.int32)
    o_kernel = kda_decode_mtp_ws(
        A_log=A_log, dt_bias=dt_bias,
        q=q.to(torch.bfloat16), k=k.to(torch.bfloat16), v=v.to(torch.bfloat16),
        a=a.to(torch.bfloat16), b=b.to(torch.bfloat16),
        initial_state_source=state_source, initial_state_indices=indices,
        scale=scale, use_qk_l2norm_in_kernel=True,
    )
    _assert_close("ws auto smem_v output", o_loop.float(), o_kernel.float())
    _assert_close("ws auto smem_v final state", state_loop, state_source)


@pytest.mark.parametrize("ilp_rows", [2, 4])
def test_kda_decode_mtp_ws_smem_v_determinism(ilp_rows):
    """sOutput is a new cross-warp SMEM write; the merged flush reads it after a
    barrier. Repeat the use_smem_v=True launch many times from the same inputs and
    assert bit-identical output + state every iteration (surfaces any sOutput race
    or flush ordering bug). The B200 gate runs the heavier 10K count."""
    N, T, H, HV, K, V = 16, 4, 8, 16, 128, 128
    tile_v = 64
    scale = K**-0.5
    q, k, v, a, b, A_log, dt_bias, state = make_inputs_mtp(N, T, H, HV, K, V)
    indices = torch.arange(N, device=q.device, dtype=torch.int32)

    def launch():
        st = state.clone().contiguous()
        o = kda_decode_mtp_ws(
            A_log=A_log, dt_bias=dt_bias,
            q=q.to(torch.bfloat16), k=k.to(torch.bfloat16), v=v.to(torch.bfloat16),
            a=a.to(torch.bfloat16), b=b.to(torch.bfloat16),
            initial_state_source=st, initial_state_indices=indices,
            scale=scale, use_qk_l2norm_in_kernel=True,
            tile_v=tile_v, ilp_rows=ilp_rows, use_packed_fma=False, use_smem_v=True,
        )
        return o.clone(), st

    o_ref, st_ref = launch()
    n_iters = int(os.environ.get("KDA_MTP_DET_ITERS", "100"))  # B200 10K gate: KDA_MTP_DET_ITERS=10000
    for i in range(n_iters):
        o_i, st_i = launch()
        assert torch.equal(o_i, o_ref), f"smem_v output non-deterministic at iter {i} (ilp={ilp_rows})"
        assert torch.equal(st_i, st_ref), f"smem_v state non-deterministic at iter {i} (ilp={ilp_rows})"


# ===========================================================================
# Stage 3-D: intermediate-state snapshots (speculative-decoding support). When an
# intermediate_states_buffer [N, T, HV, V, K] is passed, the kernel snapshots the
# post-token state after EVERY token (sequence-indexed: buffer[i_n, i_t, i_hv]).
# Produce-only — cuLA fills the buffer, never rolls back. Checks: (1) each token's
# snapshot matches the fp32 oracle state after that token; (2) the t=T-1 snapshot
# equals the final state pool BIT-FOR-BIT (same r_h written by both); (3) the new
# per-token GMEM snapshot stores are deterministic (race check).
# ===========================================================================
def oracle_intermediate_states(q, k, v, a, b, A_log, dt_bias, state, scale):
    """fp32 ground-truth per-token state: stack state_cur after each token via the
    trusted single-token reference. Returns [N, T, HV, V, K]."""
    N, T = q.shape[0], q.shape[1]
    HV, V, K = v.shape[2], v.shape[3], q.shape[3]
    state_cur = state.clone()
    inter = torch.zeros(N, T, HV, V, K, dtype=torch.float32, device=q.device)
    for t in range(T):
        _, state_cur = torch_kda_decode_ref(
            q[:, t].float(), k[:, t].float(), v[:, t].float(),
            a[:, t], b[:, t].float(), A_log, dt_bias, state_cur, scale,
        )
        inter[:, t] = state_cur
    return inter


def run_kda_decode_mtp_ws_with_intermediate(
    q, k, v, a, b, A_log, dt_bias, state, scale,
    tile_v=None, ilp_rows=2, use_packed_fma=False, use_smem_v=None,
):
    """Run the ws kernel with an intermediate-state buffer; return o, final state
    pool, and the filled buffer [N, T, HV, V, K]."""
    N, T = q.shape[0], q.shape[1]
    HV, V, K = v.shape[2], v.shape[3], q.shape[3]
    state_source = state.clone().contiguous()
    indices = torch.arange(N, device=q.device, dtype=torch.int32)
    inter = torch.zeros(N, T, HV, V, K, device=q.device, dtype=torch.float32)

    o = kda_decode_mtp_ws(
        A_log=A_log, dt_bias=dt_bias,
        q=q.to(torch.bfloat16), k=k.to(torch.bfloat16), v=v.to(torch.bfloat16),
        a=a.to(torch.bfloat16), b=b.to(torch.bfloat16),
        initial_state_source=state_source, initial_state_indices=indices,
        scale=scale, use_qk_l2norm_in_kernel=True,
        tile_v=tile_v, ilp_rows=ilp_rows, use_packed_fma=use_packed_fma,
        use_smem_v=use_smem_v, intermediate_states_buffer=inter,
    )
    return o, state_source, inter


@pytest.mark.parametrize("use_smem_v", [False, True])
@pytest.mark.parametrize("tile_v,ilp_rows", [(16, 2), (32, 4), (64, 4)])
def test_kda_decode_mtp_ws_intermediate_vs_oracle(use_smem_v, tile_v, ilp_rows):
    """Every per-token snapshot matches the fp32 oracle state after that token.
    Swept over both ilp paths and use_smem_v (orthogonal constexpr) to confirm the
    snapshot is correct regardless of the v/output path."""
    N, T, H, HV, K, V = 4, 4, 8, 16, 128, 128
    scale = K**-0.5
    q, k, v, a, b, A_log, dt_bias, state = make_inputs_mtp(N, T, H, HV, K, V)

    inter_ref = oracle_intermediate_states(q, k, v, a, b, A_log, dt_bias, state.clone(), scale)
    o_kernel, _state, inter_kernel = run_kda_decode_mtp_ws_with_intermediate(
        q, k, v, a, b, A_log, dt_bias, state, scale,
        tile_v=tile_v, ilp_rows=ilp_rows, use_packed_fma=False, use_smem_v=use_smem_v,
    )

    tag = f"ws inter smem_v={use_smem_v} tile_v={tile_v} ilp={ilp_rows}"
    # Each token's snapshot must equal the oracle state after consuming that token.
    for t in range(T):
        _assert_close(f"{tag} snapshot[t={t}]", inter_ref[:, t], inter_kernel[:, t])


@pytest.mark.parametrize("tile_v,ilp_rows", [(16, 2), (32, 4), (64, 4)])
def test_kda_decode_mtp_ws_intermediate_last_eq_final_state(tile_v, ilp_rows):
    """The t=T-1 snapshot is the same r_h the kernel writes back as the final state
    (step 5 only reads r_h), so with indices=arange(N) (cache_idx==i_n) the last
    snapshot equals the final state pool BIT-FOR-BIT. A mismatch means the snapshot
    fired at the wrong point or used the wrong index."""
    N, T, H, HV, K, V = 4, 4, 8, 16, 128, 128
    scale = K**-0.5
    q, k, v, a, b, A_log, dt_bias, state = make_inputs_mtp(N, T, H, HV, K, V)

    o_kernel, state_final, inter_kernel = run_kda_decode_mtp_ws_with_intermediate(
        q, k, v, a, b, A_log, dt_bias, state, scale,
        tile_v=tile_v, ilp_rows=ilp_rows, use_packed_fma=False,
    )

    # state_final: (N, HV, V, K); inter_kernel[:, T-1]: (N, HV, V, K). Same r_h.
    assert torch.equal(inter_kernel[:, T - 1], state_final), (
        f"t=T-1 snapshot != final state pool (tile_v={tile_v}, ilp={ilp_rows}): "
        f"max|diff|={(inter_kernel[:, T - 1] - state_final).abs().max().item()}"
    )


def test_kda_decode_mtp_ws_intermediate_disable_state_update_last_eq_oracle():
    """With disable_state_update=True the pool is untouched, but snapshots still fire
    every token; the t=T-1 snapshot must match the oracle final state (the cache is
    the only place the post-token state is exposed)."""
    N, T, H, HV, K, V = 4, 4, 8, 16, 128, 128
    scale = K**-0.5
    q, k, v, a, b, A_log, dt_bias, state = make_inputs_mtp(N, T, H, HV, K, V)

    inter_ref = oracle_intermediate_states(q, k, v, a, b, A_log, dt_bias, state.clone(), scale)

    state_source = state.clone().contiguous()
    state_before = state_source.clone()
    indices = torch.arange(N, device=q.device, dtype=torch.int32)
    inter = torch.zeros(N, T, HV, V, K, device=q.device, dtype=torch.float32)
    o = kda_decode_mtp_ws(
        A_log=A_log, dt_bias=dt_bias,
        q=q.to(torch.bfloat16), k=k.to(torch.bfloat16), v=v.to(torch.bfloat16),
        a=a.to(torch.bfloat16), b=b.to(torch.bfloat16),
        initial_state_source=state_source, initial_state_indices=indices,
        scale=scale, use_qk_l2norm_in_kernel=True,
        tile_v=32, ilp_rows=4, use_packed_fma=False,
        disable_state_update=True, intermediate_states_buffer=inter,
    )

    assert torch.equal(state_source, state_before), "pool modified despite disable_state_update=True"
    for t in range(T):
        _assert_close(f"inter+dsu snapshot[t={t}]", inter_ref[:, t], inter[:, t])


def test_kda_decode_mtp_ws_intermediate_buffer_validation():
    """Bad intermediate_states_buffer shape / dtype must raise (not silently
    mis-index or drop snapshots)."""
    N, T, H, HV, K, V = 4, 2, 8, 16, 128, 128
    scale = K**-0.5
    q, k, v, a, b, A_log, dt_bias, state = make_inputs_mtp(N, T, H, HV, K, V)
    state_source = state.clone().contiguous()
    indices = torch.arange(N, device=q.device, dtype=torch.int32)

    def _call(buf):
        return kda_decode_mtp_ws(
            A_log=A_log, dt_bias=dt_bias,
            q=q.to(torch.bfloat16), k=k.to(torch.bfloat16), v=v.to(torch.bfloat16),
            a=a.to(torch.bfloat16), b=b.to(torch.bfloat16),
            initial_state_source=state_source, initial_state_indices=indices,
            scale=scale, use_qk_l2norm_in_kernel=True, tile_v=32, ilp_rows=4,
            use_packed_fma=False, intermediate_states_buffer=buf,
        )

    # Wrong shape (T mismatch).
    with pytest.raises((ValueError, AssertionError)):
        _call(torch.zeros(N, T + 1, HV, V, K, device="cuda", dtype=torch.float32))
    # Wrong dtype.
    with pytest.raises((ValueError, AssertionError)):
        _call(torch.zeros(N, T, HV, V, K, device="cuda", dtype=torch.bfloat16))


@pytest.mark.parametrize("ilp_rows", [2, 4])
def test_kda_decode_mtp_ws_intermediate_determinism(ilp_rows):
    """The per-token snapshot is a new GMEM write; repeat from identical inputs and
    assert the full buffer (+ output + final state) is bit-identical every time."""
    N, T, H, HV, K, V = 8, 4, 8, 16, 128, 128
    tile_v = 32
    scale = K**-0.5
    q, k, v, a, b, A_log, dt_bias, state = make_inputs_mtp(N, T, H, HV, K, V)

    def launch():
        return run_kda_decode_mtp_ws_with_intermediate(
            q, k, v, a, b, A_log, dt_bias, state, scale,
            tile_v=tile_v, ilp_rows=ilp_rows, use_packed_fma=False,
        )

    o_ref, st_ref, inter_ref = launch()
    o_ref, st_ref, inter_ref = o_ref.clone(), st_ref.clone(), inter_ref.clone()
    n_iters = int(os.environ.get("KDA_MTP_DET_ITERS", "100"))  # B200 10K gate: KDA_MTP_DET_ITERS=10000
    for i in range(n_iters):
        o_i, st_i, inter_i = launch()
        assert torch.equal(inter_i, inter_ref), f"snapshot non-deterministic at iter {i} (ilp={ilp_rows})"
        assert torch.equal(o_i, o_ref), f"output non-deterministic at iter {i} (ilp={ilp_rows})"
        assert torch.equal(st_i, st_ref), f"final state non-deterministic at iter {i} (ilp={ilp_rows})"


# ===========================================================================
# Compile-knob tuning (issue 17): opt_level + fast_math
#
# Two compile-knobs are now exposed on the decode entry points, implemented in
# ONE change but TESTED SEPARATELY (each test varies exactly one knob, the other
# pinned at its default):
#   - opt_level: CuTe DSL --opt-level (codegen optimization; NOT a kernel
#     constexpr). On kda_decode (single-token / loop baseline) and
#     kda_decode_mtp_ws. Default 1 (the historical pin); 2/3 must
#     stay correct (opt-level should not change the answer beyond FP reassoc).
#   - fast_math: kernel constexpr threading fastmath= onto the ws
#     transcendentals (exp/log/rsqrt). Default False reproduces the validated
#     no-fastmath port; True is the FlashInfer-style fast intrinsic. kda_decode
#     (single-token) is intentionally NOT given fast_math (stays no-fastmath).
# All gate on the SAME fp32 torch oracle at the standard 3e-2/2e-2 band: these
# knobs are codegen/intrinsic choices, not algorithm changes, so accuracy must
# hold. The default (opt_level=1, fast_math=False) path is already covered by
# every other test in this file; these add the 2/3 and fast_math=True legs.
# ===========================================================================
@pytest.mark.parametrize("opt_level", [2, 3])
def test_kda_decode_mtp_ws_opt_level(opt_level):
    """ws kernel at --opt-level 2/3 stays correct vs the fp32 oracle (fast_math off)."""
    N, T, H, HV, K, V = 16, 4, 16, 32, 128, 128
    scale = K**-0.5
    q, k, v, a, b, A_log, dt_bias, state = make_inputs_mtp(N, T, H, HV, K, V)

    o_ref, state_ref = torch_kda_mtp_ref(
        q.float(), k.float(), v.float(), a, b.float(), A_log, dt_bias, state.clone(), scale
    )
    o_kernel, state_kernel = run_kda_decode_mtp_ws_dense(
        q, k, v, a, b, A_log, dt_bias, state, scale, opt_level=opt_level, fast_math=False
    )

    _assert_close(f"ws opt_level={opt_level} output", o_ref, o_kernel.float())
    _assert_close(f"ws opt_level={opt_level} final state", state_ref, state_kernel)


@pytest.mark.parametrize("ilp_rows", [2, 4])
def test_kda_decode_mtp_ws_fast_math(ilp_rows):
    """ws kernel with fast_math=True stays within the oracle band (opt_level fixed).

    The fast intrinsics drift more than the default exp/log/rsqrt, but bf16 input
    rounding dominates, so the 3e-2/2e-2 band still holds. tile_v=32 keeps ilp=4
    legal so both ILP paths' transcendentals are exercised."""
    N, T, H, HV, K, V = 16, 4, 16, 32, 128, 128
    tile_v = 32
    scale = K**-0.5
    q, k, v, a, b, A_log, dt_bias, state = make_inputs_mtp(N, T, H, HV, K, V)

    o_ref, state_ref = torch_kda_mtp_ref(
        q.float(), k.float(), v.float(), a, b.float(), A_log, dt_bias, state.clone(), scale
    )
    o_kernel, state_kernel = run_kda_decode_mtp_ws_dense(
        q, k, v, a, b, A_log, dt_bias, state, scale,
        tile_v=tile_v, ilp_rows=ilp_rows, use_packed_fma=False,
        opt_level=1, fast_math=True,
    )

    _assert_close(f"ws fast_math ilp={ilp_rows} output", o_ref, o_kernel.float())
    _assert_close(f"ws fast_math ilp={ilp_rows} final state", state_ref, state_kernel)


@pytest.mark.parametrize("opt_level", [2, 3])
def test_kda_decode_single_token_opt_level(opt_level):
    """Single-token kda_decode (the bench 'loop' baseline) at --opt-level 2/3 stays
    correct vs the fp32 oracle. This is the 3rd kernel in the opt-level scope; it
    has no fast_math knob (intentionally no-fastmath)."""
    N, T, H, HV, K, V = 4, 2, 16, 32, 128, 128
    scale = K**-0.5
    q, k, v, a, b, A_log, dt_bias, state = make_inputs_mtp(N, T, H, HV, K, V)

    o_ref, state_ref = torch_kda_mtp_ref(
        q.float(), k.float(), v.float(), a, b.float(), A_log, dt_bias, state.clone(), scale
    )
    o_loop, state_loop = run_kda_decode_mtp_via_loop_dense(
        q, k, v, a, b, A_log, dt_bias, state, scale, opt_level=opt_level
    )

    _assert_close(f"single-token opt_level={opt_level} output", o_ref, o_loop.float())
    _assert_close(f"single-token opt_level={opt_level} final state", state_ref, state_loop)


# --- shipped production default (opt_level=3 + fast_math=True, B200-tuned) ----
# The dense/tile_v/etc. tests drive the helpers, which pin the reference path
# (opt_level=1, fast_math=False); these two exercise the actual default callers
# get with NO knob args (both knobs at once), so the shipped config is covered.
def test_kda_decode_mtp_ws_default_config():
    """ws entry point with no knob args == production default (opt3 + fast_math)."""
    N, T, H, HV, K, V = 16, 4, 16, 32, 128, 128
    scale = K**-0.5
    q, k, v, a, b, A_log, dt_bias, state = make_inputs_mtp(N, T, H, HV, K, V)
    o_ref, state_ref = torch_kda_mtp_ref(
        q.float(), k.float(), v.float(), a, b.float(), A_log, dt_bias, state.clone(), scale
    )
    state_source = state.clone().contiguous()
    indices = torch.arange(N, device=q.device, dtype=torch.int32)
    o = kda_decode_mtp_ws(  # no opt_level / fast_math -> production default (3, True)
        A_log=A_log, dt_bias=dt_bias,
        q=q.to(torch.bfloat16), k=k.to(torch.bfloat16), v=v.to(torch.bfloat16),
        a=a.to(torch.bfloat16), b=b.to(torch.bfloat16),
        initial_state_source=state_source, initial_state_indices=indices,
        scale=scale, use_qk_l2norm_in_kernel=True,
    )
    _assert_close("ws default-config (opt3+fast_math) output", o_ref, o.float())
    _assert_close("ws default-config (opt3+fast_math) final state", state_ref, state_source)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
