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

import pathlib
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # for sibling test import

from cula.kda import kda_decode, kda_decode_mtp, kda_decode_mtp_ws

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


def run_kda_decode_mtp_via_loop_dense(q, k, v, a, b, A_log, dt_bias, state, scale):
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
        )
        o_all[:, t] = o_t.squeeze(1)  # (N, HV, V)

    return o_all, state_source


def run_kda_decode_mtp_dense(q, k, v, a, b, A_log, dt_bias, state, scale, tile_v=None):
    """
    Run the fused MTP kernel (kda_decode_mtp) in dense layout.

    Args use the MTP-shaped tensors:
        q, k: (N, T, H, K)   v: (N, T, HV, V)
        a: (N, T, HV, K)     b: (N, T, HV)     state: (N, HV, V, K)

    tile_v: optional V-tile override; None uses the work_units=N*HV heuristic.

    Returns:
        o:            (N, T, HV, V) bfloat16
        state_source: (N, HV, V, K) float32   (updated in-place by the kernel)
    """
    N = q.shape[0]
    state_source = state.clone().contiguous()  # (N, HV, V, K)
    indices = torch.arange(N, device=q.device, dtype=torch.int32)

    o = kda_decode_mtp(
        A_log=A_log,
        dt_bias=dt_bias,
        q=q.to(torch.bfloat16),
        k=k.to(torch.bfloat16),
        v=v.to(torch.bfloat16),
        a=a.to(torch.bfloat16),
        b=b.to(torch.bfloat16),
        initial_state_source=state_source,  # updated in-place after all T tokens
        initial_state_indices=indices,
        scale=scale,
        use_qk_l2norm_in_kernel=True,
        tile_v=tile_v,
    )
    return o, state_source  # (N, T, HV, V), (N, HV, V, K)


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
# Tests: fused MTP kernel (kda_decode_mtp) vs torch reference (dense layout).
# This is the load-bearing P1 test for the new kernel.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("N", [1, 4, 16, 64])
@pytest.mark.parametrize("T", [2, 4, 8])
@pytest.mark.parametrize("H,HV", [(8, 16), (16, 32)])
def test_kda_decode_mtp_kernel_dense(N, T, H, HV):
    K, V = 128, 128
    scale = K**-0.5
    q, k, v, a, b, A_log, dt_bias, state = make_inputs_mtp(N, T, H, HV, K, V)

    o_ref, state_ref = torch_kda_mtp_ref(
        q.float(), k.float(), v.float(), a, b.float(), A_log, dt_bias, state.clone(), scale
    )
    o_kernel, state_kernel = run_kda_decode_mtp_dense(
        q, k, v, a, b, A_log, dt_bias, state, scale
    )

    _assert_close("mtp kernel output", o_ref, o_kernel.float())
    _assert_close("mtp kernel final state", state_ref, state_kernel)


@pytest.mark.parametrize("T", [2, 4, 8])
def test_kda_decode_mtp_kernel_zero_state(T):
    N, H, HV, K, V = 4, 8, 16, 128, 128
    scale = K**-0.5
    q, k, v, a, b, A_log, dt_bias, _ = make_inputs_mtp(N, T, H, HV, K, V)
    state = torch.zeros(N, HV, V, K, device="cuda", dtype=torch.float32)

    o_ref, state_ref = torch_kda_mtp_ref(
        q.float(), k.float(), v.float(), a, b.float(), A_log, dt_bias, state.clone(), scale
    )
    o_kernel, state_kernel = run_kda_decode_mtp_dense(
        q, k, v, a, b, A_log, dt_bias, state, scale
    )

    _assert_close("mtp kernel zero-state output", o_ref, o_kernel.float())
    _assert_close("mtp kernel zero-state final state", state_ref, state_kernel)


@pytest.mark.parametrize("T", [2, 4])
def test_kda_decode_mtp_kernel_kv_layout(T):
    """Fused kernel with the 'kv' state layout (pool, HV, K, V)."""
    N, H, HV, K, V = 4, 8, 16, 128, 128
    scale = K**-0.5
    q, k, v, a, b, A_log, dt_bias, state_vk = make_inputs_mtp(N, T, H, HV, K, V)

    o_ref, state_ref_vk = torch_kda_mtp_ref(
        q.float(), k.float(), v.float(), a, b.float(), A_log, dt_bias, state_vk.clone(), scale
    )

    # kv layout state: (N, HV, K, V)
    state_kv = state_vk.permute(0, 1, 3, 2).contiguous()
    indices = torch.arange(N, device=q.device, dtype=torch.int32)
    o_kernel = kda_decode_mtp(
        A_log=A_log,
        dt_bias=dt_bias,
        q=q.to(torch.bfloat16),
        k=k.to(torch.bfloat16),
        v=v.to(torch.bfloat16),
        a=a.to(torch.bfloat16),
        b=b.to(torch.bfloat16),
        initial_state_source=state_kv,
        initial_state_indices=indices,
        scale=scale,
        use_qk_l2norm_in_kernel=True,
        state_layout="kv",
    )

    _assert_close("mtp kv output", o_ref, o_kernel.float())
    _assert_close("mtp kv final state", state_ref_vk, state_kv.permute(0, 1, 3, 2).contiguous())


# ---------------------------------------------------------------------------
# Tests (P2): parameterized tile_v. Each tile_v ∈ {8,16,32,64} must reproduce
# the same recurrence; only the CTA V-tiling / warp reduction width changes.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("tile_v", [8, 16, 32, 64])
@pytest.mark.parametrize("T", [2, 4])
def test_kda_decode_mtp_kernel_tile_v(tile_v, T):
    N, H, HV, K, V = 4, 8, 16, 128, 128
    scale = K**-0.5
    q, k, v, a, b, A_log, dt_bias, state = make_inputs_mtp(N, T, H, HV, K, V)

    o_ref, state_ref = torch_kda_mtp_ref(
        q.float(), k.float(), v.float(), a, b.float(), A_log, dt_bias, state.clone(), scale
    )
    o_kernel, state_kernel = run_kda_decode_mtp_dense(
        q, k, v, a, b, A_log, dt_bias, state, scale, tile_v=tile_v
    )

    _assert_close(f"mtp tile_v={tile_v} output", o_ref, o_kernel.float())
    _assert_close(f"mtp tile_v={tile_v} final state", state_ref, state_kernel)


# ---------------------------------------------------------------------------
# Test (P2): mid/large batch. N >= 1024 used to raise NotImplementedError;
# the single parameterized kernel + work_units heuristic now covers it. We
# validate against the established single-token kernel looped over T (fast,
# GPU-only) rather than the O(N*HV*T) python oracle.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("N", [1024, 2048])
def test_kda_decode_mtp_kernel_large_n(N):
    T, H, HV, K, V = 2, 8, 16, 128, 128
    scale = K**-0.5
    q, k, v, a, b, A_log, dt_bias, state = make_inputs_mtp(N, T, H, HV, K, V)

    # Reference: T sequential single-token kernel calls (state carried over).
    o_loop, state_loop = run_kda_decode_mtp_via_loop_dense(
        q, k, v, a, b, A_log, dt_bias, state, scale
    )
    # Fused MTP kernel with heuristic tile_v (work_units = N*HV -> tile_v=64 here).
    o_kernel, state_kernel = run_kda_decode_mtp_dense(
        q, k, v, a, b, A_log, dt_bias, state, scale
    )

    _assert_close(f"mtp large N={N} output", o_loop.float(), o_kernel.float())
    _assert_close(f"mtp large N={N} final state", state_loop, state_kernel)


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
# decay-first order with per-channel g). It is a SEPARATE implementation of the
# same contract as kda_decode_mtp, validated against the SAME fp32 torch oracle.
# Stage 1: vk-only, ilp_rows=2. bf16 rounding differs from the Route-2 kernel
# (different accumulation order), so we gate on the oracle, not on Route 2.
# ===========================================================================
def run_kda_decode_mtp_ws_dense(
    q, k, v, a, b, A_log, dt_bias, state, scale, tile_v=None, ilp_rows=2, use_packed_fma=None
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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
