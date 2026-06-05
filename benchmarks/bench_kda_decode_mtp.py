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
bench_kda_decode_mtp.py — perf harness for KDA MTP decode (ws vs looped).

Compares routes that compute the SAME T-token recurrence:
  1. looped : T sequential single-token kda_decode launches (state carried over).
  2. ws     : a single kda_decode_mtp_ws launch, ilp=2 (explicit) [FlashInfer warp-spec].
  3. ws4    : the same warp-spec kernel, ilp=4 (explicit; fused steps 1+2 & 4+5,
              double accumulators, packed F32x2 FMA on SM100). Skipped (n/a) when
              the selected tile_v gives (tile_v//4) not divisible by 4.
  4. ws4_smemv: ws4 plus Stage-C use_smem_v=True (v preloaded into SMEM + merged
              coalesced output writeback). Head-to-head vs ws4 isolates the
              use_smem_v win; same (tile_v//4)%4 gate as ws4.
  5. ws_auto: the warp-spec kernel with ilp_rows=None — the PRODUCTION DEFAULT,
              where the work_units=N*HV heuristic (_select_mtp_config) picks
              (tile_v, ilp_rows, use_smem_v). Shows what callers get with no
              explicit knobs; it dispatches to the same kernel config as ws or
              ws4 per the picked ilp (the "sel ilp" column reports which).

ws and ws4 pin ilp AND use_smem_v=False explicitly so the ladder ws(ilp2) ->
ws4(ilp4) -> ws4_smemv(ilp4+smem_v) each isolates ONE knob; otherwise an unset
use_smem_v defaults to None and the heuristic turns it on at large batch
(work_units>1024), making ws4 == ws4_smemv there.

Reports per (N, T): wall time, tokens/s (= N*T / time), the tile_v the
work_units=N*HV heuristic picked, the heuristic's selected ilp ("sel ilp"),
speedups vs looped, and the **ws4-vs-ws** head-to-head (the ilp=4-vs-ilp=2 win).
Also cross-checks each route's output/state against the looped route (rel error)
as a sanity gate.

Pass --routes to restrict which of {loop,ws,ws4,ws4_smemv,ws_auto}
run (default: all).

A bit-for-bit determinism check (--determinism) re-runs a state-writeback kernel
many times from the same initial state and compares output + final state exactly,
to surface state-writeback races before any kernel change lands.

--bench-intermediate measures the Stage-D snapshot overhead: the ilp=4 ws kernel
with vs without the [N,T,HV,V,K] intermediate-state buffer (fire-and-forget GMEM
stores; expect ~1.0x). Opt-in so the large buffer isn't allocated in normal runs.

--sweep-config sweeps EVERY legal (tile_v, ilp_rows, use_smem_v) config for each
(HV, N, T) cell, times tokens/s, and marks the winner per work_units=N*HV — the
data to re-tune _select_mtp_config for KDA (its thresholds are currently inherited
from FlashInfer's GDN get_mtp_config). It sweeps HV in {32, 64} by default (32 =
real Kimi-Linear KDA GQA, 64 = FlashInfer GDN) so the same work_units reached from
two different (HV, N) splits can be cross-checked for a work_units-only optimum.
Pure config sweep — no intermediate buffer; correctness still cross-checked vs the
looped route. WARNING: every (N,T,HV,config) is a distinct JIT compile (N is in the
compile-cache key), so the full grid compiles ~1.3k kernels — expect a long run;
subset via --sweep-hvs/--sweep-ns/--sweep-ts. Writes a markdown report (winner table
+ work_units-invariance check + full per-cell grid).

Fairness notes:
  - Both routes reset their state buffer before each timed iteration; the reset
    copy_() runs outside the CUDA event window and is NOT counted.
  - The looped route's per-token tensors are pre-sliced ONCE outside the timed
    loop, so only the T kernel launches are timed (not python slicing) — this
    gives the baseline its best shot; the ws win is purely about fusion/occupancy.

Usage:
    python benchmarks/bench_kda_decode_mtp.py
    python benchmarks/bench_kda_decode_mtp.py --batch-sizes 1 4 16 64 256 --Ts 2 4 8
    python benchmarks/bench_kda_decode_mtp.py --H 16 --HV 64
    python benchmarks/bench_kda_decode_mtp.py --tile-v 32          # override heuristic
    python benchmarks/bench_kda_decode_mtp.py --determinism --det-iters 10000
    python benchmarks/bench_kda_decode_mtp.py --bench-intermediate --batch-sizes 64 256
    python benchmarks/bench_kda_decode_mtp.py --sweep-config                  # full KDA config sweep
    python benchmarks/bench_kda_decode_mtp.py --sweep-config --sweep-hvs 64 --sweep-ns 64 256 --sweep-ts 2 4
    python benchmarks/bench_kda_decode_mtp.py --routes loop ws4
    python benchmarks/bench_kda_decode_mtp.py --output

Note:
  - Restricted to K=128 and V=128 (kernel constraint TILE_K=128).
"""

import argparse
import os
import pathlib
import platform
import sys
from datetime import datetime

os.environ.setdefault("FLA_USE_FAST_OPS", os.getenv("CULA_USE_FAST_MATH", "1"))

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from benchmarks.utils import benchmark_cuda_fn, relative_rms_error_rel_max
from cula.kda import (
    kda_decode,
    kda_decode_mtp_ws,
)
from cula.ops.kda_decode_mtp import _select_mtp_config, _select_mtp_tile_v


# ──────────────────────────────────────────────────────────────────────
# Input generation (mirrors tests/test_kda_decode_mtp.py::make_inputs_mtp)
# ──────────────────────────────────────────────────────────────────────
def make_inputs_mtp(N, T, H, HV, K, V, device="cuda", seed=42):
    """Generate random MTP inputs with a T token dimension."""
    torch.manual_seed(seed)
    q = torch.randn(N, T, H, K, device=device, dtype=torch.bfloat16)
    k = torch.randn(N, T, H, K, device=device, dtype=torch.bfloat16)
    v = torch.randn(N, T, HV, V, device=device, dtype=torch.bfloat16)
    a = (torch.randn(N, T, HV, K, device=device, dtype=torch.float32) * 0.1).to(torch.bfloat16)
    b = torch.randn(N, T, HV, device=device, dtype=torch.bfloat16)
    A_log = -torch.rand(HV, device=device, dtype=torch.float32) * 2  # negative → decay ∈ (0,1)
    dt_bias = torch.randn(HV, K, device=device, dtype=torch.float32) * 0.1
    state = torch.randn(N, HV, V, K, device=device, dtype=torch.float32) * 0.01
    return q, k, v, a, b, A_log, dt_bias, state


def _build_routes(q, k, v, a, b, A_log, dt_bias, state, scale, tile_v):
    """Build the ws + looped callables, their state buffers, and setup fns.

    Returns a dict with keys: call_<route>, setup_<route>, state_<route> for each
    of ws/ws4/ws4_smemv/ws_auto/loop (the live buffers, mutated in-place).

    Every route runs at its SHIPPED production config: loop = kda_decode (its own
    default), ws/ws4/ws4_smemv/ws_auto = kda_decode_mtp_ws (opt_level=3 +
    fast_math, fixed internally).
    """
    N, T = q.shape[0], q.shape[1]
    device = q.device
    indices = torch.arange(N, device=device, dtype=torch.int32)

    state_init = state.clone().contiguous()  # (N, HV, V, K)

    # --- ws route (Route 1): one warp-specialized launch over T tokens ---
    state_ws = state_init.clone()

    def call_ws():
        return kda_decode_mtp_ws(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            initial_state_source=state_ws,
            initial_state_indices=indices,
            scale=scale,
            use_qk_l2norm_in_kernel=True,
            tile_v=tile_v,
            ilp_rows=2,  # pin ilp=2 so ws-vs-ws4 isn't muddied by the None default
            use_smem_v=False,  # pin off so ws/ws4/ws4_smemv each isolate ONE knob
        )

    def setup_ws():
        state_ws.copy_(state_init)

    # --- ws4 route (Route 1, ilp=4): same warp-spec kernel, 4-row ILP path ---
    # (fused steps 1+2 & 4+5, double accumulators, packed F32x2 FMA on SM100).
    # use_packed_fma=None auto-detects SM100. Only valid when (tile_v//4)%4==0;
    # run_config gates the call on that, so small-tile_v configs skip ws4.
    # use_smem_v PINNED False: ws4 is the no-smem_v ilp=4 leg so ws4_smemv/ws4
    # isolates the use_smem_v win. Without the pin, an unset use_smem_v defaults
    # to None -> the heuristic turns smem_v ON at large batch (work_units>1024,
    # tile_v=64), which would make ws4 identical to ws4_smemv there.
    state_ws4 = state_init.clone()

    def call_ws4():
        return kda_decode_mtp_ws(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            initial_state_source=state_ws4,
            initial_state_indices=indices,
            scale=scale,
            use_qk_l2norm_in_kernel=True,
            tile_v=tile_v,
            ilp_rows=4,
            use_smem_v=False,  # see comment above: pin off so ws4_smemv/ws4 is clean
        )

    def setup_ws4():
        state_ws4.copy_(state_init)

    # --- ws4_smemv route (Route 1, ilp=4 + Stage-C use_smem_v): same ilp=4 path,
    # but v is preloaded into SMEM and outputs are accumulated in SMEM for a
    # coalesced merged writeback. Head-to-head vs ws4 isolates the use_smem_v
    # win (write bandwidth), which is expected to show up at large batch /
    # tile_v=64. Gated on ws4_ok in run_config (same tile_v//4 %4 constraint). ---
    state_ws4_smemv = state_init.clone()

    def call_ws4_smemv():
        return kda_decode_mtp_ws(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            initial_state_source=state_ws4_smemv,
            initial_state_indices=indices,
            scale=scale,
            use_qk_l2norm_in_kernel=True,
            tile_v=tile_v,
            ilp_rows=4,
            use_smem_v=True,
        )

    def setup_ws4_smemv():
        state_ws4_smemv.copy_(state_init)

    # --- ws_auto route (Route 1, PRODUCTION DEFAULT): ilp_rows=None lets the
    # work_units heuristic (_select_mtp_config) pick (tile_v, ilp_rows). Shows
    # what callers get with no explicit knobs. A tile_v override (if any) still
    # applies; with ilp_rows=None the heuristic backstop keeps it always legal. ---
    state_ws_auto = state_init.clone()

    def call_ws_auto():
        return kda_decode_mtp_ws(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            initial_state_source=state_ws_auto,
            initial_state_indices=indices,
            scale=scale,
            use_qk_l2norm_in_kernel=True,
            tile_v=tile_v,  # None in heuristic mode -> kernel auto-selects tile_v
            ilp_rows=None,  # heuristic picks ilp (the whole point of this route)
        )

    def setup_ws_auto():
        state_ws_auto.copy_(state_init)

    # --- looped route: T single-token launches, state carried in-place ---
    # Pre-slice per-token tensors once (outside the timed loop) so only the T
    # kernel launches are counted.
    q_tok = [q[:, t].unsqueeze(1).contiguous() for t in range(T)]
    k_tok = [k[:, t].unsqueeze(1).contiguous() for t in range(T)]
    v_tok = [v[:, t].unsqueeze(1).contiguous() for t in range(T)]
    a_tok = [a[:, t].unsqueeze(1).contiguous() for t in range(T)]
    b_tok = [b[:, t].unsqueeze(1).contiguous() for t in range(T)]

    state_loop = state_init.clone()
    o_loop = torch.empty(N, T, v.shape[2], v.shape[3], device=device, dtype=torch.bfloat16)

    def call_loop():
        for t in range(T):
            o_t = kda_decode(
                A_log=A_log,
                dt_bias=dt_bias,
                q=q_tok[t],
                k=k_tok[t],
                v=v_tok[t],
                a=a_tok[t],
                b=b_tok[t],
                initial_state_source=state_loop,
                initial_state_indices=indices,
                scale=scale,
                use_qk_l2norm_in_kernel=True,
            )
            o_loop[:, t] = o_t.squeeze(1)
        return o_loop

    def setup_loop():
        state_loop.copy_(state_init)

    return {
        "call_ws": call_ws,
        "setup_ws": setup_ws,
        "call_ws4": call_ws4,
        "setup_ws4": setup_ws4,
        "call_ws4_smemv": call_ws4_smemv,
        "setup_ws4_smemv": setup_ws4_smemv,
        "call_ws_auto": call_ws_auto,
        "setup_ws_auto": setup_ws_auto,
        "call_loop": call_loop,
        "setup_loop": setup_loop,
        "state_ws": state_ws,
        "state_ws4": state_ws4,
        "state_ws4_smemv": state_ws4_smemv,
        "state_ws_auto": state_ws_auto,
        "state_loop": state_loop,
        "state_init": state_init,
    }


# ──────────────────────────────────────────────────────────────────────
# Timing one config
# ──────────────────────────────────────────────────────────────────────
def run_config(N, T, H, HV, K, V, tile_v_override, warmup, rep, ncu_mode, route_set):
    device = "cuda"
    scale = K**-0.5

    q, k, v, a, b, A_log, dt_bias, state = make_inputs_mtp(N, T, H, HV, K, V, device)
    tile_v = tile_v_override if tile_v_override is not None else _select_mtp_tile_v(N, HV, V, T)

    # What the production default (tile_v=None, ilp_rows=None) would pick. The
    # ws_auto route exercises exactly this; "sel ilp" surfaces it so we can
    # confirm mid/large batch hits ilp=4 without reading kernel logs.
    sel_tile_v, sel_ilp, sel_smem_v = _select_mtp_config(N, HV, V, T)

    routes = _build_routes(q, k, v, a, b, A_log, dt_bias, state, scale, tile_v_override)

    # ws4 (ilp=4) is only valid when each warp's rows_per_group=tile_v/4 is a
    # multiple of 4; skip it (n/a) on small tile_v rather than asserting. ws_auto
    # is always valid (ilp_rows=None lets the heuristic back off to ilp=2).
    ws4_ok = (tile_v // 4) % 4 == 0
    # Routes actually run this config, in display order. Drop ws4 / ws4_smemv
    # when ilp=4 is invalid for this tile_v.
    active = [
        r
        for r in ("loop", "ws", "ws_auto", "ws4", "ws4_smemv")
        if r in route_set and (r not in ("ws4", "ws4_smemv") or ws4_ok)
    ]
    corr_routes = [r for r in active if r != "loop"]

    # Correctness reference = looped single-token route (always run once, fresh
    # state). Each ws/ws4 route is cross-checked against it (rel error).
    nan2 = (float("nan"), float("nan"))
    corr = {}  # route name -> (out_rel_max, state_rel_max)
    with torch.no_grad():
        routes["setup_loop"]()
        o_loop = routes["call_loop"]().clone()
        state_loop_final = routes["state_loop"].clone()

        for name in corr_routes:
            routes[f"setup_{name}"]()
            o_r = routes[f"call_{name}"]().clone()
            st_r = routes[f"state_{name}"].clone()
            _, out_rel_max = relative_rms_error_rel_max(o_loop, o_r)
            _, state_rel_max = relative_rms_error_rel_max(state_loop_final, st_r)
            corr[name] = (out_rel_max, state_rel_max)

    w, r = (1, 1) if ncu_mode else (warmup, rep)

    times = {}
    with torch.no_grad():
        for name in active:
            times[name] = benchmark_cuda_fn(
                routes[f"call_{name}"], setup_fn=routes[f"setup_{name}"], warmup=w, rep=r
            )

    tokens = N * T

    def mtok(t):
        return tokens / t / 1e3 if (t is not None and t > 0) else float("nan")

    def speedup_vs_loop(t):
        tl = times.get("loop")
        return (tl / t) if (tl is not None and t is not None and t > 0) else float("nan")

    def ratio(t_num, t_den):
        return (t_num / t_den) if (t_num and t_den and t_den > 0) else float("nan")

    t_ws, t_ws4 = times.get("ws"), times.get("ws4")
    t_ws_auto = times.get("ws_auto")
    t_ws4_smemv = times.get("ws4_smemv")
    return {
        "N": N,
        "T": T,
        "H": H,
        "HV": HV,
        "tile_v": tile_v,
        "ws4_ok": ws4_ok,
        # Production-default selection (tile_v=None, ilp_rows=None).
        "sel_tile_v": sel_tile_v,
        "sel_ilp": sel_ilp,
        "sel_smem_v": sel_smem_v,
        "t_loop_ms": times.get("loop"),
        "t_ws_ms": t_ws,
        "t_ws4_ms": t_ws4,
        "t_ws4_smemv_ms": t_ws4_smemv,
        "t_ws_auto_ms": t_ws_auto,
        "loop_mtok_s": mtok(times.get("loop")),
        "ws_mtok_s": mtok(t_ws),
        "ws4_mtok_s": mtok(t_ws4),
        "ws4_smemv_mtok_s": mtok(t_ws4_smemv),
        "ws_auto_mtok_s": mtok(t_ws_auto),
        "ws_speedup": speedup_vs_loop(t_ws),  # vs looped
        "ws4_speedup": speedup_vs_loop(t_ws4),  # vs looped
        "ws4_smemv_speedup": speedup_vs_loop(t_ws4_smemv),  # vs looped
        "ws_auto_speedup": speedup_vs_loop(t_ws_auto),  # vs looped
        # ilp=4 vs ilp=2 (the "biggest win" measure): >1 -> ilp=4 faster.
        "ws4_vs_ws": ratio(t_ws, t_ws4),
        # Stage C: use_smem_v win over plain ilp=4. >1 -> smem_v faster (expected
        # at large batch / tile_v=64 from the coalesced merged writeback).
        "ws4_smemv_vs_ws4": ratio(t_ws4, t_ws4_smemv),
        "ws_out_rel_max": corr.get("ws", nan2)[0],
        "ws_state_rel_max": corr.get("ws", nan2)[1],
        "ws4_out_rel_max": corr.get("ws4", nan2)[0],
        "ws4_state_rel_max": corr.get("ws4", nan2)[1],
        "ws4_smemv_out_rel_max": corr.get("ws4_smemv", nan2)[0],
        "ws4_smemv_state_rel_max": corr.get("ws4_smemv", nan2)[1],
        "ws_auto_out_rel_max": corr.get("ws_auto", nan2)[0],
        "ws_auto_state_rel_max": corr.get("ws_auto", nan2)[1],
    }


# ──────────────────────────────────────────────────────────────────────
# Determinism check (bit-for-bit, surfaces state-writeback races)
# ──────────────────────────────────────────────────────────────────────
def run_determinism(N, T, H, HV, K, V, tile_v_override, det_iters, route):
    device = "cuda"
    scale = K**-0.5

    q, k, v, a, b, A_log, dt_bias, state = make_inputs_mtp(N, T, H, HV, K, V, device)
    tile_v = tile_v_override if tile_v_override is not None else _select_mtp_tile_v(N, HV, V, T)

    # ws4 / ws4_smemv (ilp=4) require (tile_v//4)%4==0; skip otherwise.
    if route in ("ws4", "ws4_smemv") and (tile_v // 4) % 4 != 0:
        return {"N": N, "T": T, "tile_v": tile_v, "iters": 0, "route": route,
                "passed": True, "first_bad": -1, "skipped": True}

    routes = _build_routes(q, k, v, a, b, A_log, dt_bias, state, scale, tile_v_override)

    call = routes[f"call_{route}"]
    setup = routes[f"setup_{route}"]
    state_buf = routes[f"state_{route}"]

    with torch.no_grad():
        setup()
        o_ref = call().clone()
        state_ref = state_buf.clone()

        first_bad = -1
        for i in range(det_iters):
            setup()
            o_i = call()
            same = torch.equal(o_i, o_ref) and torch.equal(state_buf, state_ref)
            if not same:
                first_bad = i
                break

    return {
        "N": N,
        "T": T,
        "tile_v": tile_v,
        "iters": det_iters,
        "route": route,
        "passed": first_bad < 0,
        "first_bad": first_bad,
        "skipped": False,
    }


# ──────────────────────────────────────────────────────────────────────
# Stage D: intermediate-snapshot overhead (fire-and-forget; expected small)
# ──────────────────────────────────────────────────────────────────────
def run_intermediate_overhead(N, T, H, HV, K, V, tile_v_override, warmup, rep):
    """Time the ilp=4 ws kernel with vs without the per-token intermediate-state
    snapshot, to measure the fire-and-forget GMEM-store overhead. Opt-in (the
    [N,T,HV,V,K] buffer is large) — only allocated here, never in the main sweep.
    """
    device = "cuda"
    scale = K**-0.5
    q, k, v, a, b, A_log, dt_bias, state = make_inputs_mtp(N, T, H, HV, K, V, device)
    tile_v = tile_v_override if tile_v_override is not None else _select_mtp_tile_v(N, HV, V, T)
    if (tile_v // 4) % 4 != 0:  # ilp=4 legality
        return {"N": N, "T": T, "tile_v": tile_v, "skipped": True,
                "t_base_ms": None, "t_inter_ms": None, "inter_overhead": float("nan")}

    state_init = state.clone().contiguous()
    indices = torch.arange(N, device=device, dtype=torch.int32)
    state_base = state_init.clone()
    state_inter = state_init.clone()
    inter_buf = torch.zeros(N, T, HV, V, K, device=device, dtype=torch.float32)

    def call_base():
        return kda_decode_mtp_ws(
            A_log=A_log, dt_bias=dt_bias, q=q, k=k, v=v, a=a, b=b,
            initial_state_source=state_base, initial_state_indices=indices,
            scale=scale, use_qk_l2norm_in_kernel=True, tile_v=tile_v, ilp_rows=4,
        )

    def call_inter():
        return kda_decode_mtp_ws(
            A_log=A_log, dt_bias=dt_bias, q=q, k=k, v=v, a=a, b=b,
            initial_state_source=state_inter, initial_state_indices=indices,
            scale=scale, use_qk_l2norm_in_kernel=True, tile_v=tile_v, ilp_rows=4,
            intermediate_states_buffer=inter_buf,
        )

    with torch.no_grad():
        t_base = benchmark_cuda_fn(
            call_base, setup_fn=lambda: state_base.copy_(state_init), warmup=warmup, rep=rep
        )
        t_inter = benchmark_cuda_fn(
            call_inter, setup_fn=lambda: state_inter.copy_(state_init), warmup=warmup, rep=rep
        )

    overhead = (t_inter / t_base) if (t_base and t_inter and t_base > 0) else float("nan")
    return {"N": N, "T": T, "tile_v": tile_v, "skipped": False,
            "t_base_ms": t_base, "t_inter_ms": t_inter, "inter_overhead": overhead}


# ──────────────────────────────────────────────────────────────────────
# Config sweep: time ALL legal (tile_v, ilp, use_smem_v) per (HV, N, T) cell,
# mark the winner per work_units=N*HV, to re-tune _select_mtp_config for KDA.
# ──────────────────────────────────────────────────────────────────────
_SWEEP_TILE_V_CHOICES = (8, 16, 32, 64)
_SWEEP_ILP_CHOICES = (2, 4)
_SWEEP_SMEM_V_CHOICES = (False, True)


def _legal_sweep_configs(V):
    """All (tile_v, ilp_rows, use_smem_v) the ws kernel accepts at this V.

    Mirrors ``kda_decode_mtp_ws``'s legality asserts EXACTLY: ``tile_v % 4 == 0``,
    ``V % tile_v == 0``, and ``(tile_v // 4) % ilp_rows == 0`` (so ilp=2 needs
    ``tile_v % 8 == 0``, ilp=4 needs ``tile_v % 16 == 0``). ``use_smem_v`` is
    orthogonal (works with any tile_v / ilp). For V=128 this yields 14 configs:
    tile_v=8 -> ilp=2 only (x2 use_smem_v); tile_v in {16,32,64} -> ilp in {2,4}
    (x2 use_smem_v each).
    """
    configs = []
    for tile_v in _SWEEP_TILE_V_CHOICES:
        if tile_v > V or V % tile_v != 0 or tile_v % 4 != 0:
            continue
        rows_per_group = tile_v // 4
        for ilp in _SWEEP_ILP_CHOICES:
            if rows_per_group % ilp != 0:
                continue
            for use_smem_v in _SWEEP_SMEM_V_CHOICES:
                configs.append((tile_v, ilp, use_smem_v))
    return configs


def _fmt_cfg(tile_v, ilp, use_smem_v):
    """Compact 'tile_v/ilp/smem_v' config label, e.g. '64/4/T'."""
    return f"{tile_v}/{ilp}/{'T' if use_smem_v else 'F'}"


# Shared header for the live-progress + final winner tables (run_sweep).
_SWEEP_HDR = (
    f"{'wu':>7} | {'HV':>3} | {'N':>5} | {'T':>2} | {'winner':>9} | "
    f"{'Mtok/s':>8} | {'heur':>9} | {'h Mtok/s':>8} | {'win/heur':>8} | {'match':>5}"
)


def _sweep_row(cell):
    """One winner-table row for a cell (columns match _SWEEP_HDR)."""
    wu, HV, N, T = cell["work_units"], cell["HV"], cell["N"], cell["T"]
    if cell["winner"] is None:
        return (f"{wu:>7} | {HV:>3} | {N:>5} | {T:>2} | {'ERROR':>9} | "
                f"{'n/a':>8} | {_fmt_cfg(*cell['heur']):>9} | {'n/a':>8} | "
                f"{'n/a':>8} | {'n/a':>5}")
    w = cell["winner"]
    hr = cell["heur_result"]
    win_cfg = _fmt_cfg(w["tile_v"], w["ilp"], w["use_smem_v"])
    win_heur = (hr["t_ms"] / w["t_ms"]) if (hr and hr["t_ms"] and w["t_ms"]) else float("nan")
    match = "=" if (w["tile_v"], w["ilp"], w["use_smem_v"]) == cell["heur"] else "DIFF"
    h_mtok = hr["mtok_s"] if hr else float("nan")
    return (f"{wu:>7} | {HV:>3} | {N:>5} | {T:>2} | {win_cfg:>9} | "
            f"{w['mtok_s']:>8.2f} | {_fmt_cfg(*cell['heur']):>9} | {_fmt(h_mtok, '.2f'):>8} | "
            f"{_fmt(win_heur, '.2f'):>7}x | {match:>5}")


def run_sweep_config(N, T, H, HV, K, V, warmup, rep):
    """Time EVERY legal (tile_v, ilp_rows, use_smem_v) for one (HV, N, T) cell
    and pick the empirical winner (max tokens/s).

    Pure config sweep: each config is passed to ``kda_decode_mtp_ws`` explicitly
    (all three knobs set -> the kernel skips the heuristic), with no intermediate
    buffer; ``use_packed_fma`` stays auto (SM100 -> packed FMA for ilp=4), matching
    the production path. Every config is also cross-checked for correctness against
    the looped single-token route (rel error vs out + final state) — the same gate
    the main sweep uses. Returns a cell dict for run_sweep / the markdown report.
    A failure in any single config is captured (``error``) and skipped rather than
    aborting the (multi-hour) sweep; a cell-level failure sets ``cell['error']``.
    """
    device = "cuda"
    scale = K**-0.5
    work_units = N * HV
    tokens = N * T
    heur = _select_mtp_config(N, HV, V, T)  # (tile_v, ilp_rows, use_smem_v)

    cell = {
        "HV": HV, "N": N, "T": T, "work_units": work_units, "tokens": tokens,
        "heur": heur, "configs": [], "winner": None, "heur_result": None,
        "error": None,
    }

    try:
        q, k, v, a, b, A_log, dt_bias, state = make_inputs_mtp(N, T, H, HV, K, V, device)
        indices = torch.arange(N, device=device, dtype=torch.int32)
        state_init = state.clone().contiguous()

        # --- looped single-token reference (correctness oracle), computed once.
        # Pre-slice per-token tensors so the loop matches _build_routes' fair path.
        state_loop = state_init.clone()
        q_tok = [q[:, t].unsqueeze(1).contiguous() for t in range(T)]
        k_tok = [k[:, t].unsqueeze(1).contiguous() for t in range(T)]
        v_tok = [v[:, t].unsqueeze(1).contiguous() for t in range(T)]
        a_tok = [a[:, t].unsqueeze(1).contiguous() for t in range(T)]
        b_tok = [b[:, t].unsqueeze(1).contiguous() for t in range(T)]
        o_loop = torch.empty(N, T, HV, V, device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            for t in range(T):
                o_t = kda_decode(
                    A_log=A_log, dt_bias=dt_bias, q=q_tok[t], k=k_tok[t], v=v_tok[t],
                    a=a_tok[t], b=b_tok[t], initial_state_source=state_loop,
                    initial_state_indices=indices, scale=scale,
                    use_qk_l2norm_in_kernel=True,
                )
                o_loop[:, t] = o_t.squeeze(1)
        state_loop_final = state_loop.clone()
        del state_loop  # free before the config loop (state is the big buffer)

        # One reusable mutable state buffer shared by every config (reset each run),
        # so the cell holds at most ~4 state copies regardless of how many configs.
        state_work = state_init.clone()

        def reset():
            state_work.copy_(state_init)

        def make_call(tile_v, ilp, use_smem_v):
            def _call():
                return kda_decode_mtp_ws(
                    A_log=A_log, dt_bias=dt_bias, q=q, k=k, v=v, a=a, b=b,
                    initial_state_source=state_work, initial_state_indices=indices,
                    scale=scale, use_qk_l2norm_in_kernel=True,
                    tile_v=tile_v, ilp_rows=ilp, use_smem_v=use_smem_v,
                    # use_packed_fma left None -> auto-detect SM100 (production path)
                )
            return _call

        with torch.no_grad():
            for tile_v, ilp, use_smem_v in _legal_sweep_configs(V):
                rec = {
                    "tile_v": tile_v, "ilp": ilp, "use_smem_v": use_smem_v,
                    "t_ms": None, "mtok_s": float("nan"),
                    "out_rel_max": float("nan"), "state_rel_max": float("nan"),
                    "is_heur": (tile_v, ilp, use_smem_v) == heur, "error": None,
                }
                try:
                    call = make_call(tile_v, ilp, use_smem_v)
                    reset()
                    o_r = call().clone()  # first call also warms the JIT compile
                    st_r = state_work.clone()
                    _, rec["out_rel_max"] = relative_rms_error_rel_max(o_loop, o_r)
                    _, rec["state_rel_max"] = relative_rms_error_rel_max(state_loop_final, st_r)
                    del o_r, st_r
                    t_ms = benchmark_cuda_fn(call, setup_fn=reset, warmup=warmup, rep=rep)
                    rec["t_ms"] = t_ms
                    rec["mtok_s"] = tokens / t_ms / 1e3 if (t_ms and t_ms > 0) else float("nan")
                except Exception as e:  # noqa: BLE001 — keep the sweep going
                    rec["error"] = f"{type(e).__name__}: {e}"
                cell["configs"].append(rec)
                if rec["is_heur"]:
                    cell["heur_result"] = rec

        valid = [c for c in cell["configs"] if c["error"] is None and c["t_ms"] and c["t_ms"] > 0]
        cell["winner"] = min(valid, key=lambda c: c["t_ms"]) if valid else None
    except Exception as e:  # noqa: BLE001 — one bad cell shouldn't kill the sweep
        cell["error"] = f"{type(e).__name__}: {e}"

    return cell


def run_sweep(args, gpu_name):
    """Orchestrate the full config sweep: every (HV, N, T) cell x all legal
    configs, winner per cell, a work_units-invariance check, and a markdown dump
    to re-tune _select_mtp_config. Always writes the markdown (the full grid is
    the deliverable and the sweep is expensive)."""
    hvs, ns, ts = args.sweep_hvs, args.sweep_ns, args.sweep_ts
    n_cfg = len(_legal_sweep_configs(args.V))
    n_cells = len(hvs) * len(ns) * len(ts)
    total = n_cells * n_cfg

    print(f"Config sweep: H={args.H}, K={args.K}, V={args.V} | HV={hvs} x N={ns} x T={ts}")
    print(f"  {n_cells} cells x {n_cfg} legal configs = {total} timed configs "
          f"(work_units = N*HV is the heuristic key).")
    print("  NOTE: each (N,T,HV,tile_v,ilp,use_smem_v) is a DISTINCT JIT compile on "
          "first use (N is in the compile-cache key), so the first touch of every "
          "config compiles -> expect a LONG run. Subset via --sweep-hvs/--sweep-ns/"
          "--sweep-ts for a quick look; drop N=2048 first if you hit OOM.")
    print(f"  Timing: warmup={args.warmup}, rep={args.rep}.")
    print()
    print(f"{'progress':>10} | {_SWEEP_HDR}")
    print("-" * (13 + len(_SWEEP_HDR)))

    cells = []
    done = 0
    for HV in hvs:
        for T in ts:
            for N in ns:
                done += 1
                cell = run_sweep_config(N, T, args.H, HV, args.K, args.V, args.warmup, args.rep)
                cells.append(cell)
                print(f"{f'[{done}/{n_cells}]':>10} | {_sweep_row(cell)}")
                if cell["error"] is not None:
                    print(f"           ! cell error: {cell['error']}")
    print()

    # --- Winner table, sorted so equal work_units (across the HV split) sit
    # adjacent and the winner-per-work_units trend reads top-to-bottom.
    print("=== Winner per cell (sorted by work_units, T, HV) ===")
    print(_SWEEP_HDR)
    print("-" * len(_SWEEP_HDR))
    for c in sorted(cells, key=lambda c: (c["work_units"], c["T"], c["HV"])):
        print(_sweep_row(c))
    print()

    _print_sweep_invariance(cells)
    _print_sweep_headline(cells)

    # Persist (always): the full per-cell grid is the re-tuning deliverable.
    from benchmarks.bench_kda_decode import normalize_gpu_type

    if args.output and args.output != "__AUTO__":
        output_path = pathlib.Path(args.output)
    else:
        output_path = pathlib.Path(
            f"BENCHMARK_KDA_DECODE_MTP_SWEEP_{normalize_gpu_type(gpu_name)}.md"
        )
    write_sweep_markdown_report(args, gpu_name, cells, output_path)
    print(f"Full grid + winner + invariance tables written to: {output_path.resolve()}")
    return cells


def _invariance_groups(cells):
    """Group winners by (work_units, T); return [( (wu,T), [cells] )] for the
    groups reached from >1 (HV, N) split, sorted by (wu, T)."""
    from collections import defaultdict

    groups = defaultdict(list)
    for c in cells:
        if c["winner"] is not None:
            groups[(c["work_units"], c["T"])].append(c)
    return [(kk, v) for kk, v in sorted(groups.items()) if len(v) >= 2]


def _winner_triple(cell):
    w = cell["winner"]
    return (w["tile_v"], w["ilp"], w["use_smem_v"])


def _print_sweep_invariance(cells):
    """Same work_units & T from different (HV, N) splits — do winners agree?"""
    multi = _invariance_groups(cells)
    print("=== Work_units invariance (same work_units & T, different HV/N split) ===")
    if not multi:
        print("  (no work_units reached from >1 HV/N split in this grid)")
        print()
        return
    print("  >1 split reaches these work_units; CONSISTENT => optimum is purely work_units-keyed.")
    for (wu, T), members in multi:
        wins = {_winner_triple(g) for g in members}
        verdict = "CONSISTENT" if len(wins) == 1 else "MIXED"
        splits = ", ".join(
            f"{g['HV']}:{g['N']}->{_fmt_cfg(*_winner_triple(g))}"
            for g in sorted(members, key=lambda g: g["HV"])
        )
        print(f"  wu={wu:>7} T={T}: {splits}   [{verdict}]")
    print()


def _print_sweep_headline(cells):
    """How much is the current GDN-inherited heuristic leaving on the table?"""
    ok = [c for c in cells if c["winner"] is not None]
    if not ok:
        print("=== Headline: no successful cells ===")
        print()
        return
    hits = sum(1 for c in ok if _winner_triple(c) == c["heur"])
    ratios = [
        c["heur_result"]["t_ms"] / c["winner"]["t_ms"]
        for c in ok
        if c["heur_result"] and c["heur_result"]["t_ms"] and c["winner"]["t_ms"]
    ]
    print("=== Headline (drives Task 2 re-tune of _select_mtp_config) ===")
    print(f"  heuristic picks the winner in {hits}/{len(ok)} cells "
          f"({100.0 * hits / len(ok):.0f}%).")
    if ratios:
        avg = sum(ratios) / len(ratios)
        print(f"  win/heur speedup (gain available from re-tuning): "
              f"avg={avg:.3f}x, min={min(ratios):.3f}x, max={max(ratios):.3f}x.")
    print()


# ──────────────────────────────────────────────────────────────────────
# Markdown report
# ──────────────────────────────────────────────────────────────────────
def _fmt(x, spec):
    """Format a float that may be None/nan as 'n/a'."""
    if x is None or (isinstance(x, float) and x != x):  # None or NaN
        return "n/a"
    return format(x, spec)


def write_markdown_report(args, gpu_name, results, output_path):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def summary(vals):
        vals = [v for v in vals if v is not None and v == v]
        if not vals:
            return "n/a"
        return f"avg={sum(vals) / len(vals):.2f}x, min={min(vals):.2f}x, max={max(vals):.2f}x"

    lines = []
    lines.append("# Benchmark Results - KDA MTP Decode (ws vs looped single-token)")
    lines.append("")
    lines.append(f"> Auto-generated by `benchmarks/bench_kda_decode_mtp.py` on {now}.")
    lines.append("")
    lines.append(
        f"> **GPU:** {gpu_name}  |  **CUDA:** {torch.version.cuda or 'unknown'}  |  "
        f"**PyTorch:** {torch.__version__}  |  **Python:** {platform.python_version()}"
    )
    lines.append("")
    lines.append(
        f"> Setting: H={args.H}, HV={args.HV}, K={args.K}, V={args.V}; routes={args.routes}. "
        "ws = kda_decode_mtp_ws ilp=2 "
        "(FlashInfer warp-spec); ws4 = kda_decode_mtp_ws ilp=4 (fused steps + packed FMA, "
        "n/a where tile_v//4 not %4); ws4sv = ws4 + use_smem_v (Stage C: SMEM v preload + merged "
        "writeback); ws_auto = kda_decode_mtp_ws ilp_rows=None (production default, ilp picked by the "
        "work_units heuristic — 'sel ilp' column); looped = T× single-token kda_decode."
    )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- ws-vs-looped speedup:    {summary([r['ws_speedup'] for r in results])}")
    lines.append(f"- ws4-vs-looped speedup:   {summary([r['ws4_speedup'] for r in results])}")
    lines.append(f"- ws_auto-vs-looped speedup: {summary([r['ws_auto_speedup'] for r in results])}  (production default; ilp picked by heuristic — see 'sel ilp')")
    lines.append(f"- **ws4-vs-ws (ilp4/ilp2): {summary([r['ws4_vs_ws'] for r in results])}**  (>1 → ilp=4 wins)")
    lines.append(f"- **ws4_smemv-vs-ws4 (Stage C use_smem_v): {summary([r['ws4_smemv_vs_ws4'] for r in results])}**  (>1 → use_smem_v wins; expect at large batch / tile_v=64)")
    lines.append(f"- Batch sizes (N): {args.batch_sizes}")
    lines.append(f"- T values: {args.Ts}")
    lines.append(f"- tile_v: {'heuristic (work_units=N*HV)' if args.tile_v is None else args.tile_v}")
    lines.append(f"- Timing: warmup={args.warmup}, rep={args.rep}")
    lines.append("")
    lines.append("## Performance")
    lines.append("")
    lines.append("| N | T | tile_v | sel ilp | loop ms | ws ms | ws4 ms | ws4sv ms | ws/loop | ws4/loop | ws4/ws | ws4sv/ws4 | ws out rmax | ws4 out rmax | ws4sv out rmax |")
    lines.append("|--:|--:|-------:|--------:|--------:|------:|-------:|--------:|--------:|---------:|-------:|----------:|------------:|-------------:|---------------:|")
    for r in results:
        lines.append(
            f"| {r['N']} | {r['T']} | {r['tile_v']} | {r['sel_ilp']} | {_fmt(r['t_loop_ms'], '.4f')} | "
            f"{_fmt(r['t_ws_ms'], '.4f')} | {_fmt(r['t_ws4_ms'], '.4f')} | {_fmt(r['t_ws4_smemv_ms'], '.4f')} | "
            f"{_fmt(r['ws_speedup'], '.2f')}x | {_fmt(r['ws4_speedup'], '.2f')}x | "
            f"**{_fmt(r['ws4_vs_ws'], '.2f')}x** | **{_fmt(r['ws4_smemv_vs_ws4'], '.2f')}x** | "
            f"{_fmt(r['ws_out_rel_max'], '.2e')} | {_fmt(r['ws4_out_rel_max'], '.2e')} | {_fmt(r['ws4_smemv_out_rel_max'], '.2e')} |"
        )
    lines.append("")
    lines.append("## Reproduce")
    lines.append("")
    lines.append("```bash")
    lines.append("python benchmarks/bench_kda_decode_mtp.py")
    lines.append("```")
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_sweep_markdown_report(args, gpu_name, cells, output_path):
    """Write the --sweep-config report: winner-per-cell table (sorted by
    work_units), work_units-invariance check, and the full per-cell grid (every
    config's tokens/s + correctness). This is the artifact for re-tuning
    _select_mtp_config and for the PR doc's config-heuristic section."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    by_wu = sorted(cells, key=lambda c: (c["work_units"], c["T"], c["HV"]))

    lines = []
    lines.append("# Config Sweep — KDA MTP Decode (warp-spec `kda_decode_mtp_ws`)")
    lines.append("")
    lines.append(f"> Auto-generated by `benchmarks/bench_kda_decode_mtp.py --sweep-config` on {now}.")
    lines.append("")
    lines.append(
        f"> **GPU:** {gpu_name}  |  **CUDA:** {torch.version.cuda or 'unknown'}  |  "
        f"**PyTorch:** {torch.__version__}  |  **Python:** {platform.python_version()}"
    )
    lines.append("")
    lines.append(
        f"> Grid: H={args.H}, K={args.K}, V={args.V}; HV={args.sweep_hvs}, N={args.sweep_ns}, "
        f"T={args.sweep_ts}. Per (HV, N, T) cell, ALL legal (tile_v∈{{8,16,32,64}}, "
        f"ilp∈{{2,4}}, use_smem_v∈{{F,T}}) configs are timed (tokens/s = N*T/time); ilp=4 "
        f"requires tile_v%16==0; use_smem_v is orthogonal. `use_packed_fma` auto-detects "
        f"(SM100 → packed FMA for ilp=4). Winner = max tokens/s. **heur** = what "
        f"`_select_mtp_config` currently picks (FlashInfer-GDN-inherited thresholds). "
        f"Correctness for every config is cross-checked vs the looped single-token route. "
        f"Timing: warmup={args.warmup}, rep={args.rep}."
    )
    lines.append("")

    # --- Headline ---
    ok = [c for c in cells if c["winner"] is not None]
    hits = sum(1 for c in ok if _winner_triple(c) == c["heur"])
    ratios = [
        c["heur_result"]["t_ms"] / c["winner"]["t_ms"]
        for c in ok
        if c["heur_result"] and c["heur_result"]["t_ms"] and c["winner"]["t_ms"]
    ]
    lines.append("## Headline")
    lines.append("")
    if ok:
        lines.append(f"- Heuristic picks the winner in **{hits}/{len(ok)}** cells "
                     f"({100.0 * hits / len(ok):.0f}%).")
    if ratios:
        avg = sum(ratios) / len(ratios)
        lines.append(f"- win/heur speedup (gain available from re-tuning): "
                     f"**avg={avg:.3f}x, min={min(ratios):.3f}x, max={max(ratios):.3f}x**.")
    if len(ok) != len(cells):
        lines.append(f"- ⚠️ {len(cells) - len(ok)} cell(s) failed (see grid for the error).")
    lines.append("")

    # --- Winner table ---
    lines.append("## Winner per cell (sorted by work_units, T, HV)")
    lines.append("")
    lines.append("| work_units | HV | N | T | winner (tv/ilp/sv) | win Mtok/s | win ms | "
                 "heuristic (tv/ilp/sv) | heur Mtok/s | win/heur | match |")
    lines.append("|--:|--:|--:|--:|:--|--:|--:|:--|--:|--:|:--|")
    for c in by_wu:
        heur_cfg = _fmt_cfg(*c["heur"])
        if c["winner"] is None:
            lines.append(f"| {c['work_units']} | {c['HV']} | {c['N']} | {c['T']} | ERROR | "
                         f"n/a | n/a | {heur_cfg} | n/a | n/a | n/a |")
            continue
        w = c["winner"]
        hr = c["heur_result"]
        win_heur = (hr["t_ms"] / w["t_ms"]) if (hr and hr["t_ms"] and w["t_ms"]) else float("nan")
        match = "=" if _winner_triple(c) == c["heur"] else "**DIFF**"
        lines.append(
            f"| {c['work_units']} | {c['HV']} | {c['N']} | {c['T']} | "
            f"`{_fmt_cfg(*_winner_triple(c))}` | {w['mtok_s']:.2f} | {_fmt(w['t_ms'], '.4f')} | "
            f"`{heur_cfg}` | {_fmt(hr['mtok_s'] if hr else float('nan'), '.2f')} | "
            f"{_fmt(win_heur, '.2f')}x | {match} |"
        )
    lines.append("")

    # --- Work_units invariance ---
    lines.append("## Work_units invariance (same work_units & T, different HV/N split)")
    lines.append("")
    lines.append("> Tests whether the optimal config is purely work_units-keyed: for each "
                 "(work_units, T) reached from >1 (HV, N) split, do the winners agree? A "
                 "**MIXED** row means the optimum also depends on the N/HV split and a "
                 "work_units-only threshold table can't fully capture it.")
    lines.append("")
    multi = _invariance_groups(cells)
    if not multi:
        lines.append("_(no work_units reached from >1 HV/N split in this grid)_")
    else:
        lines.append("| work_units | T | splits (HV:N → winner) | verdict |")
        lines.append("|--:|--:|:--|:--|")
        for (wu, T), members in multi:
            wins = {_winner_triple(g) for g in members}
            verdict = "CONSISTENT" if len(wins) == 1 else "**MIXED**"
            splits = ", ".join(
                f"{g['HV']}:{g['N']}→`{_fmt_cfg(*_winner_triple(g))}`"
                for g in sorted(members, key=lambda g: g["HV"])
            )
            lines.append(f"| {wu} | {T} | {splits} | {verdict} |")
    lines.append("")

    # --- Full per-cell grid ---
    lines.append("## Full grid (every config per cell)")
    lines.append("")
    lines.append("> `★win` marks the per-cell winner; `H` marks the current heuristic pick.")
    for c in by_wu:
        lines.append("")
        lines.append(f"### work_units={c['work_units']} — HV={c['HV']}, N={c['N']}, T={c['T']} "
                     f"(heuristic `{_fmt_cfg(*c['heur'])}`)")
        if c["error"] is not None:
            lines.append("")
            lines.append(f"**CELL ERROR:** {c['error']}")
            continue
        lines.append("")
        lines.append("| tile_v | ilp | smem_v | time ms | Mtok/s | out rmax | state rmax | mark |")
        lines.append("|--:|--:|:--:|--:|--:|--:|--:|:--|")
        ordered = sorted(
            c["configs"],
            key=lambda r: (float("inf") if not (r["t_ms"] and r["t_ms"] > 0) else r["t_ms"]),
        )
        for rec in ordered:
            marks = []
            if c["winner"] is rec:
                marks.append("★win")
            if rec["is_heur"]:
                marks.append("H")
            if rec["error"]:
                marks.append(f"ERR({rec['error']})")
            lines.append(
                f"| {rec['tile_v']} | {rec['ilp']} | {'T' if rec['use_smem_v'] else 'F'} | "
                f"{_fmt(rec['t_ms'], '.4f')} | {_fmt(rec['mtok_s'], '.2f')} | "
                f"{_fmt(rec['out_rel_max'], '.2e')} | {_fmt(rec['state_rel_max'], '.2e')} | "
                f"{' '.join(marks)} |"
            )
    lines.append("")
    lines.append("## Reproduce")
    lines.append("")
    lines.append("```bash")
    lines.append(
        "python benchmarks/bench_kda_decode_mtp.py --sweep-config "
        f"--H {args.H} --sweep-hvs {' '.join(map(str, args.sweep_hvs))} "
        f"--sweep-ns {' '.join(map(str, args.sweep_ns))} "
        f"--sweep-ts {' '.join(map(str, args.sweep_ts))} --output"
    )
    lines.append("```")
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def build_parser():
    parser = argparse.ArgumentParser(description="Benchmark KDA MTP decode: ws vs T× single-token")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4, 16, 64, 256])
    parser.add_argument("--Ts", nargs="+", type=int, default=[2, 4], help="MTP token counts to benchmark")
    parser.add_argument("--H", type=int, default=16, help="Q/K head count")
    parser.add_argument("--HV", type=int, default=64, help="V head count (GVA); work_units = N*HV")
    parser.add_argument("--K", type=int, default=128, help="Head dim K (only 128 supported)")
    parser.add_argument("--V", type=int, default=128, help="Head dim V (only 128 supported)")
    parser.add_argument("--tile-v", type=int, default=None, help="Override tile_v; default uses the work_units heuristic")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--rep", type=int, default=200)
    parser.add_argument("--ncu", action="store_true", help="NCU mode: warmup=1, rep=1")
    parser.add_argument(
        "--routes",
        nargs="+",
        choices=["loop", "ws", "ws4", "ws4_smemv", "ws_auto"],
        default=["loop", "ws", "ws_auto", "ws4", "ws4_smemv"],
        help="Which routes to run/time. loop=T× single-token, "
        "ws=kda_decode_mtp_ws ilp=2, ws4=kda_decode_mtp_ws ilp=4 (skipped when tile_v//4 not %4), "
        "ws_auto=kda_decode_mtp_ws ilp_rows=None (production default; heuristic picks ilp).",
    )
    parser.add_argument("--determinism", action="store_true", help="Run bit-for-bit determinism check instead of timing")
    parser.add_argument("--det-iters", type=int, default=10000, help="Determinism check repetitions per config")
    parser.add_argument(
        "--bench-intermediate",
        action="store_true",
        help="Measure Stage-D intermediate-snapshot overhead (ilp=4 ws with vs without "
             "the [N,T,HV,V,K] snapshot buffer) instead of the main route sweep",
    )
    parser.add_argument(
        "--sweep-config",
        action="store_true",
        help="Sweep ALL legal (tile_v, ilp, use_smem_v) per (HV, N, T) cell and mark the "
             "winner per work_units=N*HV — the data to re-tune _select_mtp_config for KDA. "
             "Uses --sweep-hvs/--sweep-ns/--sweep-ts (not --batch-sizes/--Ts/--HV); ignores "
             "--tile-v (it enumerates tile_v). Writes a markdown report.",
    )
    parser.add_argument(
        "--sweep-hvs", nargs="+", type=int, default=[32, 64],
        help="HV values to sweep with --sweep-config (default 32=Kimi-Linear KDA GQA, "
             "64=FlashInfer GDN).",
    )
    parser.add_argument(
        "--sweep-ns", nargs="+", type=int,
        default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048],
        help="N (batch) values to sweep with --sweep-config.",
    )
    parser.add_argument(
        "--sweep-ts", nargs="+", type=int, default=[2, 3, 4, 6],
        help="T (MTP token counts) to sweep with --sweep-config.",
    )
    parser.add_argument(
        "--output",
        nargs="?",
        const="__AUTO__",
        default=None,
        help="Write markdown report. Omit value to use BENCHMARK_KDA_DECODE_MTP_<GPU>.md.",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.K != 128 or args.V != 128:
        raise ValueError(f"bench_kda_decode_mtp.py only supports K=128 and V=128, got K={args.K}, V={args.V}")

    gpu_name = torch.cuda.get_device_name(0)
    route_set = set(args.routes)

    # Every route runs at its shipped production config (loop=kda_decode default,
    # ws=kda_decode_mtp_ws opt_level=3 + fast_math, fixed internally).
    print(f"GPU: {gpu_name}")
    print(f"Config: H={args.H}, HV={args.HV}, K={args.K}, V={args.V}, "
          f"tile_v={'heuristic' if args.tile_v is None else args.tile_v}, routes={args.routes}")
    print()

    if args.determinism:
        det_routes = [r for r in args.routes if r in ("ws", "ws4", "ws4_smemv", "ws_auto")]
        if not det_routes:
            print("No state-writeback route selected (--routes must include ws, ws4, ws4_smemv, and/or ws_auto for --determinism).")
            return True
        hdr = f"{'route':>6} | {'N':>5} | {'T':>3} | {'tile_v':>6} | {'iters':>8} | {'result':>8}"
        print(hdr)
        print("-" * len(hdr))
        all_ok = True
        for route in det_routes:
            for T in args.Ts:
                for N in args.batch_sizes:
                    res = run_determinism(N, T, args.H, args.HV, args.K, args.V, args.tile_v, args.det_iters, route)
                    if res.get("skipped"):
                        tag = "SKIP"
                    elif res["passed"]:
                        tag = "PASS"
                    else:
                        tag = f"FAIL@{res['first_bad']}"
                    all_ok = all_ok and res["passed"]
                    print(f"{res['route']:>6} | {res['N']:5d} | {res['T']:3d} | {res['tile_v']:6d} | {res['iters']:8d} | {tag:>8}")
        print()
        print("ALL DETERMINISTIC" if all_ok else "NON-DETERMINISM DETECTED — investigate state writeback race")
        return all_ok

    if args.sweep_config:
        if args.tile_v is not None:
            print("NOTE: --tile-v is ignored in --sweep-config mode (the sweep enumerates tile_v).\n")
        run_sweep(args, gpu_name)
        return True

    if args.bench_intermediate:
        hdr = (
            f"{'N':>5} | {'T':>3} | {'tile_v':>6} | {'ws4 ms':>9} | {'ws4+inter ms':>12} | "
            f"{'inter overhead':>14}"
        )
        print(hdr)
        print("-" * len(hdr))
        for T in args.Ts:
            for N in args.batch_sizes:
                res = run_intermediate_overhead(
                    N, T, args.H, args.HV, args.K, args.V, args.tile_v, args.warmup, args.rep
                )
                if res["skipped"]:
                    print(f"{res['N']:5d} | {res['T']:3d} | {res['tile_v']:6d} | "
                          f"{'n/a (ilp=4 invalid)':>39}")
                    continue
                print(f"{res['N']:5d} | {res['T']:3d} | {res['tile_v']:6d} | "
                      f"{_fmt(res['t_base_ms'], '.4f'):>9} | {_fmt(res['t_inter_ms'], '.4f'):>12} | "
                      f"{_fmt(res['inter_overhead'], '.3f'):>13}x")
        print()
        print("Stage-D snapshot overhead = ws4+inter / ws4 (fire-and-forget; expect ~1.0x).")
        return True

    # Dynamic head-to-head table: every column follows the selected --routes, so
    # e.g. `--routes loop ws4` drops the ws/ws4sv/wsAuto ms
    # columns AND every comparison that involves them (an h2h shows only when BOTH
    # of its routes are selected; a route's vs-loop speedup + out-rmax show only
    # when that route is selected). One ms col + one /loop speedup + one out-rmax
    # per selected route; the cross-route ratios are added only where applicable.
    # (route_key, t_key, speedup_key, display)
    ROUTE_COLS = [
        ("loop", "t_loop_ms", None, "loop"),
        ("ws", "t_ws_ms", "ws_speedup", "ws"),
        ("ws_auto", "t_ws_auto_ms", "ws_auto_speedup", "wsAuto"),
        ("ws4", "t_ws4_ms", "ws4_speedup", "ws4"),
        ("ws4_smemv", "t_ws4_smemv_ms", "ws4_smemv_speedup", "ws4sv"),
    ]
    # (result_key, label, routes_required)
    H2H = [
        ("ws4_vs_ws", "ws4/ws", ("ws4", "ws")),
        ("ws4_smemv_vs_ws4", "ws4sv/ws4", ("ws4_smemv", "ws4")),
    ]
    sel_cols = [c for c in ROUTE_COLS if c[0] in route_set]
    sel_speed = [c for c in sel_cols if c[2] is not None]  # speedup vs loop
    sel_h2h = [h for h in H2H if all(rr in route_set for rr in h[2])]
    sel_corr = [c for c in sel_cols if c[0] != "loop"]  # out rmax per non-loop route

    head = [f"{'N':>5}", f"{'T':>3}", f"{'tile_v':>6}", f"{'sel ilp':>7}"]
    head += [f"{disp + ' ms':>10}" for (_, _, _, disp) in sel_cols]
    head += [f"{disp + '/loop':>10}" for (_, _, _, disp) in sel_speed]
    head += [f"{label:>9}" for (_, label, _) in sel_h2h]
    head += [f"{disp + ' out rmax':>13}" for (_, _, _, disp) in sel_corr]
    hdr = " | ".join(head)
    print(hdr)
    print("-" * len(hdr))

    results = []
    for T in args.Ts:
        for N in args.batch_sizes:
            res = run_config(N, T, args.H, args.HV, args.K, args.V, args.tile_v, args.warmup, args.rep, args.ncu, route_set)
            results.append(res)
            row = [f"{res['N']:5d}", f"{res['T']:3d}", f"{res['tile_v']:6d}", f"{res['sel_ilp']:7d}"]
            row += [f"{_fmt(res[tk], '.4f'):>10}" for (_, tk, _, _) in sel_cols]
            row += [f"{_fmt(res[sk], '.2f'):>9}x" for (_, _, sk, _) in sel_speed]
            row += [f"{_fmt(res[hk], '.2f'):>8}x" for (hk, _, _) in sel_h2h]
            row += [f"{_fmt(res[f'{rk}_out_rel_max'], '.2e'):>13}" for (rk, _, _, _) in sel_corr]
            print(" | ".join(row))
    print()

    if args.output is not None:
        from benchmarks.bench_kda_decode import normalize_gpu_type

        if args.output == "__AUTO__":
            output_path = pathlib.Path(f"BENCHMARK_KDA_DECODE_MTP_{normalize_gpu_type(gpu_name)}.md")
        else:
            output_path = pathlib.Path(args.output)
        write_markdown_report(args, gpu_name, results, output_path)
        print(f"Markdown report written to: {output_path.resolve()}")

    return args, gpu_name, results


if __name__ == "__main__":
    main()
