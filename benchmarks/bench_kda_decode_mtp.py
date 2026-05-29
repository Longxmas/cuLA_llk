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
bench_kda_decode_mtp.py — P3.0 baseline + Route1-vs-Route2 harness for KDA MTP decode.

Compares routes that compute the SAME T-token recurrence:
  1. looped : T sequential single-token kda_decode launches (state carried over).
  2. fused  : a single kda_decode_mtp launch over T tokens     [Route 2: cuLA SMEM-state].
  3. ws     : a single kda_decode_mtp_ws launch over T tokens  [Route 1: FlashInfer warp-spec].

Reports per (N, T): wall time, tokens/s (= N*T / time), the tile_v the
work_units=N*HV heuristic picked, fused/ws speedups vs looped, and the
**ws-vs-fused** head-to-head ratio (the Route1-vs-Route2 judge). Also cross-checks
each fused/ws output/state against the looped route (rel error) as a sanity gate.

Pass --routes to restrict which of {loop,fused,ws} run (default: all three).

A bit-for-bit determinism check (--determinism) re-runs the fused kernel many
times from the same initial state and compares output + final state exactly, to
surface state-writeback races before any P3 kernel change lands.

Fairness notes:
  - Both routes reset their state buffer before each timed iteration; the reset
    copy_() runs outside the CUDA event window and is NOT counted.
  - The looped route's per-token tensors are pre-sliced ONCE outside the timed
    loop, so only the T kernel launches are timed (not python slicing) — this
    gives the baseline its best shot; the fused win is purely about fusion.

Usage:
    python benchmarks/bench_kda_decode_mtp.py
    python benchmarks/bench_kda_decode_mtp.py --batch-sizes 1 4 16 64 256 --Ts 2 4 8
    python benchmarks/bench_kda_decode_mtp.py --H 16 --HV 64
    python benchmarks/bench_kda_decode_mtp.py --tile-v 32          # override heuristic
    python benchmarks/bench_kda_decode_mtp.py --determinism --det-iters 10000
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
from cula.kda import kda_decode, kda_decode_mtp, kda_decode_mtp_ws
from cula.ops.kda_decode_mtp import _select_mtp_tile_v


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
    """Build the fused + looped callables, their state buffers, and setup fns.

    Returns a dict with keys: call_fused, setup_fused, call_loop, setup_loop,
    state_fused, state_loop (the live buffers, mutated in-place).
    """
    N, T = q.shape[0], q.shape[1]
    device = q.device
    indices = torch.arange(N, device=device, dtype=torch.int32)

    state_init = state.clone().contiguous()  # (N, HV, V, K)

    # --- fused route: one launch over T tokens ---
    state_fused = state_init.clone()

    def call_fused():
        return kda_decode_mtp(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            initial_state_source=state_fused,
            initial_state_indices=indices,
            scale=scale,
            use_qk_l2norm_in_kernel=True,
            tile_v=tile_v,
        )

    def setup_fused():
        state_fused.copy_(state_init)

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
        )

    def setup_ws():
        state_ws.copy_(state_init)

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
        "call_fused": call_fused,
        "setup_fused": setup_fused,
        "call_ws": call_ws,
        "setup_ws": setup_ws,
        "call_loop": call_loop,
        "setup_loop": setup_loop,
        "state_fused": state_fused,
        "state_ws": state_ws,
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

    routes = _build_routes(q, k, v, a, b, A_log, dt_bias, state, scale, tile_v_override)

    # Correctness reference = looped single-token route (always run once, fresh
    # state). Each fused/ws route is cross-checked against it (rel error).
    nan2 = (float("nan"), float("nan"))
    corr = {}  # route name -> (out_rel_max, state_rel_max)
    with torch.no_grad():
        routes["setup_loop"]()
        o_loop = routes["call_loop"]().clone()
        state_loop_final = routes["state_loop"].clone()

        for name in ("fused", "ws"):
            if name in route_set:
                routes[f"setup_{name}"]()
                o_r = routes[f"call_{name}"]().clone()
                st_r = routes[f"state_{name}"].clone()
                _, out_rel_max = relative_rms_error_rel_max(o_loop, o_r)
                _, state_rel_max = relative_rms_error_rel_max(state_loop_final, st_r)
                corr[name] = (out_rel_max, state_rel_max)

    w, r = (1, 1) if ncu_mode else (warmup, rep)

    times = {}
    with torch.no_grad():
        for name in ("loop", "fused", "ws"):
            if name in route_set:
                times[name] = benchmark_cuda_fn(
                    routes[f"call_{name}"], setup_fn=routes[f"setup_{name}"], warmup=w, rep=r
                )

    tokens = N * T

    def mtok(t):
        return tokens / t / 1e3 if (t is not None and t > 0) else float("nan")

    def speedup_vs_loop(t):
        tl = times.get("loop")
        return (tl / t) if (tl is not None and t is not None and t > 0) else float("nan")

    t_fused, t_ws = times.get("fused"), times.get("ws")
    return {
        "N": N,
        "T": T,
        "H": H,
        "HV": HV,
        "tile_v": tile_v,
        "t_loop_ms": times.get("loop"),
        "t_fused_ms": t_fused,
        "t_ws_ms": t_ws,
        "loop_mtok_s": mtok(times.get("loop")),
        "fused_mtok_s": mtok(t_fused),
        "ws_mtok_s": mtok(t_ws),
        "fused_speedup": speedup_vs_loop(t_fused),  # vs looped
        "ws_speedup": speedup_vs_loop(t_ws),  # vs looped
        # Head-to-head Route1-vs-Route2: >1 means ws (Route 1) beats fused (Route 2).
        "ws_vs_fused": (t_fused / t_ws) if (t_fused and t_ws and t_ws > 0) else float("nan"),
        "fused_out_rel_max": corr.get("fused", nan2)[0],
        "fused_state_rel_max": corr.get("fused", nan2)[1],
        "ws_out_rel_max": corr.get("ws", nan2)[0],
        "ws_state_rel_max": corr.get("ws", nan2)[1],
    }


# ──────────────────────────────────────────────────────────────────────
# Determinism check (bit-for-bit, surfaces state-writeback races)
# ──────────────────────────────────────────────────────────────────────
def run_determinism(N, T, H, HV, K, V, tile_v_override, det_iters, route):
    device = "cuda"
    scale = K**-0.5

    q, k, v, a, b, A_log, dt_bias, state = make_inputs_mtp(N, T, H, HV, K, V, device)
    tile_v = tile_v_override if tile_v_override is not None else _select_mtp_tile_v(N, HV, V, T)
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
    }


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
    lines.append("# Benchmark Results - KDA MTP Decode (Route 1 ws vs Route 2 fused)")
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
        "fused = kda_decode_mtp (Route 2, cuLA SMEM-state); ws = kda_decode_mtp_ws "
        "(Route 1, FlashInfer warp-spec); looped = T× single-token kda_decode."
    )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- fused-vs-looped speedup: {summary([r['fused_speedup'] for r in results])}")
    lines.append(f"- ws-vs-looped speedup:    {summary([r['ws_speedup'] for r in results])}")
    lines.append(f"- **ws-vs-fused (Route1/Route2): {summary([r['ws_vs_fused'] for r in results])}**  (>1 → Route 1 wins)")
    lines.append(f"- Batch sizes (N): {args.batch_sizes}")
    lines.append(f"- T values: {args.Ts}")
    lines.append(f"- tile_v: {'heuristic (work_units=N*HV)' if args.tile_v is None else args.tile_v}")
    lines.append(f"- Timing: warmup={args.warmup}, rep={args.rep}")
    lines.append("")
    lines.append("## Performance")
    lines.append("")
    lines.append("| N | T | tile_v | loop ms | fused ms | ws ms | fused Mtok/s | ws Mtok/s | fused/loop | ws/loop | ws/fused | ws out rmax | ws state rmax |")
    lines.append("|--:|--:|-------:|--------:|---------:|------:|-------------:|----------:|-----------:|--------:|---------:|------------:|--------------:|")
    for r in results:
        lines.append(
            f"| {r['N']} | {r['T']} | {r['tile_v']} | {_fmt(r['t_loop_ms'], '.4f')} | {_fmt(r['t_fused_ms'], '.4f')} | "
            f"{_fmt(r['t_ws_ms'], '.4f')} | {_fmt(r['fused_mtok_s'], '.2f')} | {_fmt(r['ws_mtok_s'], '.2f')} | "
            f"{_fmt(r['fused_speedup'], '.2f')}x | {_fmt(r['ws_speedup'], '.2f')}x | **{_fmt(r['ws_vs_fused'], '.2f')}x** | "
            f"{_fmt(r['ws_out_rel_max'], '.2e')} | {_fmt(r['ws_state_rel_max'], '.2e')} |"
        )
    lines.append("")
    lines.append("## Reproduce")
    lines.append("")
    lines.append("```bash")
    lines.append("python benchmarks/bench_kda_decode_mtp.py")
    lines.append("```")
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def build_parser():
    parser = argparse.ArgumentParser(description="Benchmark KDA MTP decode: fused vs T× single-token")
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
        choices=["loop", "fused", "ws"],
        default=["loop", "fused", "ws"],
        help="Which routes to run/time. loop=T× single-token, fused=kda_decode_mtp (Route 2), ws=kda_decode_mtp_ws (Route 1).",
    )
    parser.add_argument("--determinism", action="store_true", help="Run bit-for-bit determinism check instead of timing")
    parser.add_argument("--det-iters", type=int, default=10000, help="Determinism check repetitions per config")
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
    print(f"GPU: {gpu_name}")
    print(f"Config: H={args.H}, HV={args.HV}, K={args.K}, V={args.V}, "
          f"tile_v={'heuristic' if args.tile_v is None else args.tile_v}, routes={args.routes}")
    print()

    if args.determinism:
        det_routes = [r for r in args.routes if r in ("fused", "ws")]
        if not det_routes:
            print("No state-writeback route selected (--routes must include fused and/or ws for --determinism).")
            return True
        hdr = f"{'route':>6} | {'N':>5} | {'T':>3} | {'tile_v':>6} | {'iters':>8} | {'result':>8}"
        print(hdr)
        print("-" * len(hdr))
        all_ok = True
        for route in det_routes:
            for T in args.Ts:
                for N in args.batch_sizes:
                    res = run_determinism(N, T, args.H, args.HV, args.K, args.V, args.tile_v, args.det_iters, route)
                    tag = "PASS" if res["passed"] else f"FAIL@{res['first_bad']}"
                    all_ok = all_ok and res["passed"]
                    print(f"{res['route']:>6} | {res['N']:5d} | {res['T']:3d} | {res['tile_v']:6d} | {res['iters']:8d} | {tag:>8}")
        print()
        print("ALL DETERMINISTIC" if all_ok else "NON-DETERMINISM DETECTED — investigate state writeback race")
        return all_ok

    hdr = (
        f"{'N':>5} | {'T':>3} | {'tile_v':>6} | {'loop ms':>9} | {'fused ms':>9} | {'ws ms':>9} | "
        f"{'fused Mt/s':>11} | {'ws Mt/s':>11} | {'ws/loop':>8} | {'ws/fused':>9} | {'ws out rmax':>11} | {'ws st rmax':>11}"
    )
    print(hdr)
    print("-" * len(hdr))

    results = []
    for T in args.Ts:
        for N in args.batch_sizes:
            res = run_config(N, T, args.H, args.HV, args.K, args.V, args.tile_v, args.warmup, args.rep, args.ncu, route_set)
            results.append(res)
            print(
                f"{res['N']:5d} | {res['T']:3d} | {res['tile_v']:6d} | "
                f"{_fmt(res['t_loop_ms'], '.4f'):>9} | {_fmt(res['t_fused_ms'], '.4f'):>9} | {_fmt(res['t_ws_ms'], '.4f'):>9} | "
                f"{_fmt(res['fused_mtok_s'], '.2f'):>11} | {_fmt(res['ws_mtok_s'], '.2f'):>11} | "
                f"{_fmt(res['ws_speedup'], '.2f'):>7}x | {_fmt(res['ws_vs_fused'], '.2f'):>8}x | "
                f"{_fmt(res['ws_out_rel_max'], '.2e'):>11} | {_fmt(res['ws_state_rel_max'], '.2e'):>11}"
            )
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
