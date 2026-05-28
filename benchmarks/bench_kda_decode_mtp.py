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
bench_kda_decode_mtp.py — P3.0 baseline harness for KDA MTP decode.

Compares two routes that compute the SAME T-token recurrence:
  1. fused  : a single kda_decode_mtp launch over T tokens.
  2. looped : T sequential single-token kda_decode launches (state carried over).

Reports per (N, T): wall time, tokens/s (= N*T / time), fused-vs-looped speedup,
and the tile_v the work_units=N*HV heuristic picked. Also cross-checks the fused
output/state against the looped route (rel error) as a sanity gate.

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
from cula.kda import kda_decode, kda_decode_mtp
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
        "call_loop": call_loop,
        "setup_loop": setup_loop,
        "state_fused": state_fused,
        "state_loop": state_loop,
        "state_init": state_init,
    }


# ──────────────────────────────────────────────────────────────────────
# Timing one config
# ──────────────────────────────────────────────────────────────────────
def run_config(N, T, H, HV, K, V, tile_v_override, warmup, rep, ncu_mode):
    device = "cuda"
    scale = K**-0.5

    q, k, v, a, b, A_log, dt_bias, state = make_inputs_mtp(N, T, H, HV, K, V, device)
    tile_v = tile_v_override if tile_v_override is not None else _select_mtp_tile_v(N, HV, V, T)

    routes = _build_routes(q, k, v, a, b, A_log, dt_bias, state, scale, tile_v_override)

    # Correctness sanity: fused vs looped (run once, fresh state each).
    with torch.no_grad():
        routes["setup_fused"]()
        o_fused = routes["call_fused"]().clone()
        state_fused_final = routes["state_fused"].clone()
        routes["setup_loop"]()
        o_loop = routes["call_loop"]().clone()
        state_loop_final = routes["state_loop"].clone()

    out_rms, out_rel_max = relative_rms_error_rel_max(o_loop, o_fused)
    state_rms, state_rel_max = relative_rms_error_rel_max(state_loop_final, state_fused_final)

    w, r = (1, 1) if ncu_mode else (warmup, rep)

    with torch.no_grad():
        t_fused = benchmark_cuda_fn(routes["call_fused"], setup_fn=routes["setup_fused"], warmup=w, rep=r)
        t_loop = benchmark_cuda_fn(routes["call_loop"], setup_fn=routes["setup_loop"], warmup=w, rep=r)

    tokens = N * T
    return {
        "N": N,
        "T": T,
        "H": H,
        "HV": HV,
        "tile_v": tile_v,
        "t_fused_ms": t_fused,
        "t_loop_ms": t_loop,
        "fused_mtok_s": tokens / t_fused / 1e3 if t_fused > 0 else float("inf"),
        "loop_mtok_s": tokens / t_loop / 1e3 if t_loop > 0 else float("inf"),
        "speedup": t_loop / t_fused if t_fused > 0 else float("inf"),
        "out_rms": out_rms,
        "out_rel_max": out_rel_max,
        "state_rms": state_rms,
        "state_rel_max": state_rel_max,
    }


# ──────────────────────────────────────────────────────────────────────
# Determinism check (bit-for-bit, surfaces state-writeback races)
# ──────────────────────────────────────────────────────────────────────
def run_determinism(N, T, H, HV, K, V, tile_v_override, det_iters):
    device = "cuda"
    scale = K**-0.5

    q, k, v, a, b, A_log, dt_bias, state = make_inputs_mtp(N, T, H, HV, K, V, device)
    tile_v = tile_v_override if tile_v_override is not None else _select_mtp_tile_v(N, HV, V, T)
    routes = _build_routes(q, k, v, a, b, A_log, dt_bias, state, scale, tile_v_override)

    with torch.no_grad():
        routes["setup_fused"]()
        o_ref = routes["call_fused"]().clone()
        state_ref = routes["state_fused"].clone()

        first_bad = -1
        for i in range(det_iters):
            routes["setup_fused"]()
            o_i = routes["call_fused"]()
            same = torch.equal(o_i, o_ref) and torch.equal(routes["state_fused"], state_ref)
            if not same:
                first_bad = i
                break

    return {
        "N": N,
        "T": T,
        "tile_v": tile_v,
        "iters": det_iters,
        "passed": first_bad < 0,
        "first_bad": first_bad,
    }


# ──────────────────────────────────────────────────────────────────────
# Markdown report
# ──────────────────────────────────────────────────────────────────────
def write_markdown_report(args, gpu_name, results, output_path):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    speedups = [r["speedup"] for r in results]

    def summary(vals):
        if not vals:
            return "n/a"
        return f"avg={sum(vals) / len(vals):.2f}x, min={min(vals):.2f}x, max={max(vals):.2f}x"

    lines = []
    lines.append("# Benchmark Results - KDA MTP Decode (P3.0 baseline)")
    lines.append("")
    lines.append(f"> Auto-generated by `benchmarks/bench_kda_decode_mtp.py` on {now}.")
    lines.append("")
    lines.append(
        f"> **GPU:** {gpu_name}  |  **CUDA:** {torch.version.cuda or 'unknown'}  |  "
        f"**PyTorch:** {torch.__version__}  |  **Python:** {platform.python_version()}"
    )
    lines.append("")
    lines.append(f"> Setting: H={args.H}, HV={args.HV}, K={args.K}, V={args.V}; fused (1 launch) vs looped (T× single-token).")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Fused-vs-looped speedup: {summary(speedups)}")
    lines.append(f"- Batch sizes (N): {args.batch_sizes}")
    lines.append(f"- T values: {args.Ts}")
    lines.append(f"- tile_v: {'heuristic (work_units=N*HV)' if args.tile_v is None else args.tile_v}")
    lines.append(f"- Timing: warmup={args.warmup}, rep={args.rep}")
    lines.append("")
    lines.append("## Performance")
    lines.append("")
    lines.append("| N | T | tile_v | fused (ms) | looped (ms) | fused (Mtok/s) | looped (Mtok/s) | speedup | out rel_max | state rel_max |")
    lines.append("|--:|--:|-------:|-----------:|------------:|---------------:|----------------:|--------:|------------:|--------------:|")
    for r in results:
        lines.append(
            f"| {r['N']} | {r['T']} | {r['tile_v']} | {r['t_fused_ms']:.4f} | {r['t_loop_ms']:.4f} | "
            f"{r['fused_mtok_s']:.2f} | {r['loop_mtok_s']:.2f} | **{r['speedup']:.2f}x** | "
            f"{r['out_rel_max']:.2e} | {r['state_rel_max']:.2e} |"
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
    print(f"GPU: {gpu_name}")
    print(f"Config: H={args.H}, HV={args.HV}, K={args.K}, V={args.V}, "
          f"tile_v={'heuristic' if args.tile_v is None else args.tile_v}")
    print()

    if args.determinism:
        hdr = f"{'N':>5} | {'T':>3} | {'tile_v':>6} | {'iters':>8} | {'result':>8}"
        print(hdr)
        print("-" * len(hdr))
        all_ok = True
        for T in args.Ts:
            for N in args.batch_sizes:
                res = run_determinism(N, T, args.H, args.HV, args.K, args.V, args.tile_v, args.det_iters)
                tag = "PASS" if res["passed"] else f"FAIL@{res['first_bad']}"
                all_ok = all_ok and res["passed"]
                print(f"{res['N']:5d} | {res['T']:3d} | {res['tile_v']:6d} | {res['iters']:8d} | {tag:>8}")
        print()
        print("ALL DETERMINISTIC" if all_ok else "NON-DETERMINISM DETECTED — investigate state writeback race")
        return all_ok

    hdr = (
        f"{'N':>5} | {'T':>3} | {'tile_v':>6} | {'fused ms':>9} | {'loop ms':>9} | "
        f"{'fused Mtok/s':>12} | {'loop Mtok/s':>12} | {'speedup':>8} | {'out rmax':>9} | {'state rmax':>10}"
    )
    print(hdr)
    print("-" * len(hdr))

    results = []
    for T in args.Ts:
        for N in args.batch_sizes:
            res = run_config(N, T, args.H, args.HV, args.K, args.V, args.tile_v, args.warmup, args.rep, args.ncu)
            results.append(res)
            print(
                f"{res['N']:5d} | {res['T']:3d} | {res['tile_v']:6d} | {res['t_fused_ms']:9.4f} | {res['t_loop_ms']:9.4f} | "
                f"{res['fused_mtok_s']:12.2f} | {res['loop_mtok_s']:12.2f} | {res['speedup']:7.2f}x | "
                f"{res['out_rel_max']:9.2e} | {res['state_rel_max']:10.2e}"
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
