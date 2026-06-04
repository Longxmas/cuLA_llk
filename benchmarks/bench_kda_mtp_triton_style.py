#!/usr/bin/env python3
"""对照测试 kda_decode_mtp_triton_style(Triton-layout 复刻 1-warp 算子)。

两件事:
  1. 数值正确性:对 Triton(KDA/topk=1/decode) 与 cuLA ws 各算一次 max|Δ|(阈值 5e-2)。
  2. 性能(kernel-only / CUDA graph t_graph):triton vs ws(32:4 recompute) vs triton-style,
     看 "1-warp 瘦 program + reduce-over-K 无 shuffle" 的 layout 在小 batch(尤其 N=4)
     能否补回 ws 那 ~2us 的 wave-量化缺口。

复用 diag_kda_mtp_small_batch 的输入构造 / Triton 封装 / 计时口径。需在装好 cuLA +
triton 的 CUDA 机器(B200)上跑;Mac 无 GPU 不能跑。

用法:
    python benchmarks/bench_kda_mtp_triton_style.py
    python benchmarks/bench_kda_mtp_triton_style.py --batch-sizes 1 2 4 8 --Ts 2 3 4
    python benchmarks/bench_kda_mtp_triton_style.py --check   # 只做数值校验,不计时
"""

import argparse
import os
import pathlib
import sys

import torch

os.environ.setdefault("FLA_USE_FAST_OPS", os.getenv("CULA_USE_FAST_MATH", "1"))
_here = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_here.parent))  # cuLA/
sys.path.insert(0, str(_here))  # benchmarks/ (for diag helpers)

from cula.kda import kda_decode_mtp_ws
from cula.ops.kda_decode_mtp_triton_style import kda_decode_mtp_triton_style

from diag_kda_mtp_small_batch import (  # noqa: E402  复用 diag 基建
    TRITON_MAX_GRID_Z,
    _HAVE_TRITON,
    make_dense_inputs,
    make_triton_call,
    t_graph_ms,
    to_triton_varlen,
    warmup,
)


# tsl 的 opt_level / fast_math / k_split,由 main() 从 CLI 设置(供 register/spill 调优 A/B)。
_TSL_OPT_LEVEL = 3
_TSL_FAST_MATH = True
_TSL_K_SPLIT = 1


def make_triton_style_call(q, k, v, a, b, A_log, dt_bias, state, indices, scale, dsu):
    def call():
        return kda_decode_mtp_triton_style(
            A_log=A_log, dt_bias=dt_bias, q=q, k=k, v=v, a=a, b=b,
            initial_state_source=state, initial_state_indices=indices, scale=scale,
            use_qk_l2norm_in_kernel=True, softplus_beta=1.0, softplus_threshold=20.0,
            disable_state_update=dsu,
            opt_level=_TSL_OPT_LEVEL, fast_math=_TSL_FAST_MATH, k_split=_TSL_K_SPLIT,
        )

    return call


def make_ws_call(q, k, v, a, b, A_log, dt_bias, state, indices, scale, dsu, tile_v, ilp):
    def call():
        return kda_decode_mtp_ws(
            A_log=A_log, dt_bias=dt_bias, q=q, k=k, v=v, a=a, b=b,
            initial_state_source=state, initial_state_indices=indices, scale=scale,
            use_qk_l2norm_in_kernel=True, softplus_beta=1.0, softplus_threshold=20.0,
            tile_v=tile_v, ilp_rows=ilp, use_smem_v=False,
            disable_state_update=dsu, use_gate_in_kernel=True,  # recompute 分支
        )

    return call


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--Ts", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--H", type=int, default=16)
    ap.add_argument("--HV", type=int, default=64)
    ap.add_argument("--K", type=int, default=128)
    ap.add_argument("--V", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--rep", type=int, default=300)
    ap.add_argument("--ws-tile-v", type=int, default=32, help="ws baseline 的 tile_v")
    ap.add_argument("--ws-ilp", type=int, default=4, help="ws baseline 的 ilp_rows")
    ap.add_argument("--check", action="store_true", help="只数值校验,不计时")
    ap.add_argument("--profile", nargs=2, type=int, metavar=("N", "T"), default=None,
                    help="单 (N,T) 长跑某变体 profile_iters 次,供 ncu/nsys 包裹(只 forward,不计时)")
    ap.add_argument("--profile-iters", type=int, default=50)
    ap.add_argument("--profile-variant", choices=["tsl", "triton", "ws"], default="tsl",
                    help="--profile 跑哪个变体(默认 triton-style)")
    ap.add_argument("--tsl-opt-level", type=int, default=3, choices=[0, 1, 2, 3],
                    help="tsl 的 --opt-level(调 ptxas 流水深度/寄存器 A/B)")
    ap.add_argument("--tsl-fast-math", type=int, default=1, choices=[0, 1],
                    help="tsl 的 fast_math(0/1)")
    ap.add_argument("--tsl-k-split", type=int, default=1, choices=[-1, 1, 2, 4],
                    help="tsl 的 k_split:每 V 列由 k_split 个 lane 分摊 K(降寄存器/提 occupancy);-1=auto(按 work_units wave 适配)")
    args = ap.parse_args()

    global _TSL_OPT_LEVEL, _TSL_FAST_MATH, _TSL_K_SPLIT
    _TSL_OPT_LEVEL = args.tsl_opt_level
    _TSL_FAST_MATH = bool(args.tsl_fast_math)
    _TSL_K_SPLIT = args.tsl_k_split

    if not torch.cuda.is_available():
        sys.exit("需要 CUDA GPU(B200 等);Mac 无法跑。")
    if not _HAVE_TRITON:
        sys.exit("Triton 不可用,本对照脚本需要 Triton 作基准。")
    assert args.HV % args.H == 0
    device = "cuda"
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"形状 H={args.H} HV={args.HV} K={args.K} V={args.V}  dsu=True(forward-only)")

    # ---------------- 单 config 长跑(供 ncu/nsys 外部 profiler) ----------------
    if args.profile is not None:
        N, T = args.profile
        q, k, v, a, b, A_log, dt_bias, state0, indices = make_dense_inputs(
            N, T, args.H, args.HV, args.K, args.V, device)
        scale = args.K ** -0.5
        if args.profile_variant == "triton":
            qt, kt, vt, at, bt, cu = to_triton_varlen(q, k, v, a, b)
            fn = make_triton_call(qt, kt, vt, at, bt, cu, A_log, dt_bias,
                                  state0.clone(), indices, scale, True)
        elif args.profile_variant == "ws":
            fn = make_ws_call(q, k, v, a, b, A_log, dt_bias, state0.clone(), indices,
                              scale, True, args.ws_tile_v, args.ws_ilp)
        else:
            fn = make_triton_style_call(q, k, v, a, b, A_log, dt_bias,
                                        state0.clone(), indices, scale, True)
        warmup(fn, args.warmup)
        print(f"[profile] variant={args.profile_variant} N={N} T={T}: "
              f"{args.profile_iters} forward iters(供外部 profiler 包裹)")
        torch.cuda.synchronize()
        for _ in range(args.profile_iters):
            fn()
        torch.cuda.synchronize()
        print("[profile] done.")
        return

    # ---------------- 数值校验 ----------------
    print("\n=== 数值校验 (max|Δ|, 阈值 5e-2) ===")
    print(f"{'N':>4} {'T':>3} | {'Δ tsl-vs-triton':>16} | {'Δ tsl-vs-ws':>12} | flag")
    print("-" * 52)
    ok_all = True
    for N in args.batch_sizes:
        for T in args.Ts:
            q, k, v, a, b, A_log, dt_bias, state0, indices = make_dense_inputs(
                N, T, args.H, args.HV, args.K, args.V, device)
            qt, kt, vt, at, bt, cu = to_triton_varlen(q, k, v, a, b)
            scale = args.K ** -0.5
            o_tri = make_triton_call(qt, kt, vt, at, bt, cu, A_log, dt_bias,
                                     state0.clone(), indices, scale, True)()
            o_tri = o_tri.reshape(N, T, args.HV, args.V).float()
            o_tsl = make_triton_style_call(q, k, v, a, b, A_log, dt_bias,
                                           state0.clone(), indices, scale, True)().float()
            o_ws = make_ws_call(q, k, v, a, b, A_log, dt_bias, state0.clone(), indices,
                                scale, True, args.ws_tile_v, args.ws_ilp)().float()
            d_tri = (o_tsl - o_tri).abs().max().item()
            d_ws = (o_tsl - o_ws).abs().max().item()
            flag = "OK" if (d_tri < 5e-2 and d_ws < 5e-2) else "DIFF!"
            if flag != "OK":
                ok_all = False
            print(f"{N:>4} {T:>3} | {d_tri:>16.2e} | {d_ws:>12.2e} | {flag}")
    print("数值校验:", "全部 OK" if ok_all else "有 DIFF,先查正确性再看性能!")

    if args.check or not ok_all:
        return

    # ---------------- 性能 (t_graph, kernel-only) ----------------
    print("\n=== 性能 t_graph (CUDA graph replay,纯 device kernel) ===")
    print(f"  ws baseline = tile_v={args.ws_tile_v} ilp={args.ws_ilp} (recompute);"
          f" triton-style = 1-warp/BV=32  warmup={args.warmup} rep={args.rep}")
    hdr = (f"{'N':>4} {'T':>3} | {'tg_triton':>9} {'tg_ws':>8} {'tg_tsl':>8} | "
           f"{'tsl/tri':>8} {'ws/tri':>7} | {'tsl vs ws':>9}")
    print(hdr)
    print("-" * len(hdr))
    for N in args.batch_sizes:
        for T in args.Ts:
            q, k, v, a, b, A_log, dt_bias, state0, indices = make_dense_inputs(
                N, T, args.H, args.HV, args.K, args.V, device)
            scale = args.K ** -0.5

            tg_tri = tg_ws = tg_tsl = None
            if N * args.HV <= TRITON_MAX_GRID_Z:
                qt, kt, vt, at, bt, cu = to_triton_varlen(q, k, v, a, b)
                tri = make_triton_call(qt, kt, vt, at, bt, cu, A_log, dt_bias,
                                       state0.clone(), indices, scale, True)
                try:
                    warmup(tri, args.warmup)
                    tg_tri = t_graph_ms(tri, 3, args.rep)
                except Exception as e:
                    print(f"{N:>4} {T:>3} | triton FAIL: {str(e)[:50]}")

            ws = make_ws_call(q, k, v, a, b, A_log, dt_bias, state0.clone(), indices,
                              scale, True, args.ws_tile_v, args.ws_ilp)
            tsl = make_triton_style_call(q, k, v, a, b, A_log, dt_bias,
                                         state0.clone(), indices, scale, True)
            try:
                warmup(ws, args.warmup)
                tg_ws = t_graph_ms(ws, 3, args.rep)
            except Exception as e:
                print(f"{N:>4} {T:>3} | ws FAIL: {str(e)[:50]}")
            try:
                warmup(tsl, args.warmup)
                tg_tsl = t_graph_ms(tsl, 3, args.rep)
            except Exception as e:
                print(f"{N:>4} {T:>3} | tsl FAIL: {str(e)[:50]}")

            def us(x):
                return f"{x * 1e3:>8.1f}" if x else f"{'n/a':>8}"

            r_tsl = f"{tg_tri / tg_tsl:.2f}x" if (tg_tri and tg_tsl) else "n/a"
            r_ws = f"{tg_tri / tg_ws:.2f}x" if (tg_tri and tg_ws) else "n/a"
            r_tw = f"{tg_ws / tg_tsl:.2f}x" if (tg_ws and tg_tsl) else "n/a"
            print(f"{N:>4} {T:>3} | {us(tg_tri):>9} {us(tg_ws)} {us(tg_tsl)} | "
                  f"{r_tsl:>8} {r_ws:>7} | {r_tw:>9}")

    print("\n解读: tsl/tri>1 = triton-style 比 triton 快(复刻还更优);ws/tri 是现状对照。")
    print("     tsl vs ws>1 = triton-style 比 ws 的 32:4 快 → 1-warp layout 在该点更优。")
    print("     重点看 N=4: 若 tsl 追平/超过 triton(tsl/tri≈1.0) → 瘦 CTA 补回了 wave 缺口。")


if __name__ == "__main__":
    main()
