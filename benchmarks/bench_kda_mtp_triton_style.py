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
from cula.ops.kda_decode_mtp_triton_style import (
    kda_decode_mtp_triton_aligned,
    kda_decode_mtp_triton_style,
)

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
_TSL_SKIP_LOAD = False  # ablation:tsl 传 -1 indices 跳过 state load,只测 perf(定位 fixed deficit 是否=state load)


def make_triton_style_call(q, k, v, a, b, A_log, dt_bias, state, indices, scale, dsu,
                           state_layout="vk"):
    if _TSL_SKIP_LOAD:
        indices = torch.full_like(indices, -1)  # ablation:cache_idx<0 → kernel 跳过 state load 这一步
    if state_layout == "kv":
        # vk→kv 预转置(计时外一次)。dsu=True → state 只读不变,转一次后所有 forward 复用,
        # 不进计时;让 lane=V列 的 state load/store 直接 coalesced,消掉硬塞 vk 的 uncoalesced 惩罚。
        state = state.transpose(-2, -1).contiguous()
    def call():
        return kda_decode_mtp_triton_style(
            A_log=A_log, dt_bias=dt_bias, q=q, k=k, v=v, a=a, b=b,
            initial_state_source=state, initial_state_indices=indices, scale=scale,
            use_qk_l2norm_in_kernel=True, softplus_beta=1.0, softplus_threshold=20.0,
            disable_state_update=dsu, state_layout=state_layout,
            opt_level=_TSL_OPT_LEVEL, fast_math=_TSL_FAST_MATH, k_split=_TSL_K_SPLIT,
        )

    return call


def make_triton_aligned_call(q, k, v, a, b, A_log, dt_bias, state, indices, scale, dsu):
    if _TSL_SKIP_LOAD:
        indices = torch.full_like(indices, -1)
    def call():
        return kda_decode_mtp_triton_aligned(
            A_log=A_log, dt_bias=dt_bias, q=q, k=k, v=v, a=a, b=b,
            initial_state_source=state, initial_state_indices=indices, scale=scale,
            use_qk_l2norm_in_kernel=True, softplus_beta=1.0, softplus_threshold=20.0,
            disable_state_update=dsu,
            opt_level=_TSL_OPT_LEVEL, fast_math=_TSL_FAST_MATH,
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
    ap.add_argument("--check-cases", type=str, nargs="+", default=["1:2", "4:4"],
                    help="精度校验只跑这些 N:T 样例(默认 2 个角点,覆盖最小/最大 T 的累加深度;"
                         "精度对 N 不敏感)。'all'=用 --batch-sizes×--Ts 全网格。大幅提速验证。")
    ap.add_argument("--profile", nargs=2, type=int, metavar=("N", "T"), default=None,
                    help="单 (N,T) 长跑某变体 profile_iters 次,供 ncu/nsys 包裹(只 forward,不计时)")
    ap.add_argument("--profile-iters", type=int, default=50)
    ap.add_argument("--profile-variant", choices=["tsl", "tsl_kv", "tsl_aligned", "triton", "ws"], default="tsl",
                    help="--profile 跑哪个变体(tsl_kv=kv 布局;tsl_aligned=lane=K+shuffle 对齐 triton)")
    ap.add_argument("--tsl-opt-level", type=int, default=3, choices=[0, 1, 2, 3],
                    help="tsl 的 --opt-level(调 ptxas 流水深度/寄存器 A/B)")
    ap.add_argument("--tsl-fast-math", type=int, default=1, choices=[0, 1],
                    help="tsl 的 fast_math(0/1)")
    ap.add_argument("--tsl-k-split", type=int, default=1, choices=[-1, 1, 2, 4],
                    help="tsl 的 k_split:每 V 列由 k_split 个 lane 分摊 K(降寄存器/提 occupancy);-1=auto(按 work_units wave 适配)")
    ap.add_argument("--tsl-skip-load", action="store_true",
                    help="ablation:tsl 传 -1 indices 跳过 state load(只测 perf,正确性必错)——定位 fixed deficit 是否来自 state load")
    args = ap.parse_args()

    global _TSL_OPT_LEVEL, _TSL_FAST_MATH, _TSL_K_SPLIT, _TSL_SKIP_LOAD
    _TSL_OPT_LEVEL = args.tsl_opt_level
    _TSL_FAST_MATH = bool(args.tsl_fast_math)
    _TSL_K_SPLIT = args.tsl_k_split
    _TSL_SKIP_LOAD = args.tsl_skip_load
    if _TSL_SKIP_LOAD:
        print("[--tsl-skip-load] tsl 跳过 state load:数值校验对 tsl 必报 DIFF(预期);只看下面 perf 的 tg_tsl vs 有 load 基准。")

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
        elif args.profile_variant == "tsl_kv":
            fn = make_triton_style_call(q, k, v, a, b, A_log, dt_bias,
                                        state0.clone(), indices, scale, True, state_layout="kv")
        elif args.profile_variant == "tsl_aligned":
            fn = make_triton_aligned_call(q, k, v, a, b, A_log, dt_bias,
                                          state0.clone(), indices, scale, True)
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
    print(f"{'N':>4} {'T':>3} | {'Δ tsl-vs-tri':>13} | {'Δ tslkv-vs-tri':>14} | {'Δ tslK-vs-tri':>13} | {'Δ tsl-vs-ws':>12} | flag")
    print("-" * 82)
    if len(args.check_cases) == 1 and args.check_cases[0].lower() == "all":
        check_cases = [(N, T) for N in args.batch_sizes for T in args.Ts]
    else:
        check_cases = [tuple(int(x) for x in c.split(":")) for c in args.check_cases]
    ok_all = True
    for N, T in check_cases:
        q, k, v, a, b, A_log, dt_bias, state0, indices = make_dense_inputs(
            N, T, args.H, args.HV, args.K, args.V, device)
        qt, kt, vt, at, bt, cu = to_triton_varlen(q, k, v, a, b)
        scale = args.K ** -0.5
        o_tri = make_triton_call(qt, kt, vt, at, bt, cu, A_log, dt_bias,
                                 state0.clone(), indices, scale, True)()
        o_tri = o_tri.reshape(N, T, args.HV, args.V).float()
        o_tsl = make_triton_style_call(q, k, v, a, b, A_log, dt_bias,
                                       state0.clone(), indices, scale, True)().float()
        o_tsl_kv = make_triton_style_call(q, k, v, a, b, A_log, dt_bias,
                                          state0.clone(), indices, scale, True,
                                          state_layout="kv")().float()
        o_tslK = make_triton_aligned_call(q, k, v, a, b, A_log, dt_bias,
                                          state0.clone(), indices, scale, True)().float()
        o_ws = make_ws_call(q, k, v, a, b, A_log, dt_bias, state0.clone(), indices,
                            scale, True, args.ws_tile_v, args.ws_ilp)().float()
        d_tri = (o_tsl - o_tri).abs().max().item()
        d_kv = (o_tsl_kv - o_tri).abs().max().item()
        d_K = (o_tslK - o_tri).abs().max().item()
        d_ws = (o_tsl - o_ws).abs().max().item()
        flag = "OK" if (d_tri < 5e-2 and d_kv < 5e-2 and d_K < 5e-2 and d_ws < 5e-2) else "DIFF!"
        if flag != "OK":
            ok_all = False
        print(f"{N:>4} {T:>3} | {d_tri:>13.2e} | {d_kv:>14.2e} | {d_K:>13.2e} | {d_ws:>12.2e} | {flag}")
    print("数值校验:", "全部 OK" if ok_all else "有 DIFF,先查正确性再看性能!")

    if args.check or (not ok_all and not _TSL_SKIP_LOAD):
        return

    # ---------------- 性能 (t_graph, kernel-only) ----------------
    print("\n=== 性能 t_graph (CUDA graph replay,纯 device kernel) ===")
    print(f"  ws baseline = tile_v={args.ws_tile_v} ilp={args.ws_ilp} (recompute);"
          f" triton-style = 1-warp/BV=32  warmup={args.warmup} rep={args.rep}")
    hdr = (f"{'N':>4} {'T':>3} | {'tg_triton':>9} {'tg_ws':>8} {'tg_tsl':>8} {'tg_tslkv':>9} {'tg_tslK':>9} | "
           f"{'tslK/tri':>8} {'tslkv/tri':>9} {'tsl/tri':>8} {'ws/tri':>7}")
    print(hdr)
    print("-" * len(hdr))
    for N in args.batch_sizes:
        for T in args.Ts:
            q, k, v, a, b, A_log, dt_bias, state0, indices = make_dense_inputs(
                N, T, args.H, args.HV, args.K, args.V, device)
            scale = args.K ** -0.5

            tg_tri = tg_ws = tg_tsl = tg_tslkv = tg_tslK = None
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
            tsl_kv = make_triton_style_call(q, k, v, a, b, A_log, dt_bias,
                                            state0.clone(), indices, scale, True,
                                            state_layout="kv")
            tslK = make_triton_aligned_call(q, k, v, a, b, A_log, dt_bias,
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
            try:
                warmup(tsl_kv, args.warmup)
                tg_tslkv = t_graph_ms(tsl_kv, 3, args.rep)
            except Exception as e:
                print(f"{N:>4} {T:>3} | tsl_kv FAIL: {str(e)[:50]}")
            try:
                warmup(tslK, args.warmup)
                tg_tslK = t_graph_ms(tslK, 3, args.rep)
            except Exception as e:
                print(f"{N:>4} {T:>3} | tsl_aligned FAIL: {str(e)[:50]}")

            def us(x):
                return f"{x * 1e3:>8.1f}" if x else f"{'n/a':>8}"

            r_K = f"{tg_tri / tg_tslK:.2f}x" if (tg_tri and tg_tslK) else "n/a"
            r_kv = f"{tg_tri / tg_tslkv:.2f}x" if (tg_tri and tg_tslkv) else "n/a"
            r_tsl = f"{tg_tri / tg_tsl:.2f}x" if (tg_tri and tg_tsl) else "n/a"
            r_ws = f"{tg_tri / tg_ws:.2f}x" if (tg_tri and tg_ws) else "n/a"
            print(f"{N:>4} {T:>3} | {us(tg_tri):>9} {us(tg_ws)} {us(tg_tsl)} {us(tg_tslkv):>9} {us(tg_tslK):>9} | "
                  f"{r_K:>8} {r_kv:>9} {r_tsl:>8} {r_ws:>7}")


if __name__ == "__main__":
    main()
