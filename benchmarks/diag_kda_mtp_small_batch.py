#!/usr/bin/env python3
"""KDA MTP decode — 小 batch 诊断 (issue #17, Route-1 ws/inline)。

目标:在 batch<16 处 cuLA 比 Triton 慢 (0.85~0.90x)。这个脚本一次 B200 run 同时回答
「主因是 host wrapper 开销 / 占用率不足 / gating 冗余 哪一个」,再决定下一步走
便宜的 heuristic retune (C) / wrapper 瘦身 还是结构性 gating 预pass (B)。

两大部分:

  Part 1 —— host-vs-device 时间拆分 (决定性诊断)
    对 ws / inline / Triton 各测三个口径:
      t_cpu  : 隔离的 CPU 入队开销。每 iter 先 cuda.synchronize() 排空 GPU,再用
               perf_counter 计 fn() —— kernel 是异步 launch,在计时点后才真正跑,
               所以这是 *纯 Python wrapper + enqueue* 成本。
      t_pipe : 稳态吞吐。连续入队 rep 次、不在循环里 sync、末尾一次 sync,再 /rep。
               CPU 与 GPU 跨 iter 完全重叠,所以 t_pipe ≈ max(t_cpu, t_dev)。
      t_event: per-iter CUDA event 计时 (对齐现有 benchmark_cuda_fn 口径,≈ 串行
               cpu+dev,作连续性参照)。
    判据:
      t_pipe ≲ t_cpu*1.15  → HOST-bound  (wrapper 是天花板,kernel 已被 CPU 掩盖)
      t_pipe >  t_cpu*1.15  → DEVICE-bound (t_dev ≈ t_pipe,看 tile_v/结构)
    cuLA wrapper 比 Triton 重 (config 选择 + 5× _normalize_* + 连续性检查 +
    _prepare_output_tensor 分配 o + 2× .view() + 每次 torch.zeros(1,1,1) dummy);
    若小 batch 是 HOST-bound,差距就在这里,不在 kernel。

  Part 2 —— 小 N 的 tile_v × {ws, inline} sweep vs Triton
    heuristic 在小 work_units 选 tile_v=8 (→ num_v_tiles=16 → 16× gating 冗余,
    Triton BV=32 → 4×)。但小 tile_v 是为了多 CTA 填占用率。这里对 N∈{1,2,4,8,16}
    枚举 tile_v∈{8,16,32,64} (V%tile_v==0)、变体 {ws, inline},直接看哪个 tile_v
    在小 N 最快、能否翻过 Triton —— 既验证冗余假设,又白送 heuristic-retune 的答案。
    (ilp 取 4 if tile_v%16==0 else 2,镜像 _select_mtp_config;use_smem_v 钉 False
    以隔离 tile_v 效应。)

计时默认 disable_state_update=True (forward-only,无状态漂移,kernel 计时干净;
writeback 对 cuLA/Triton 是同种 scatter,vs-Triton 的 *比值* 不受影响)。用
--state-update 翻成 False 复现含写回的口径。

约束:K=V=128 (cuLA TILE_K=128)。HV 必须能被 H 整除 (GQA)。需在装好 cuLA +
triton 的 CUDA 机器上跑 (如 B200);Mac 无 GPU 不能跑。

用法:
    python benchmarks/diag_kda_mtp_small_batch.py                       # Part 1 + Part 2, 默认 HV=64
    python benchmarks/diag_kda_mtp_small_batch.py --part 1 --HV 64
    python benchmarks/diag_kda_mtp_small_batch.py --part 2 --batch-sizes 1 2 4 8 16 --Ts 2 4
    # 显式 config 对比(把 tile_v 提高/ilp=4)@ kernel-only,聚焦 N=4,T=2:
    python benchmarks/diag_kda_mtp_small_batch.py --part 4 --batch-sizes 4 --Ts 2
    python benchmarks/diag_kda_mtp_small_batch.py --check               # 跑一次数值校验再计时
    # 单 config 长跑供外部 profiler (nsys/ncu) 包裹:
    python benchmarks/diag_kda_mtp_small_batch.py --profile-config 1 2 8 ws --profile-iters 2000
"""

import argparse
import os
import pathlib
import sys
from time import perf_counter

import torch

# cuLA 的 fla fast-ops 开关(默认开),须在 import cula 之前设置
os.environ.setdefault("FLA_USE_FAST_OPS", os.getenv("CULA_USE_FAST_MATH", "1"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from cula.kda import kda_decode_mtp_ws, kda_decode_mtp_ws_inline
from cula.ops.kda_decode_mtp_ws import _select_mtp_config

# Triton 基线:优先用同目录的独立文件 (零 sglang 依赖),否则从 sglang 包导入。
_HAVE_TRITON = True
fused_sigmoid_gating_delta_rule_update = None
try:
    import importlib.util

    # 候选路径(KDA_TRITON_FILE 最高优先):脚本同目录、repo 内/外的 Issue 17/ 子目录。
    # 这样无论 Issue 17/ 放在 cuLA repo 里还是外都能找到独立 Triton 文件;都没有则退回
    # sglang 包导入;再没有就仅测 cuLA(Part 1 的 host/device 拆分不需要 Triton)。
    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(_here)
    _candidates = [
        os.environ.get("KDA_TRITON_FILE", ""),
        os.path.join(_here, "fused_sigmoid_gating_recurrent.py"),
        os.path.join(_repo, "Issue 17", "fused_sigmoid_gating_recurrent.py"),
        os.path.join(_repo, "..", "Issue 17", "fused_sigmoid_gating_recurrent.py"),
    ]
    _triton_path = next((p for p in _candidates if p and os.path.exists(p)), None)
    if _triton_path is not None:
        _spec = importlib.util.spec_from_file_location("_kda_triton_standalone", _triton_path)
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        fused_sigmoid_gating_delta_rule_update = _mod.fused_sigmoid_gating_delta_rule_update
    else:
        from sglang.srt.layers.attention.fla.fused_sigmoid_gating_recurrent import (
            fused_sigmoid_gating_delta_rule_update,
        )
except Exception as e:  # pragma: no cover
    _HAVE_TRITON = False
    _TRITON_ERR = repr(e)

# CUDA gridDim.z 上限。Triton 把 N*HV 放 z 轴,超过即 launch 失败 (cuLA 不受此限)。
TRITON_MAX_GRID_Z = 65535

# 合法 tile_v (V 的因子、4 的倍数)。
TILE_V_CHOICES = (8, 16, 32, 64)


# ============================================================================
# 输入构造(dense),与 bench_kda_mtp_ws_vs_triton.py::make_dense_inputs 对齐
# ============================================================================
def make_dense_inputs(N, T, H, HV, K, V, device, seed=42):
    g = torch.Generator(device=device).manual_seed(seed)
    bf16 = torch.bfloat16
    q = torch.randn(N, T, H, K, device=device, dtype=bf16, generator=g)
    k = torch.randn(N, T, H, K, device=device, dtype=bf16, generator=g)
    v = torch.randn(N, T, HV, V, device=device, dtype=bf16, generator=g)
    a = (torch.randn(N, T, HV, K, device=device, dtype=torch.float32, generator=g) * 0.1).to(bf16)
    b = torch.randn(N, T, HV, device=device, dtype=bf16, generator=g)
    A_log = -torch.rand(HV, device=device, dtype=torch.float32, generator=g) * 2  # neg -> decay∈(0,1)
    dt_bias = torch.randn(HV, K, device=device, dtype=torch.float32, generator=g) * 0.1
    state = torch.randn(N, HV, V, K, device=device, dtype=torch.float32, generator=g) * 0.01
    indices = torch.arange(N, device=device, dtype=torch.int32)
    return q, k, v, a, b, A_log, dt_bias, state, indices


def to_triton_varlen(q, k, v, a, b):
    """dense (N,T,...) -> varlen packed [1, N*T, ...] + 等长 cu_seqlens。"""
    N, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[3]
    NT = N * T
    q_t = q.reshape(1, NT, H, K).contiguous()
    k_t = k.reshape(1, NT, H, K).contiguous()
    v_t = v.reshape(1, NT, HV, V).contiguous()
    a_t = a.reshape(1, NT, HV * K).contiguous()
    b_t = b.reshape(1, NT, HV).contiguous()
    cu_seqlens = torch.arange(0, (N + 1) * T, T, device=q.device, dtype=torch.int32)
    return q_t, k_t, v_t, a_t, b_t, cu_seqlens


# ============================================================================
# 被测 kernel 的调用封装。tile_v=None / ilp_rows=None -> 走 production heuristic。
# ============================================================================
def make_cula_call(variant, q, k, v, a, b, A_log, dt_bias, state, indices, scale,
                   tile_v, ilp_rows, use_smem_v, dsu, precompute_gating=True):
    fn = kda_decode_mtp_ws if variant == "ws" else kda_decode_mtp_ws_inline

    def call():
        return fn(
            A_log=A_log, dt_bias=dt_bias, q=q, k=k, v=v, a=a, b=b,
            initial_state_source=state, initial_state_indices=indices, scale=scale,
            use_qk_l2norm_in_kernel=True, softplus_beta=1.0, softplus_threshold=20.0,
            tile_v=tile_v, ilp_rows=ilp_rows, use_smem_v=use_smem_v,
            disable_state_update=dsu, intermediate_states_buffer=None,
            use_gate_in_kernel=not precompute_gating,
        )

    return call


def make_triton_call(qt, kt, vt, at, bt, cu_seqlens, A_log, dt_bias, state, indices, scale, dsu):
    def call():
        return fused_sigmoid_gating_delta_rule_update(
            A_log=A_log, a=at, dt_bias=dt_bias, softplus_beta=1.0, softplus_threshold=20.0,
            q=qt, k=kt, v=vt, b=bt, initial_state_source=state, initial_state_indices=indices,
            scale=scale, use_qk_l2norm_in_kernel=True, cu_seqlens=cu_seqlens, is_kda=True,
            disable_state_update=dsu, intermediate_states_buffer=None,
            retrieve_parent_token=None, lower_bound=None,
        )

    return call


# ============================================================================
# 三个计时口径 (都接受已 warmup 的 fn)
# ============================================================================
def _iqr_mean(xs):
    xs = sorted(xs)
    n = len(xs)
    if n <= 2:
        return sum(xs) / n
    lo, hi = n // 4, n - n // 4
    return sum(xs[lo:hi]) / (hi - lo)


def warmup(fn, n):
    for _ in range(n):
        fn()
    torch.cuda.synchronize()


def t_cpu_ms(fn, rep):
    """隔离的 CPU 入队开销:每 iter 先 sync 排空 GPU,再计 fn() (异步 launch 立即返回)。"""
    xs = []
    for _ in range(rep):
        torch.cuda.synchronize()
        t0 = perf_counter()
        fn()
        t1 = perf_counter()
        xs.append((t1 - t0) * 1e3)
    return _iqr_mean(xs)


def t_pipe_ms(fn, rep):
    """稳态吞吐:连续入队不 sync,末尾一次 sync。= max(t_cpu, t_dev) (CPU/GPU 全重叠)。"""
    torch.cuda.synchronize()
    t0 = perf_counter()
    for _ in range(rep):
        fn()
    torch.cuda.synchronize()
    return (perf_counter() - t0) / rep * 1e3


def t_event_ms(fn, rep):
    """per-iter CUDA event (≈ 串行 cpu+dev;对齐现有 benchmark_cuda_fn 口径)。"""
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    for i in range(rep):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    return _iqr_mean([s.elapsed_time(e) for s, e in zip(starts, ends)])


def t_graph_ms(fn, warmup_iters, rep):
    """Kernel-only via CUDA graph: capture fn() once (wrapper runs once to RECORD),
    then time graph.replay() = pure device kernel work — BOTH cuLA's and Triton's
    Python wrapper AND launcher are gone (replay only re-issues the recorded CUDA
    ops). Also == the real cost under CUDA-graph serving. Raises on capture failure
    (caller catches -> n/a); dsu MUST be True so replay is idempotent (no state drift
    across replays writing to the captured state buffer)."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warmup_iters):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(10):
        g.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        g.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / rep


def measure_all(fn, warmup_iters, rep):
    warmup(fn, warmup_iters)
    # 顺序:先 t_pipe (吞吐),再 t_cpu (隔离),再 t_event (对齐)。各自独立,无残留。
    tp = t_pipe_ms(fn, rep)
    tc = t_cpu_ms(fn, rep)
    te = t_event_ms(fn, rep)
    return tc, tp, te


def verdict(tc, tp):
    """t_pipe ≈ t_cpu → HOST-bound;否则 DEVICE-bound。"""
    if tp <= tc * 1.15:
        return "HOST", tc / tp if tp > 0 else float("nan")
    return "DEVICE", tc / tp if tp > 0 else float("nan")


# ============================================================================
# 数值校验(可选):以 Triton 为基准对一次 ws/inline,确认 swept config 没静默算错
# ============================================================================
def check_vs_triton(variant, N, T, H, HV, K, V, tile_v, ilp_rows, device):
    q, k, v, a, b, A_log, dt_bias, state0, indices = make_dense_inputs(N, T, H, HV, K, V, device)
    qt, kt, vt, at, bt, cu = to_triton_varlen(q, k, v, a, b)
    scale = K ** -0.5
    o_t = make_triton_call(qt, kt, vt, at, bt, cu, A_log, dt_bias, state0.clone(), indices, scale, True)()
    o_t = o_t.reshape(N, T, HV, V).float()
    o_c = make_cula_call(variant, q, k, v, a, b, A_log, dt_bias, state0.clone(), indices, scale,
                         tile_v, ilp_rows, False, True)().float()
    return (o_c - o_t).abs().max().item()


# ============================================================================
# Part 1 — host/device 拆分
# ============================================================================
def run_part1(args, device):
    print("\n" + "=" * 100)
    print("Part 1 — host-vs-device 时间拆分 (t_cpu=隔离CPU入队, t_pipe=稳态吞吐≈max(cpu,dev), t_event≈串行)")
    print(f"  H={args.H} HV={args.HV} K={args.K} V={args.V}  dsu={args.dsu}  "
          f"warmup={args.warmup} rep={args.rep}  (config=production heuristic)")
    print("=" * 100)
    hdr = (f"{'N':>4} {'T':>3} | {'route':>7} {'tile_v':>6} {'ilp':>3} | "
           f"{'t_cpu us':>9} {'t_pipe us':>10} {'t_event us':>10} | "
           f"{'t_dev≈ us':>10} {'cpu/pipe':>8} | {'verdict':>7} | {'vs Tri pipe':>11}")
    print(hdr)
    print("-" * len(hdr))

    for N in args.batch_sizes:
        for T in args.Ts:
            q, k, v, a, b, A_log, dt_bias, state0, indices = make_dense_inputs(
                N, T, args.H, args.HV, args.K, args.V, device)
            scale = args.K ** -0.5
            tile_v, ilp_rows, use_smem_v = _select_mtp_config(N, args.HV, args.V, T)

            # Triton 基线 (受 gridDim.z 限制;小 N 不会触发)。
            tri_pipe = None
            triton_ok = _HAVE_TRITON and N * args.HV <= TRITON_MAX_GRID_Z
            if triton_ok:
                qt, kt, vt, at, bt, cu = to_triton_varlen(q, k, v, a, b)
                tri = make_triton_call(qt, kt, vt, at, bt, cu, A_log, dt_bias,
                                       state0.clone(), indices, scale, args.dsu)
                try:
                    tc, tp, te = measure_all(tri, args.warmup, args.rep)
                    tri_pipe = tp
                    vd, frac = verdict(tc, tp)
                    tdev = "<cpu" if vd == "HOST" else f"{tp * 1e3:.1f}"
                    print(f"{N:>4} {T:>3} | {'triton':>7} {'-':>6} {'-':>3} | "
                          f"{tc * 1e3:>9.1f} {tp * 1e3:>10.1f} {te * 1e3:>10.1f} | "
                          f"{tdev:>10} {frac:>8.2f} | {vd:>7} | {'(base)':>11}")
                except Exception as e:
                    print(f"{N:>4} {T:>3} | triton ERROR: {e}")

            for variant in ("ws", "inline"):
                v_ilp = ilp_rows
                if v_ilp == 4 and tile_v % 16 != 0:
                    v_ilp = 2
                fn = make_cula_call(variant, q, k, v, a, b, A_log, dt_bias,
                                    state0.clone(), indices, scale,
                                    tile_v, v_ilp, use_smem_v, args.dsu)
                try:
                    tc, tp, te = measure_all(fn, args.warmup, args.rep)
                    vd, frac = verdict(tc, tp)
                    tdev = "<cpu" if vd == "HOST" else f"{tp * 1e3:.1f}"
                    vs_tri = f"{tri_pipe / tp:.2f}x" if tri_pipe else "n/a"
                    print(f"{N:>4} {T:>3} | {variant:>7} {tile_v:>6} {v_ilp:>3} | "
                          f"{tc * 1e3:>9.1f} {tp * 1e3:>10.1f} {te * 1e3:>10.1f} | "
                          f"{tdev:>10} {frac:>8.2f} | {vd:>7} | {vs_tri:>11}")
                except Exception as e:
                    print(f"{N:>4} {T:>3} | {variant} ERROR: {e}")
            print()

    print("解读:cpu/pipe→1 (verdict=HOST) = wrapper 是天花板,kernel 已被掩盖 → 先瘦 wrapper;")
    print("     cpu/pipe<<1 (verdict=DEVICE) = t_dev≈t_pipe 主导 → 看 Part 2 的 tile_v/结构。")
    print("     对比 cuLA 与 triton 的 t_cpu:差值≈两边 Python wrapper 的纯开销差。")


# ============================================================================
# Part 2 — 小 N 的 tile_v × {ws,inline} sweep vs Triton
# ============================================================================
def run_part2(args, device):
    print("\n" + "=" * 100)
    print("Part 2 — tile_v × {ws,inline} sweep vs Triton (口径=t_pipe 稳态吞吐;ilp=4 if tile_v%16==0 else 2)")
    print(f"  H={args.H} HV={args.HV} K={args.K} V={args.V}  dsu={args.dsu}  "
          f"warmup={args.warmup} rep={args.rep}  use_smem_v=False(钉死隔离 tile_v)")
    print("=" * 100)
    tvs = [tv for tv in TILE_V_CHOICES if args.V % tv == 0]
    hdr = (f"{'N':>4} {'T':>3} | {'route':>7} | "
           + " ".join(f"tv{tv:>2} us" for tv in tvs)
           + f" | {'Tri us':>8} | {'best tv':>7} {'best/Tri':>8}")
    print(hdr)
    print("-" * len(hdr))

    for N in args.batch_sizes:
        for T in args.Ts:
            q, k, v, a, b, A_log, dt_bias, state0, indices = make_dense_inputs(
                N, T, args.H, args.HV, args.K, args.V, device)
            scale = args.K ** -0.5

            # Triton 基线
            tri_pipe = None
            if _HAVE_TRITON and N * args.HV <= TRITON_MAX_GRID_Z:
                qt, kt, vt, at, bt, cu = to_triton_varlen(q, k, v, a, b)
                tri = make_triton_call(qt, kt, vt, at, bt, cu, A_log, dt_bias,
                                       state0.clone(), indices, scale, args.dsu)
                try:
                    warmup(tri, args.warmup)
                    tri_pipe = t_pipe_ms(tri, args.rep)
                except Exception as e:
                    print(f"{N:>4} {T:>3} | triton ERROR: {e}")

            for variant in ("ws", "inline"):
                cells = []
                best_tv, best_ms = None, float("inf")
                for tv in tvs:
                    ilp = 4 if tv % 16 == 0 else 2
                    fn = make_cula_call(variant, q, k, v, a, b, A_log, dt_bias,
                                        state0.clone(), indices, scale, tv, ilp, False, args.dsu)
                    try:
                        warmup(fn, args.warmup)
                        ms = t_pipe_ms(fn, args.rep)
                        cells.append(f"{ms * 1e3:>7.1f}")
                        if ms < best_ms:
                            best_ms, best_tv = ms, tv
                    except Exception:
                        cells.append(f"{'n/a':>7}")
                tri_s = f"{tri_pipe * 1e3:>8.1f}" if tri_pipe else f"{'n/a':>8}"
                spd = f"{tri_pipe / best_ms:.2f}x" if (tri_pipe and best_tv) else "n/a"
                bt = f"{best_tv}" if best_tv else "-"
                print(f"{N:>4} {T:>3} | {variant:>7} | " + " ".join(cells)
                      + f" | {tri_s} | {bt:>7} {spd:>8}")
            print()

    print("解读:若某个 tile_v 比 heuristic 的 tile_v=8 明显快、且翻过 Triton(best/Tri>1) →")
    print("     冗余假设成立 + retune _select_mtp_config 小 work_units 阈值即可(便宜的 C)。")
    print("     若所有 tile_v 都翻不过 Triton 且 Part 1 是 DEVICE-bound → 才考虑结构 pre-pass(B)。")


# ============================================================================
# Part 3 — kernel-only (CUDA graph replay):移除 cuLA + Triton 双方 wrapper+launcher
# ============================================================================
def run_part3(args, device):
    print("\n" + "=" * 100)
    print("Part 3 — kernel-only (CUDA graph replay):移除 cuLA + Triton 双方 wrapper AND launcher,纯 device kernel")
    print(f"  H={args.H} HV={args.HV} K={args.K} V={args.V}  dsu=True(强制,replay 需状态只读)  "
          f"warmup={args.warmup} rep={args.rep}  (config=production heuristic)")
    print("=" * 100)
    hdr = (f"{'N':>4} {'T':>3} | {'route':>7} {'tile_v':>6} {'ilp':>3} | "
           f"{'t_pipe us':>9} {'t_graph us':>10} {'wrap+lnch':>10} | {'vs Tri graph':>12}")
    print(hdr)
    print("-" * len(hdr))

    for N in args.batch_sizes:
        for T in args.Ts:
            q, k, v, a, b, A_log, dt_bias, state0, indices = make_dense_inputs(
                N, T, args.H, args.HV, args.K, args.V, device)
            scale = args.K ** -0.5
            tile_v, ilp_rows, use_smem_v = _select_mtp_config(N, args.HV, args.V, T)

            tri_g = None
            if _HAVE_TRITON and N * args.HV <= TRITON_MAX_GRID_Z:
                qt, kt, vt, at, bt, cu = to_triton_varlen(q, k, v, a, b)
                tri = make_triton_call(qt, kt, vt, at, bt, cu, A_log, dt_bias,
                                       state0.clone(), indices, scale, True)
                try:
                    warmup(tri, args.warmup)
                    tp = t_pipe_ms(tri, args.rep)
                    tg = t_graph_ms(tri, 3, args.rep)
                    tri_g = tg
                    print(f"{N:>4} {T:>3} | {'triton':>7} {'-':>6} {'-':>3} | "
                          f"{tp * 1e3:>9.1f} {tg * 1e3:>10.1f} {(tp - tg) * 1e3:>9.1f}u | {'(base)':>12}")
                except Exception as e:
                    print(f"{N:>4} {T:>3} | triton graph-capture FAIL: {str(e)[:70]}")

            # A/B the structural gating fix in one run: ws+pre (gating pre-pass on)
            # vs ws-rec (in-kernel recompute = the pre-fix baseline) vs inl+pre.
            # Same tile_v/ilp so the only delta is precompute_gating.
            cula_variants = [
                ("ws+pre", "ws", True),
                ("ws-rec", "ws", False),
                ("inl+pre", "inline", True),
            ]
            for label, variant, precomp in cula_variants:
                v_ilp = ilp_rows
                if v_ilp == 4 and tile_v % 16 != 0:
                    v_ilp = 2
                fn = make_cula_call(variant, q, k, v, a, b, A_log, dt_bias,
                                    state0.clone(), indices, scale, tile_v, v_ilp, use_smem_v, True,
                                    precompute_gating=precomp)
                try:
                    warmup(fn, args.warmup)
                    tp = t_pipe_ms(fn, args.rep)
                    tg = t_graph_ms(fn, 3, args.rep)
                    vs = f"{tri_g / tg:.2f}x" if tri_g else "n/a"
                    print(f"{N:>4} {T:>3} | {label:>7} {tile_v:>6} {v_ilp:>3} | "
                          f"{tp * 1e3:>9.1f} {tg * 1e3:>10.1f} {(tp - tg) * 1e3:>9.1f}u | {vs:>12}")
                except Exception as e:
                    print(f"{N:>4} {T:>3} | {label} graph-capture FAIL: {str(e)[:70]}")
            print()

    print("解读: t_graph = 纯 device kernel(wrapper+launcher 全移除,= CUDA-graph serving 真实代价)。")
    print("     wrap+lnch = t_pipe - t_graph = eager 下被 wrapper+launcher 吃掉的部分。")
    print("     vs Tri graph >1 = 移除双方 wrapper 后 cuLA kernel 本身更快(这才是算子真实对比)。")
    print("     ws+pre vs ws-rec = gating 预pass(本次结构改动)的净效果;N=4 区应从 <1 翻到 >=1。")


# ============================================================================
# Part 4 — 显式 (tile_v, ilp) config sweep @ kernel-only (CUDA graph replay)
#   回答:把 tile_v 提高 / ilp 设 4,在 t_graph(纯 device)口径下相对 Triton 如何。
#   隔离逻辑:同一 (N,T) 下并排 (16,2)=现状 / (16,4)=只换 ilp(同 grid 同冗余) /
#   (32,4)=提议修复(冗余 8x→4x + grid 减半)。口径强制 t_graph(小 batch 的 t_pipe
#   是 host-bound 假象);分支强制 recompute(use_gate_in_kernel=True,即要优化的
#   分支);变体钉 ws;use_smem_v=False 以隔离 tile_v 效应。
# ============================================================================
def _parse_configs(spec_list):
    """["16:2","16:4","32:4"] -> [(16,2),(16,4),(32,4)]。"""
    out = []
    for s in spec_list:
        if ":" not in s:
            raise ValueError(f"--configs 项格式应为 TILE_V:ILP,得到 {s!r}")
        tv_s, ilp_s = s.split(":", 1)
        out.append((int(tv_s), int(ilp_s)))
    return out


def _config_legal(tile_v, ilp, V):
    """镜像 kda_decode_mtp_ws 的合法性断言,返回 (ok, 原因)。"""
    if ilp not in (2, 4):
        return False, "ilp∉{2,4}"
    if tile_v not in TILE_V_CHOICES:
        return False, "tv∉{8,16,32,64}"
    if V % tile_v != 0:
        return False, "V%tv!=0"
    rows_per_group = tile_v // 4
    if rows_per_group % ilp != 0:
        return False, f"(tv//4={rows_per_group})%ilp"
    if ilp == 4 and tile_v % 16 != 0:
        return False, "ilp4 需 tv%16==0"
    return True, ""


def run_part4(args, device):
    configs = _parse_configs(args.configs)
    print("\n" + "=" * 100)
    print("Part 4 — 显式 (tile_v, ilp) config sweep @ kernel-only (CUDA graph replay,纯 device)")
    print(f"  分支=recompute(use_gate_in_kernel=True) 变体={args.variants}(ws=warp-spec/inl=无 warp-spec) "
          f"use_smem_v=False dsu=True")
    print(f"  H={args.H} HV={args.HV} K={args.K} V={args.V}  warmup={args.warmup} rep={args.rep}")
    print(f"  configs={['%d:%d' % c for c in configs]}  ('*'=该 (N,T) 下 heuristic 当前选中=现状基准)")
    print("=" * 100)
    hdr = (f"{'N':>4} {'T':>3} | {'variant':>7} {'tile_v':>6} {'ilp':>3} | "
           f"{'redund':>6} {'grid':>6} | {'maxΔ':>9} | {'t_graph us':>10} | {'vs Tri':>7} | heur")
    print(hdr)
    print("-" * len(hdr))

    for N in args.batch_sizes:
        for T in args.Ts:
            q, k, v, a, b, A_log, dt_bias, state0, indices = make_dense_inputs(
                N, T, args.H, args.HV, args.K, args.V, device)
            scale = args.K ** -0.5

            # heuristic 当前在该 (N,T) 会选的 config(用于标 '*' = 现状基准)。
            h_tv, h_ilp, _ = _select_mtp_config(N, args.HV, args.V, T)
            if h_ilp == 4 and h_tv % 16 != 0:
                h_ilp = 2

            # Triton 基线(t_graph) + 数值参照 o_ref。
            tri_g = None
            o_ref = None
            if _HAVE_TRITON and N * args.HV <= TRITON_MAX_GRID_Z:
                qt, kt, vt, at, bt, cu = to_triton_varlen(q, k, v, a, b)
                tri = make_triton_call(qt, kt, vt, at, bt, cu, A_log, dt_bias,
                                       state0.clone(), indices, scale, True)
                try:
                    o_ref = tri().reshape(N, T, args.HV, args.V).float()
                    warmup(tri, args.warmup)
                    tri_g = t_graph_ms(tri, 3, args.rep)
                    print(f"{N:>4} {T:>3} | {'triton':>7} {'-':>6} {'-':>3} | "
                          f"{'4x':>6} {N * args.HV * (args.V // 32):>6} | {'-':>9} | "
                          f"{tri_g * 1e3:>10.1f} | {'(base)':>7} |")
                except Exception as e:
                    print(f"{N:>4} {T:>3} | triton graph-capture FAIL: {str(e)[:60]}")

            for (tv, ilp) in configs:
                ok, why = _config_legal(tv, ilp, args.V)
                star = " *" if (tv, ilp) == (h_tv, h_ilp) else ""
                if not ok:
                    print(f"{N:>4} {T:>3} | {'-':>7} {tv:>6} {ilp:>3} | "
                          f"{'-':>6} {'-':>6} | {'illegal':>9} | {why:>10} | {'-':>7} |{star}")
                    continue
                num_vt = args.V // tv
                grid = N * args.HV * num_vt
                # ws=warp-spec(warp0 算 gating + barrier 广播);inl=无 warp-spec(4 warp
                # 各自算 gating,无 barrier,但仍 4-warp CTA)。两者同 grid/tile_v/ilp,唯一
                # 变量是 warp-spec → 判定 config 治不了的差距是否来自 warp-spec。
                for variant in args.variants:
                    vlabel = "ws-rec" if variant == "ws" else "inl-rec"
                    fn = make_cula_call(variant, q, k, v, a, b, A_log, dt_bias,
                                        state0.clone(), indices, scale, tv, ilp, False, True,
                                        precompute_gating=False)
                    try:
                        # 先对 Triton ref 校验,避免拿算错的 config 去比快。
                        d_s = "n/a"
                        if o_ref is not None:
                            d = (fn().float() - o_ref).abs().max().item()
                            d_s = f"{d:.1e}{'!' if d > 5e-2 else ''}"
                        warmup(fn, args.warmup)
                        tg = t_graph_ms(fn, 3, args.rep)
                        vs = f"{tri_g / tg:.2f}x" if tri_g else "n/a"
                        print(f"{N:>4} {T:>3} | {vlabel:>7} {tv:>6} {ilp:>3} | "
                              f"{str(num_vt) + 'x':>6} {grid:>6} | {d_s:>9} | "
                              f"{tg * 1e3:>10.1f} | {vs:>7} |{star}")
                    except Exception as e:
                        print(f"{N:>4} {T:>3} | {vlabel:>7} {tv:>6} {ilp:>3} | "
                              f"graph FAIL: {str(e)[:50]}")
            print()

    print("解读: t_graph=纯 device kernel(双方 wrapper+launcher 全移除)。vs Tri>1 = cuLA kernel 更快。")
    print("     redund=num_v_tiles=V/tile_v(每 (i_n,i_hv) 的 gating 重算次数;Triton 固定 4x)。")
    print("     关键对照 @ N=4,T=2: (16,2)现状 → (16,4)只换 ilp → (32,4)提议修复,看缺口由谁补回:")
    print("       现状 vs (16,4) = 纯 ilp 贡献(同 grid 同冗余);(16,4) vs (32,4) = 冗余+grid 贡献。")
    print("     ws-rec vs inl-rec(同 tile_v/ilp,唯一差 warp-spec): inl 优于 ws 且≈triton →")
    print("       config 治不了的差距来自 warp-spec(warp0 串行 gating/barrier 等待);inl≈ws(都差) →")
    print("       来自 4-warp 胖 CTA 占用率/wave 量化(与 warp-spec 无关)。注意 inline 是 scalar-FMA")
    print("       (无 packed F32x2),ilp=4 的 compute 略逊 ws,解读 ilp 差异时扣掉这一项。")
    print("     '*' = 当前 heuristic 在该 (N,T) 选中的 config。")


# ============================================================================
# 单 config 长跑,供 nsys/ncu 等外部 profiler 包裹
# ============================================================================
def run_profile_config(args, device):
    N, T, tile_v, variant = args.profile_config
    N, T, tile_v = int(N), int(T), int(tile_v)
    ilp = 4 if tile_v % 16 == 0 else 2
    q, k, v, a, b, A_log, dt_bias, state0, indices = make_dense_inputs(
        N, T, args.H, args.HV, args.K, args.V, device)
    scale = args.K ** -0.5
    if variant == "triton":
        qt, kt, vt, at, bt, cu = to_triton_varlen(q, k, v, a, b)
        fn = make_triton_call(qt, kt, vt, at, bt, cu, A_log, dt_bias, state0.clone(),
                              indices, scale, args.dsu)
    else:
        fn = make_cula_call(variant, q, k, v, a, b, A_log, dt_bias, state0.clone(),
                            indices, scale, tile_v, ilp, False, args.dsu)
    warmup(fn, args.warmup)
    print(f"[profile] {variant} N={N} T={T} tile_v={tile_v} ilp={ilp}: "
          f"running {args.profile_iters} iters for external profiler ...")
    torch.cuda.synchronize()
    t0 = perf_counter()
    for _ in range(args.profile_iters):
        fn()
    torch.cuda.synchronize()
    print(f"[profile] done. wall/iter = {(perf_counter() - t0) / args.profile_iters * 1e3 * 1e3:.1f} us")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16], help="N 列表(小 batch)")
    ap.add_argument("--Ts", type=int, nargs="+", default=[2, 4], help="draft 长度 T 列表")
    ap.add_argument("--H", type=int, default=16, help="q/k head 数")
    ap.add_argument("--HV", type=int, default=64, help="v head 数(须被 H 整除)")
    ap.add_argument("--K", type=int, default=128)
    ap.add_argument("--V", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--rep", type=int, default=300)
    ap.add_argument("--part", choices=["1", "2", "3", "4", "all"], default="all",
                    help="1=host/device拆分 2=tile_v sweep 3=kernel-only(CUDA graph) "
                         "4=显式config sweep@kernel-only(recompute分支) all=1+2")
    ap.add_argument("--state-update", dest="dsu", action="store_false",
                    help="计时含状态写回(disable_state_update=False,复现含写回口径);默认 forward-only")
    ap.add_argument("--check", action="store_true", help="计时前先对 Triton 跑一次数值校验")
    ap.add_argument("--profile-config", nargs=4, metavar=("N", "T", "TILE_V", "VARIANT"),
                    default=None, help="单 config 长跑供 nsys/ncu 包裹,如: 1 2 8 ws")
    ap.add_argument("--profile-iters", type=int, default=2000)
    ap.add_argument("--configs", type=str, nargs="+",
                    default=["16:2", "16:4", "32:4", "32:2", "64:4"],
                    help="Part 4 的显式 TILE_V:ILP 列表(默认覆盖 N=4,T=2 的 (16,2)现状→(16,4)→(32,4) 对照)")
    ap.add_argument("--variants", type=str, nargs="+", default=["ws", "inline"],
                    choices=["ws", "inline"],
                    help="Part 4 测哪些变体(默认 ws+inline,对照 warp-spec 的影响)")
    ap.set_defaults(dsu=True)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("需要 CUDA GPU(本脚本在 B200 等机器上运行;Mac 无法跑)。")
    if not _HAVE_TRITON:
        print(f"[warn] Triton 不可用,仅测 cuLA: {_TRITON_ERR}", file=sys.stderr)
    assert args.HV % args.H == 0, f"HV({args.HV}) 必须被 H({args.H}) 整除"
    device = "cuda"
    print(f"GPU: {torch.cuda.get_device_name()}")

    if args.profile_config is not None:
        run_profile_config(args, device)
        return

    if args.check:
        print("数值校验 (cuLA vs Triton, max|Δ|, 阈值 5e-2):")
        for N in args.batch_sizes[:2]:
            for variant in ("ws", "inline"):
                for tv in [tv for tv in TILE_V_CHOICES if args.V % tv == 0]:
                    ilp = 4 if tv % 16 == 0 else 2
                    try:
                        d = check_vs_triton(variant, N, args.Ts[0], args.H, args.HV,
                                            args.K, args.V, tv, ilp, device)
                        flag = "OK" if d < 5e-2 else "DIFF!"
                        print(f"  {variant:>7} N={N} T={args.Ts[0]} tile_v={tv:>2} ilp={ilp}: "
                              f"max|Δ|={d:.2e} {flag}")
                    except Exception as e:
                        print(f"  {variant:>7} N={N} tile_v={tv:>2}: ERROR {e}")

    if args.part in ("1", "all"):
        run_part1(args, device)
    if args.part in ("2", "all"):
        run_part2(args, device)
    if args.part == "3":
        run_part3(args, device)
    if args.part == "4":
        run_part4(args, device)


if __name__ == "__main__":
    main()
