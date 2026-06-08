"""对照 small_batch(vk/kv)/ws-auto/auto-dispatch vs triton + loop(T× 单 token kda_decode)。
数值校验(max|Δ| 阈值 5e-2,以 triton 为基准)+ 性能(kernel-only CUDA graph t_graph)。
用法:python benchmarks/bench_kda_decode_mtp.py [--batch-sizes ... --Ts ... --check]
"""

import argparse
import os
import pathlib
import sys

import torch

os.environ.setdefault("FLA_USE_FAST_OPS", os.getenv("CULA_USE_FAST_MATH", "1"))
_here = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_here.parent))  # cuLA/

from cula.kda import kda_decode, kda_decode_mtp, kda_decode_mtp_ws
from cula.ops.kda_decode_mtp import kda_decode_mtp_small_batch

# CUDA gridDim.z 上限。Triton 把 N*HV 放 z 轴,超过即 launch 失败(cuLA 不受此限)。
TRITON_MAX_GRID_Z = 65535

# Triton 基线:KDA_TRITON_FILE 指定基线文件则从该文件加载,否则退回 sglang 包。
_HAVE_TRITON = True
_TRITON_ERR = ""
fused_sigmoid_gating_delta_rule_update = None
try:
    _triton_file = os.environ.get("KDA_TRITON_FILE", "")
    if _triton_file and os.path.exists(_triton_file):
        import importlib.util

        _spec = importlib.util.spec_from_file_location("_kda_triton_standalone", _triton_file)
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


def make_dense_inputs(N, T, H, HV, K, V, device, seed=42):
    """dense (N,T,...) KDA 输入。"""
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


def warmup(fn, n):
    for _ in range(n):
        fn()
    torch.cuda.synchronize()


def t_graph_ms(fn, warmup_iters, rep, graph_calls=1):
    """Kernel-only 计时:CUDA graph capture + replay,纯 device kernel(wrapper/launcher 全移除);dsu 须 True 保证 replay 幂等。
    graph_calls>1:graph 内连发 K 次 op,放大 per-op 信号、摊薄固定量化/overhead(幂等才合法)。"""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warmup_iters):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(graph_calls):
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
    return start.elapsed_time(end) / rep / graph_calls


# small_batch 的 opt_level / fast_math / k_split,由 main() 从 CLI 设置。
_SB_OPT_LEVEL = 3
_SB_FAST_MATH = True
_SB_K_SPLIT = 1
_VK_BV = -1  # vk 的 BV(每 program V 列数);-1=auto(按 work_units 挑 8/16/32 提 occupancy)


def make_small_batch_call(q, k, v, a, b, A_log, dt_bias, state, indices, scale, dsu, variant="kv"):
    """small_batch 封装;variant='kv'(lane=V/kv 布局)或 'vk'(lane=K/vk 布局)。"""
    if variant == "kv":
        state = state.transpose(-2, -1).contiguous()  # vk→kv 预转置(计时外一次,coalesced)
    common = dict(
        A_log=A_log, dt_bias=dt_bias, q=q, k=k, v=v, a=a, b=b,
        initial_state_source=state, initial_state_indices=indices, scale=scale,
        use_qk_l2norm_in_kernel=True, softplus_beta=1.0, softplus_threshold=20.0,
        disable_state_update=dsu, opt_level=_SB_OPT_LEVEL, fast_math=_SB_FAST_MATH,
    )
    if variant == "kv":
        def call():
            return kda_decode_mtp_small_batch(**common, variant="kv", k_split=_SB_K_SPLIT)
    else:
        def call():
            return kda_decode_mtp_small_batch(**common, variant="vk", bv=_VK_BV)

    return call


def make_ws_call(q, k, v, a, b, A_log, dt_bias, state, indices, scale, dsu):
    """wsAuto:不锁 tile_v/ilp/use_smem_v,由 ws 的 work_units heuristic 自动选。"""
    def call():
        return kda_decode_mtp_ws(
            A_log=A_log, dt_bias=dt_bias, q=q, k=k, v=v, a=a, b=b,
            initial_state_source=state, initial_state_indices=indices, scale=scale,
            use_qk_l2norm_in_kernel=True, softplus_beta=1.0, softplus_threshold=20.0,
            disable_state_update=dsu,
        )

    return call


def make_auto_call(q, k, v, a, b, A_log, dt_bias, state, indices, scale, dsu):
    """auto:dispatch 入口 kda_decode_mtp(state_layout='vk'),按 work_units=N*HV 自适应选
    small_batch vk(<=512)或 ws(>512)——看它能否选出最优。"""
    def call():
        return kda_decode_mtp(
            A_log=A_log, dt_bias=dt_bias, q=q, k=k, v=v, a=a, b=b,
            initial_state_source=state, initial_state_indices=indices, scale=scale,
            use_qk_l2norm_in_kernel=True, softplus_beta=1.0, softplus_threshold=20.0,
            disable_state_update=dsu, state_layout="vk",
        )

    return call


def make_loop_call(q, k, v, a, b, A_log, dt_bias, state, indices, scale, dsu):
    """loop 基线:T 次单 token kda_decode 顺序携带 state(token 切片计时外预切;
    kda_decode 无 dsu,故 loop 始终写 state)。"""
    N, T = q.shape[0], q.shape[1]
    HV, V = v.shape[2], v.shape[3]
    qs = [q[:, t].unsqueeze(1).contiguous() for t in range(T)]
    ks = [k[:, t].unsqueeze(1).contiguous() for t in range(T)]
    vs = [v[:, t].unsqueeze(1).contiguous() for t in range(T)]
    as_ = [a[:, t].unsqueeze(1).contiguous() for t in range(T)]
    bs = [b[:, t].unsqueeze(1).contiguous() for t in range(T)]
    st = state.clone().contiguous()
    o = torch.empty(N, T, HV, V, device=q.device, dtype=torch.bfloat16)

    def call():
        for t in range(T):
            o_t = kda_decode(
                A_log=A_log, dt_bias=dt_bias, q=qs[t], k=ks[t], v=vs[t], a=as_[t], b=bs[t],
                initial_state_source=st, initial_state_indices=indices, scale=scale,
                use_qk_l2norm_in_kernel=True,
            )
            o[:, t] = o_t.squeeze(1)
        return o

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
    ap.add_argument("--graph-calls", type=int, default=1, help="graph 内连发 K 次 op(放大信号、摊薄量化);dsu=True 幂等才合法")
    ap.add_argument("--check", action="store_true", help="只数值校验,不计时")
    ap.add_argument("--check-cases", type=str, nargs="+", default=["1:2", "4:4"],
                    help="精度校验只跑这些 N:T 样例(默认 2 个角点,覆盖最小/最大 T 的累加深度;"
                         "精度对 N 不敏感)。'all'=用 --batch-sizes×--Ts 全网格。大幅提速验证。")
    ap.add_argument("--profile", nargs=2, type=int, metavar=("N", "T"), default=None,
                    help="单 (N,T) 长跑某变体 profile_iters 次,供 ncu/nsys 包裹(只 forward,不计时)")
    ap.add_argument("--profile-iters", type=int, default=50)
    ap.add_argument("--profile-variant", choices=["sbkv", "sbvk", "triton", "ws", "auto", "loop"], default="sbkv",
                    help="--profile 跑哪个变体(sbkv=kv;sbvk=vk;ws=wsAuto;auto=dispatch;loop=T×单token)")
    ap.add_argument("--sb-opt-level", type=int, default=3, choices=[0, 1, 2, 3],
                    help="small_batch 的 --opt-level(调 ptxas 流水深度/寄存器 A/B)")
    ap.add_argument("--sb-fast-math", type=int, default=1, choices=[0, 1],
                    help="small_batch 的 fast_math(0/1)")
    ap.add_argument("--sb-k-split", type=int, default=1, choices=[-1, 1, 2, 4],
                    help="small_batch 的 k_split:每 V 列由 k_split 个 lane 分摊 K(降寄存器/提 occupancy);-1=auto(按 work_units wave 适配)")
    ap.add_argument("--vk-bv", type=int, default=-1, choices=[-1, 8, 16, 32],
                    help="vk 的 BV(每 program V 列数);-1=auto(小批降 BV 提 occupancy 填 wave),或 8/16/32 扫")
    args = ap.parse_args()

    global _SB_OPT_LEVEL, _SB_FAST_MATH, _SB_K_SPLIT, _VK_BV
    _VK_BV = args.vk_bv
    _SB_OPT_LEVEL = args.sb_opt_level
    _SB_FAST_MATH = bool(args.sb_fast_math)
    _SB_K_SPLIT = args.sb_k_split

    if not _HAVE_TRITON:
        sys.exit(f"Triton 不可用,本对照脚本需要 Triton 作基准:{_TRITON_ERR}")
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
                              scale, True)
        elif args.profile_variant == "auto":
            fn = make_auto_call(q, k, v, a, b, A_log, dt_bias, state0.clone(), indices,
                                scale, True)
        elif args.profile_variant == "loop":
            fn = make_loop_call(q, k, v, a, b, A_log, dt_bias, state0.clone(), indices,
                                scale, True)
        elif args.profile_variant == "sbkv":
            fn = make_small_batch_call(q, k, v, a, b, A_log, dt_bias,
                                       state0.clone(), indices, scale, True, variant="kv")
        else:  # sbvk
            fn = make_small_batch_call(q, k, v, a, b, A_log, dt_bias,
                                       state0.clone(), indices, scale, True, variant="vk")
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
    print(f"{'N':>4} {'T':>3} | {'Δ sbkv-tri':>12} | {'Δ sbvk-tri':>12} | {'Δ auto-tri':>12} | flag")
    print("-" * 64)
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
        o_sbkv = make_small_batch_call(q, k, v, a, b, A_log, dt_bias,
                                       state0.clone(), indices, scale, True, variant="kv")().float()
        o_sbvk = make_small_batch_call(q, k, v, a, b, A_log, dt_bias,
                                        state0.clone(), indices, scale, True, variant="vk")().float()
        o_auto = make_auto_call(q, k, v, a, b, A_log, dt_bias, state0.clone(), indices,
                                scale, True)().float()
        d_sbkv = (o_sbkv - o_tri).abs().max().item()
        d_sbvk = (o_sbvk - o_tri).abs().max().item()
        d_auto = (o_auto - o_tri).abs().max().item()
        flag = "OK" if (d_sbkv < 5e-2 and d_sbvk < 5e-2 and d_auto < 5e-2) else "DIFF!"
        if flag != "OK":
            ok_all = False
        print(f"{N:>4} {T:>3} | {d_sbkv:>12.2e} | {d_sbvk:>12.2e} | {d_auto:>12.2e} | {flag}")
    print("数值校验:", "全部 OK" if ok_all else "有 DIFF,先查正确性再看性能!")

    if args.check or not ok_all:
        return

    # ---------------- 性能 (t_graph, kernel-only) ----------------
    # 基线两个:tri(triton)+ loop(T× 单 token)。比值 xx/tri、auto/loop(>1=更快)。
    print("\n=== 性能 t_graph (CUDA graph replay,纯 device kernel) ===")
    print(f"  wsAuto=ws heuristic 自选; small_batch=1-warp; auto=dispatch(wu<=512→sbvk/否则 ws);"
          f" loop=T×单token  warmup={args.warmup} rep={args.rep}")
    hdr = (f"{'N':>4} {'T':>3} | {'tg_loop':>8} {'tg_tri':>7} {'tg_wsA':>7} {'tg_sbkv':>8} {'tg_sbvk':>8} {'tg_auto':>8} | "
           f"{'sbvk/tri':>9} {'sbkv/tri':>9} {'auto/tri':>9} {'auto/loop':>10} | {'pick':>5} {'best':>5}")
    print(hdr)
    print("-" * len(hdr))
    for N in args.batch_sizes:
        for T in args.Ts:
            q, k, v, a, b, A_log, dt_bias, state0, indices = make_dense_inputs(
                N, T, args.H, args.HV, args.K, args.V, device)
            scale = args.K ** -0.5

            tg_tri = None
            if N * args.HV <= TRITON_MAX_GRID_Z:
                qt, kt, vt, at, bt, cu = to_triton_varlen(q, k, v, a, b)
                tri = make_triton_call(qt, kt, vt, at, bt, cu, A_log, dt_bias,
                                       state0.clone(), indices, scale, True)
                try:
                    warmup(tri, args.warmup)
                    tg_tri = t_graph_ms(tri, 3, args.rep, args.graph_calls)
                except Exception as e:
                    print(f"{N:>4} {T:>3} | triton FAIL: {str(e)[:50]}")

            makers = {
                "loop": make_loop_call(q, k, v, a, b, A_log, dt_bias, state0.clone(), indices, scale, True),
                "ws": make_ws_call(q, k, v, a, b, A_log, dt_bias, state0.clone(), indices, scale, True),
                "sbkv": make_small_batch_call(q, k, v, a, b, A_log, dt_bias,
                                              state0.clone(), indices, scale, True, variant="kv"),
                "sbvk": make_small_batch_call(q, k, v, a, b, A_log, dt_bias,
                                              state0.clone(), indices, scale, True, variant="vk"),
                "auto": make_auto_call(q, k, v, a, b, A_log, dt_bias, state0.clone(), indices, scale, True),
            }
            tg = {}
            for name, fn_obj in makers.items():
                try:
                    warmup(fn_obj, args.warmup)
                    tg[name] = t_graph_ms(fn_obj, 3, args.rep, args.graph_calls)
                except Exception as e:
                    tg[name] = None
                    print(f"{N:>4} {T:>3} | {name} FAIL: {str(e)[:50]}")

            def us(x):
                return f"{x * 1e3:.1f}" if x else "n/a"

            def r(x):
                return f"{tg_tri / x:.2f}x" if (tg_tri and x) else "n/a"

            def rl(x):
                lp = tg.get("loop")
                return f"{lp / x:.2f}x" if (lp and x) else "n/a"

            # pick = dispatch 在 vk path 的选择(threshold 512);best = 实测最快(候选不含 loop 基线)
            pick = "sbvk" if N * args.HV <= 512 else "ws"
            cands = {nm: t for nm, t in tg.items() if t and nm != "loop"}
            best = min(cands, key=cands.get) if cands else "-"
            print(f"{N:>4} {T:>3} | {us(tg.get('loop')):>8} {us(tg_tri):>7} {us(tg.get('ws')):>7} "
                  f"{us(tg.get('sbkv')):>8} {us(tg.get('sbvk')):>8} {us(tg.get('auto')):>8} | "
                  f"{r(tg.get('sbvk')):>9} {r(tg.get('sbkv')):>9} {r(tg.get('auto')):>9} {rl(tg.get('auto')):>10} | "
                  f"{pick:>5} {best:>5}")


if __name__ == "__main__":
    main()
