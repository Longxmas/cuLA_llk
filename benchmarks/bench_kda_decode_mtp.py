import argparse
import os
import pathlib
import sys

import torch

os.environ.setdefault("FLA_USE_FAST_OPS", os.getenv("CULA_USE_FAST_MATH", "1"))
_here = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_here.parent))

from cula.kda import kda_decode, kda_decode_mtp, kda_decode_mtp_ws
from cula.ops.kda_decode_mtp import kda_decode_mtp_small_batch

TRITON_MAX_GRID_Z = 65535
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
except Exception as e:
    _HAVE_TRITON = False
    _TRITON_ERR = repr(e)


def make_dense_inputs(N, T, H, HV, K, V, device, seed=42):
    """Dense (N,T,...) KDA inputs."""
    g = torch.Generator(device=device).manual_seed(seed)
    bf16 = torch.bfloat16
    q = torch.randn(N, T, H, K, device=device, dtype=bf16, generator=g)
    k = torch.randn(N, T, H, K, device=device, dtype=bf16, generator=g)
    v = torch.randn(N, T, HV, V, device=device, dtype=bf16, generator=g)
    a = (torch.randn(N, T, HV, K, device=device, dtype=torch.float32, generator=g) * 0.1).to(bf16)
    b = torch.randn(N, T, HV, device=device, dtype=bf16, generator=g)
    A_log = -torch.rand(HV, device=device, dtype=torch.float32, generator=g) * 2  # neg -> decay in (0,1)
    dt_bias = torch.randn(HV, K, device=device, dtype=torch.float32, generator=g) * 0.1
    state = torch.randn(N, HV, V, K, device=device, dtype=torch.float32, generator=g) * 0.01
    indices = torch.arange(N, device=device, dtype=torch.int32)
    return q, k, v, a, b, A_log, dt_bias, state, indices


def to_triton_varlen(q, k, v, a, b):
    """Dense (N,T,...) -> varlen-packed [1, N*T, ...] + equal-length cu_seqlens."""
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


def make_triton_call(qt, kt, vt, at, bt, cu_seqlens, A_log, dt_bias, state, indices, scale, dsu, intermediate=False):
    inter_buf = inter_idx = cache_steps = None
    if intermediate:
        Nt = cu_seqlens.numel() - 1
        Tt = int(cu_seqlens[1] - cu_seqlens[0])
        HVt, Vt, Kt = vt.shape[2], vt.shape[3], qt.shape[3]
        inter_buf = torch.zeros(Nt, Tt, HVt, Vt, Kt, device=qt.device, dtype=torch.float32)
        inter_idx = torch.arange(Nt, device=qt.device, dtype=torch.int32)
        cache_steps = Tt
    def call():
        return fused_sigmoid_gating_delta_rule_update(
            A_log=A_log, a=at, dt_bias=dt_bias, softplus_beta=1.0, softplus_threshold=20.0,
            q=qt, k=kt, v=vt, b=bt, initial_state_source=state, initial_state_indices=indices,
            scale=scale, use_qk_l2norm_in_kernel=True, cu_seqlens=cu_seqlens, is_kda=True,
            disable_state_update=dsu, intermediate_states_buffer=inter_buf, intermediate_state_indices=inter_idx, cache_steps=cache_steps,
            retrieve_parent_token=None, lower_bound=None,
        )

    return call


def warmup(fn, n):
    for _ in range(n):
        fn()
    torch.cuda.synchronize()


def t_graph_ms(fn, warmup_iters, rep, graph_calls=1):
    """Kernel-only timing via CUDA graph capture+replay; graph_calls>1 fires K ops/graph to amortize fixed overhead (needs idempotent dsu)."""
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

_SB_OPT_LEVEL = 3
_SB_FAST_MATH = True
_SB_K_SPLIT = 1
_VK_BV = -1

def make_small_batch_call(q, k, v, a, b, A_log, dt_bias, state, indices, scale, dsu, variant="kv", intermediate=False):
    """small_batch wrapper; variant='kv' (lane=V) or 'vk' (lane=K)."""
    if variant == "kv":
        state = state.transpose(-2, -1).contiguous()  # vk->kv pre-transpose (once, outside timing, coalesced)
    inter_buf = None
    if intermediate and variant == "vk":
        Nb, Tb = q.shape[0], q.shape[1]; HVb, Vb, Kb = v.shape[2], v.shape[3], q.shape[3]
        inter_buf = torch.zeros(Nb, Tb, HVb, Vb, Kb, device=q.device, dtype=torch.float32)
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
            return kda_decode_mtp_small_batch(**common, variant="vk", bv=_VK_BV, intermediate_states_buffer=inter_buf)

    return call


def make_ws_call(q, k, v, a, b, A_log, dt_bias, state, indices, scale, dsu, intermediate=False):
    """wsAuto: tile_v/ilp/use_smem_v auto-picked by the ws work_units heuristic."""
    inter_buf = None
    if intermediate:
        Nw, Tw = q.shape[0], q.shape[1]; HVw, Vw, Kw = v.shape[2], v.shape[3], q.shape[3]
        inter_buf = torch.zeros(Nw, Tw, HVw, Vw, Kw, device=q.device, dtype=torch.float32)
    def call():
        return kda_decode_mtp_ws(
            A_log=A_log, dt_bias=dt_bias, q=q, k=k, v=v, a=a, b=b,
            initial_state_source=state, initial_state_indices=indices, scale=scale,
            use_qk_l2norm_in_kernel=True, softplus_beta=1.0, softplus_threshold=20.0,
            disable_state_update=dsu, intermediate_states_buffer=inter_buf,
        )

    return call


def make_auto_call(q, k, v, a, b, A_log, dt_bias, state, indices, scale, dsu, intermediate=False):
    """auto: kda_decode_mtp dispatch (state_layout='vk'); picks small_batch vk (wu<=512) or ws (>512) by work_units=N*HV."""
    inter_buf = None
    if intermediate:
        Na, Ta = q.shape[0], q.shape[1]; HVa, Va, Ka = v.shape[2], v.shape[3], q.shape[3]
        inter_buf = torch.zeros(Na, Ta, HVa, Va, Ka, device=q.device, dtype=torch.float32)
    def call():
        return kda_decode_mtp(
            A_log=A_log, dt_bias=dt_bias, q=q, k=k, v=v, a=a, b=b,
            initial_state_source=state, initial_state_indices=indices, scale=scale,
            use_qk_l2norm_in_kernel=True, softplus_beta=1.0, softplus_threshold=20.0,
            disable_state_update=dsu, state_layout="vk", intermediate_states_buffer=inter_buf,
        )

    return call


def make_loop_call(q, k, v, a, b, A_log, dt_bias, state, indices, scale, dsu):
    """loop baseline: T single-token kda_decode carrying state (slices pre-cut outside timing; kda_decode always writes state)."""
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
    ap.add_argument("--graph-calls", type=int, default=1, help="fire K ops per graph (amplify signal, amortize quantization); needs idempotent dsu=True")
    ap.add_argument("--dsu", type=int, default=1, choices=[0, 1], help="disable_state_update; 0=write state (production), 1=forward-only (idempotent, default)")
    ap.add_argument("--intermediate", type=int, default=0, choices=[0, 1], help="write per-token snapshot buffer (vk/ws/auto + triton; kv unsupported)")
    ap.add_argument("--check", action="store_true", help="numerical check only, no timing")
    ap.add_argument("--check-cases", type=str, nargs="+", default=["1:2", "4:4"],
                    help="legacy; numerical check now always covers the full --batch-sizes x --Ts grid"
                         )
    ap.add_argument("--profile", nargs=2, type=int, metavar=("N", "T"), default=None,
                    help="long single-(N,T) run of one variant for ncu/nsys wrapping (forward-only, no timing)")
    ap.add_argument("--profile-iters", type=int, default=50)
    ap.add_argument("--profile-variant", choices=["sbkv", "sbvk", "triton", "ws", "auto", "loop"], default="sbkv",
                    help="which variant for --profile (sbkv=kv; sbvk=vk; ws=wsAuto; auto=dispatch; loop=T x single-token)")
    ap.add_argument("--sb-opt-level", type=int, default=3, choices=[0, 1, 2, 3],
                    help="small_batch opt-level (ptxas pipeline-depth / register A/B)")
    ap.add_argument("--sb-fast-math", type=int, default=1, choices=[0, 1],
                    help="small_batch fast_math (0/1)")
    ap.add_argument("--sb-k-split", type=int, default=1, choices=[-1, 1, 2, 4],
                    help="small_batch k_split: k_split lanes split each V-col K (lower regs/higher occupancy); -1=auto")
    ap.add_argument("--vk-bv", type=int, default=-1, choices=[-1, 8, 16, 32],
                    help="vk BV (V-cols per program); -1=auto (small batch lowers BV for occupancy), or sweep 8/16/32")
    args = ap.parse_args()
    DSU = bool(args.dsu)
    INTER = bool(args.intermediate)

    global _SB_OPT_LEVEL, _SB_FAST_MATH, _SB_K_SPLIT, _VK_BV
    _VK_BV = args.vk_bv
    _SB_OPT_LEVEL = args.sb_opt_level
    _SB_FAST_MATH = bool(args.sb_fast_math)
    _SB_K_SPLIT = args.sb_k_split

    if not _HAVE_TRITON:
        sys.exit(f"Triton unavailable; this comparison script needs Triton as baseline: {_TRITON_ERR}")
    assert args.HV % args.H == 0
    device = "cuda"
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"shape H={args.H} HV={args.HV} K={args.K} V={args.V}  dsu={DSU} intermediate={INTER}")

    # ---------------- single-config long run (for ncu/nsys external profiler) ----------------
    if args.profile is not None:
        N, T = args.profile
        q, k, v, a, b, A_log, dt_bias, state0, indices = make_dense_inputs(
            N, T, args.H, args.HV, args.K, args.V, device)
        scale = args.K ** -0.5
        if args.profile_variant == "triton":
            qt, kt, vt, at, bt, cu = to_triton_varlen(q, k, v, a, b)
            fn = make_triton_call(qt, kt, vt, at, bt, cu, A_log, dt_bias,
                                  state0.clone(), indices, scale, DSU, intermediate=INTER)
        elif args.profile_variant == "ws":
            fn = make_ws_call(q, k, v, a, b, A_log, dt_bias, state0.clone(), indices,
                              scale, DSU, intermediate=INTER)
        elif args.profile_variant == "auto":
            fn = make_auto_call(q, k, v, a, b, A_log, dt_bias, state0.clone(), indices,
                                scale, DSU, intermediate=INTER)
        elif args.profile_variant == "loop":
            fn = make_loop_call(q, k, v, a, b, A_log, dt_bias, state0.clone(), indices,
                                scale, True)
        elif args.profile_variant == "sbkv":
            fn = make_small_batch_call(q, k, v, a, b, A_log, dt_bias,
                                       state0.clone(), indices, scale, DSU, variant="kv")
        else:  # sbvk
            fn = make_small_batch_call(q, k, v, a, b, A_log, dt_bias,
                                       state0.clone(), indices, scale, DSU, variant="vk", intermediate=INTER)
        warmup(fn, args.warmup)
        print(f"[profile] variant={args.profile_variant} N={N} T={T}: "
              f"{args.profile_iters} forward iters (for external profiler to wrap)")
        torch.cuda.synchronize()
        for _ in range(args.profile_iters):
            fn()
        torch.cuda.synchronize()
        print("[profile] done.")
        return

    # ---------------- numerical check (all timed cases; +buffer compare when --intermediate) ----------------
    INTER_CHK = bool(args.intermediate)
    print("\n=== numerical check (max|Δ|, threshold 5e-2) ===")
    if INTER_CHK:
        print(f"{'N':>4} {'T':>3} | {'Δ sbvk-out':>12} | {'Δ sbvk-inter':>13} | {'Δ wsA-out':>12} | {'Δ wsA-inter':>13} | flag")
    else:
        print(f"{'N':>4} {'T':>3} | {'Δ sbkv-tri':>12} | {'Δ sbvk-tri':>12} | {'Δ auto-tri':>12} | flag")
    print("-" * 66)
    check_cases = [(N, T) for N in args.batch_sizes for T in args.Ts]
    ok_all = True
    for N, T in check_cases:
        q, k, v, a, b, A_log, dt_bias, state0, indices = make_dense_inputs(
            N, T, args.H, args.HV, args.K, args.V, device)
        qt, kt, vt, at, bt, cu = to_triton_varlen(q, k, v, a, b)
        scale = args.K ** -0.5
        if INTER_CHK:
            HV, V, K = args.HV, args.V, args.K
            idx32 = torch.arange(N, device=device, dtype=torch.int32)
            inter_tri = torch.zeros(N, T, HV, V, K, device=device, dtype=torch.float32)
            o_tri = fused_sigmoid_gating_delta_rule_update(
                A_log=A_log, a=at, dt_bias=dt_bias, softplus_beta=1.0, softplus_threshold=20.0,
                q=qt, k=kt, v=vt, b=bt, initial_state_source=state0.clone(), initial_state_indices=indices,
                scale=scale, use_qk_l2norm_in_kernel=True, cu_seqlens=cu, is_kda=True,
                disable_state_update=DSU, intermediate_states_buffer=inter_tri,
                intermediate_state_indices=idx32, cache_steps=T, retrieve_parent_token=None, lower_bound=None,
            ).reshape(N, T, HV, V).float()
            inter_vk = torch.zeros(N, T, HV, V, K, device=device, dtype=torch.float32)
            o_vk = kda_decode_mtp_small_batch(
                A_log=A_log, dt_bias=dt_bias, q=q, k=k, v=v, a=a, b=b,
                initial_state_source=state0.clone(), initial_state_indices=indices, scale=scale,
                use_qk_l2norm_in_kernel=True, softplus_beta=1.0, softplus_threshold=20.0,
                disable_state_update=DSU, variant="vk", bv=_VK_BV, intermediate_states_buffer=inter_vk,
            ).float()
            inter_ws = torch.zeros(N, T, HV, V, K, device=device, dtype=torch.float32)
            o_ws = kda_decode_mtp_ws(
                A_log=A_log, dt_bias=dt_bias, q=q, k=k, v=v, a=a, b=b,
                initial_state_source=state0.clone(), initial_state_indices=indices, scale=scale,
                use_qk_l2norm_in_kernel=True, softplus_beta=1.0, softplus_threshold=20.0,
                disable_state_update=DSU, intermediate_states_buffer=inter_ws,
            ).float()
            d_vk_o = (o_vk - o_tri).abs().max().item(); d_vk_i = (inter_vk - inter_tri).abs().max().item()
            d_ws_o = (o_ws - o_tri).abs().max().item(); d_ws_i = (inter_ws - inter_tri).abs().max().item()
            flag = "OK" if max(d_vk_o, d_vk_i, d_ws_o, d_ws_i) < 5e-2 else "DIFF!"
            if flag != "OK":
                ok_all = False
            print(f"{N:>4} {T:>3} | {d_vk_o:>12.2e} | {d_vk_i:>13.2e} | {d_ws_o:>12.2e} | {d_ws_i:>13.2e} | {flag}")
        else:
            o_tri = make_triton_call(qt, kt, vt, at, bt, cu, A_log, dt_bias,
                                     state0.clone(), indices, scale, True)()
            o_tri = o_tri.reshape(N, T, args.HV, args.V).float()
            o_sbkv = make_small_batch_call(q, k, v, a, b, A_log, dt_bias,
                                           state0.clone(), indices, scale, DSU, variant="kv")().float()
            o_sbvk = make_small_batch_call(q, k, v, a, b, A_log, dt_bias,
                                           state0.clone(), indices, scale, DSU, variant="vk")().float()
            o_auto = make_auto_call(q, k, v, a, b, A_log, dt_bias, state0.clone(), indices,
                                    scale, True)().float()
            d_sbkv = (o_sbkv - o_tri).abs().max().item()
            d_sbvk = (o_sbvk - o_tri).abs().max().item()
            d_auto = (o_auto - o_tri).abs().max().item()
            flag = "OK" if (d_sbkv < 5e-2 and d_sbvk < 5e-2 and d_auto < 5e-2) else "DIFF!"
            if flag != "OK":
                ok_all = False
            print(f"{N:>4} {T:>3} | {d_sbkv:>12.2e} | {d_sbvk:>12.2e} | {d_auto:>12.2e} | {flag}")

    if args.check or not ok_all:
        return

    # ---------------- performance (t_graph, kernel-only) ----------------
    print("\n=== performance t_graph (CUDA graph replay, kernel-only) ===")
    print(f"  wsAuto=ws heuristic; small_batch=1-warp; auto=dispatch(wu<=512->sbvk else ws);"
          f" loop=T x single-token  warmup={args.warmup} rep={args.rep}")
    if INTER:
        hdr = (f"{'N':>4} {'T':>3} | {'tg_tri':>7} {'tg_wsA':>7} {'tg_sbvk':>8} {'tg_auto':>8} | "
               f"{'sbvk/tri':>9} {'wsA/tri':>9} {'auto/tri':>9} | {'pick':>5} {'best':>5}")
    else:
        hdr = (f"{'N':>4} {'T':>3} | {'tg_loop':>8} {'tg_tri':>7} {'tg_wsA':>7} {'tg_sbkv':>8} {'tg_sbvk':>8} {'tg_auto':>8} | "
               f"{'sbvk/tri':>9} {'sbkv/tri':>9} {'auto/tri':>9} {'auto/loop':>10} | {'pick':>5} {'best':>5}")
    print(hdr)
    print("-" * len(hdr))
    for N in args.batch_sizes:
        gc = 1 if N >= 16 else args.graph_calls  # large batch forces gc=1 to save time
        for T in args.Ts:
            q, k, v, a, b, A_log, dt_bias, state0, indices = make_dense_inputs(
                N, T, args.H, args.HV, args.K, args.V, device)
            scale = args.K ** -0.5

            tg_tri = None
            if N * args.HV <= TRITON_MAX_GRID_Z:
                qt, kt, vt, at, bt, cu = to_triton_varlen(q, k, v, a, b)
                tri = make_triton_call(qt, kt, vt, at, bt, cu, A_log, dt_bias,
                                       state0.clone(), indices, scale, DSU, intermediate=INTER)
                try:
                    warmup(tri, args.warmup)
                    tg_tri = t_graph_ms(tri, 3, args.rep, gc)
                except Exception as e:
                    print(f"{N:>4} {T:>3} | triton FAIL: {str(e)[:50]}")

            makers = {
                "ws": make_ws_call(q, k, v, a, b, A_log, dt_bias, state0.clone(), indices, scale, DSU, intermediate=INTER),
                "sbvk": make_small_batch_call(q, k, v, a, b, A_log, dt_bias, state0.clone(), indices, scale, DSU, variant="vk", intermediate=INTER),
                "auto": make_auto_call(q, k, v, a, b, A_log, dt_bias, state0.clone(), indices, scale, DSU, intermediate=INTER),
            }
            if not INTER:
                makers["loop"] = make_loop_call(q, k, v, a, b, A_log, dt_bias, state0.clone(), indices, scale, True)
                makers["sbkv"] = make_small_batch_call(q, k, v, a, b, A_log, dt_bias, state0.clone(), indices, scale, DSU, variant="kv")
            tg = {}
            for name, fn_obj in makers.items():
                try:
                    warmup(fn_obj, args.warmup)
                    tg[name] = t_graph_ms(fn_obj, 3, args.rep, gc)
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

            pick = "sbvk" if N * args.HV <= 512 else "ws"
            cands = {nm: t for nm, t in tg.items() if t and nm != "loop"}
            best = min(cands, key=cands.get) if cands else "-"
            if INTER:
                print(f"{N:>4} {T:>3} | {us(tg_tri):>7} {us(tg.get('ws')):>7} {us(tg.get('sbvk')):>8} {us(tg.get('auto')):>8} | "
                      f"{r(tg.get('sbvk')):>9} {r(tg.get('ws')):>9} {r(tg.get('auto')):>9} | {pick:>5} {best:>5}")
            else:
                print(f"{N:>4} {T:>3} | {us(tg.get('loop')):>8} {us(tg_tri):>7} {us(tg.get('ws')):>7} "
                      f"{us(tg.get('sbkv')):>8} {us(tg.get('sbvk')):>8} {us(tg.get('auto')):>8} | "
                      f"{r(tg.get('sbvk')):>9} {r(tg.get('sbkv')):>9} {r(tg.get('auto')):>9} {rl(tg.get('auto')):>10} | "
                      f"{pick:>5} {best:>5}")


if __name__ == "__main__":
    main()
