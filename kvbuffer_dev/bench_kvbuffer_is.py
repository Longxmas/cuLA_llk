"""IS-enabled compare: KVBuffer (verify+flush) vs Triton recurrent WITH intermediate states.

Production-faithful spec-decode comparison. The recurrent baseline is the Triton
operator (benchmarks/fused_sigmoid_gating_recurrent.py) in *target_verify* mode:
it writes T per-token intermediate states to `intermediate_states_buffer`
([N,T,HV,V,K] fp32) — the HBM cost KVBuffer is designed to avoid. (The small_batch
vk/kv ops can't store intermediate states, so they are NOT a baseline here.)

  recurrent (triton+IS) : verify + write T·d² states   (one fused kernel)
  KVBuffer              : verify (chunkwise, 0 state writes) + flush (1·d² over accepted)

SELF-CONTAINED on purpose: inlines the small input/timing helpers instead of
importing the benchmark module (which the cleanup agent is actively renaming /
refactoring), so this dev script survives that churn. Only hard deps = the triton
operator file + the kvbuffer op. Run on GB200:

  python kvbuffer_dev/bench_kvbuffer_is.py --check
  python kvbuffer_dev/bench_kvbuffer_is.py --batch-sizes 1 4 16 --Ts 2 4 6 8 --HV 32
"""

import argparse
import importlib.util
import os
import pathlib
import sys

import torch

os.environ.setdefault("FLA_USE_FAST_OPS", os.getenv("CULA_USE_FAST_MATH", "1"))
_here = pathlib.Path(__file__).resolve().parent
_repo = _here.parent
sys.path.insert(0, str(_repo))  # cuLA repo root -> import cula.ops.*

from cula.ops.kda_decode_mtp_kvbuffer import (  # noqa: E402
    kda_decode_mtp_kvbuffer,
    kda_flush_kvbuffer,
)

# ---- Triton recurrent baseline: load the local operator file directly ----
_TRITON_FILE = os.environ.get(
    "KDA_TRITON_FILE", str(_repo / "benchmarks" / "fused_sigmoid_gating_recurrent.py")
)
_spec = importlib.util.spec_from_file_location("_kda_triton_standalone", _TRITON_FILE)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
fused_sigmoid_gating_delta_rule_update = _mod.fused_sigmoid_gating_delta_rule_update


# ---- inlined helpers (stable copies; do not import from the bench module) ----
def make_dense_inputs(N, T, H, HV, K, V, device, seed=42):
    g = torch.Generator(device=device).manual_seed(seed)
    bf16 = torch.bfloat16
    q = torch.randn(N, T, H, K, device=device, dtype=bf16, generator=g)
    k = torch.randn(N, T, H, K, device=device, dtype=bf16, generator=g)
    v = torch.randn(N, T, HV, V, device=device, dtype=bf16, generator=g)
    a = (torch.randn(N, T, HV, K, device=device, dtype=torch.float32, generator=g) * 0.1).to(bf16)
    b = torch.randn(N, T, HV, device=device, dtype=bf16, generator=g)
    A_log = -torch.rand(HV, device=device, dtype=torch.float32, generator=g) * 2
    dt_bias = torch.randn(HV, K, device=device, dtype=torch.float32, generator=g) * 0.1
    state = torch.randn(N, HV, V, K, device=device, dtype=torch.float32, generator=g) * 0.01
    indices = torch.arange(N, device=device, dtype=torch.int32)
    return q, k, v, a, b, A_log, dt_bias, state, indices


def to_triton_varlen(q, k, v, a, b):
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


def warmup(fn, n):
    for _ in range(n):
        fn()
    torch.cuda.synchronize()


def t_graph_ms(fn, warmup_iters, rep):
    """Kernel-only timing via CUDA graph capture + replay."""
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


# ---- call wrappers ----
def triton_call(q, k, v, a, b, A_log, dt_bias, state, indices, scale, *, with_is, N, T, HV, V, K):
    """Triton recurrent. with_is=True -> target_verify mode writing T intermediate states."""
    qt, kt, vt, at, bt, cu = to_triton_varlen(q, k, v, a, b)
    if with_is:
        inter = torch.empty(N, T, HV, V, K, dtype=torch.float32, device=q.device)
        inter_idx = torch.arange(N, device=q.device, dtype=torch.int32)
        cache_steps = T
    else:
        inter, inter_idx, cache_steps = None, None, None

    def call():
        return fused_sigmoid_gating_delta_rule_update(
            A_log=A_log, a=at, dt_bias=dt_bias, softplus_beta=1.0, softplus_threshold=20.0,
            q=qt, k=kt, v=vt, b=bt, initial_state_source=state, initial_state_indices=indices,
            scale=scale, use_qk_l2norm_in_kernel=True, cu_seqlens=cu, is_kda=True,
            disable_state_update=True,
            intermediate_states_buffer=inter, intermediate_state_indices=inter_idx,
            cache_steps=cache_steps,
        )

    return call, inter


def kvb_verify_call(q, k, v, a, b, A_log, dt_bias, state, indices, scale):
    def call():
        return kda_decode_mtp_kvbuffer(
            A_log, dt_bias, q, k, v, a, b, state, indices, scale=scale,
            use_qk_l2norm_in_kernel=True, disable_state_update=True, emit_output=True,
        )
    return call


def kvb_flush_call(q, k, v, a, b, A_log, dt_bias, state, indices, scale):
    def call():
        return kda_decode_mtp_kvbuffer(
            A_log, dt_bias, q, k, v, a, b, state, indices, scale=scale,
            use_qk_l2norm_in_kernel=True, disable_state_update=False, emit_output=False,
        )
    return call


def kvb_full_call(q, k, v, a, b, A_log, dt_bias, state, indices, scale):
    # full-accept: 一个 kernel 同时出输出 + 更新最终态(不拆 verify/flush,避免重算 Phase A+B)
    def call():
        return kda_decode_mtp_kvbuffer(
            A_log, dt_bias, q, k, v, a, b, state, indices, scale=scale,
            use_qk_l2norm_in_kernel=True, disable_state_update=False, emit_output=True,
        )
    return call


def kvb_flush_m_call(q, k, v, a, b, A_log, dt_bias, state, indices, scale, m):
    # flush over first m accepted tokens (chain): 输入切到 [:, :m],只更新态,跳过输出
    qm, km, vm, am, bm = q[:, :m], k[:, :m], v[:, :m], a[:, :m], b[:, :m]

    def call():
        return kda_decode_mtp_kvbuffer(
            A_log, dt_bias, qm, km, vm, am, bm, state, indices, scale=scale,
            use_qk_l2norm_in_kernel=True, disable_state_update=False, emit_output=False,
        )
    return call


def _alloc_ubuf(N, T, HV, V, K, device):
    return (
        torch.empty(N, T, HV, V, dtype=torch.float32, device=device),
        torch.empty(N, T, HV, K, dtype=torch.float32, device=device),
        torch.empty(N, T, HV, K, dtype=torch.float32, device=device),
    )


def kvb_verify_ubuf_call(q, k, v, a, b, A_log, dt_bias, state, indices, scale, ubuf):
    # verify: 出输出 + 写紧凑 u-buffer(不提交 state)
    u_b, kinv_b, b_b = ubuf

    def call():
        return kda_decode_mtp_kvbuffer(
            A_log, dt_bias, q, k, v, a, b, state, indices, scale=scale,
            use_qk_l2norm_in_kernel=True, disable_state_update=True, emit_output=True,
            u_buffer=u_b, kinv_buffer=kinv_b, b_buffer=b_b,
        )
    return call


def kvb_flush_ubuf_call(state, indices, ubuf, m):
    # flush: 读 u-buffer 对前 m 个 token 做 rank-m 更新 → S_m(无重算)
    u_b, kinv_b, b_b = ubuf

    def call():
        return kda_flush_kvbuffer(state, indices, u_b, kinv_b, b_b, m)
    return call


def recurrent_commit_call(state, inter, m):
    # recurrent rollback (hybrid_linear_attn_backend.update_mamba_state_after_mtp_verify):
    # gather 第 m 份预存中间态 → 持久 ssm 池(纯 copy,无计算)。inter=[N,T,HV,V,K]。
    midx = m - 1

    def call():
        state.copy_(inter[:, midx])
        return state
    return call


def _accept_len(T, accept):
    if accept == "full":
        return T
    if accept == "half":
        return max(1, (T + 1) // 2)
    if accept == "one":
        return 1
    return max(1, min(int(accept), T))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--Ts", type=int, nargs="+", default=[2, 4, 6, 8])
    ap.add_argument("--H", type=int, default=16)
    ap.add_argument("--HV", type=int, default=32)
    ap.add_argument("--K", type=int, default=128)
    ap.add_argument("--V", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--rep", type=int, default=300)
    ap.add_argument("--check", action="store_true", help="只数值校验,不计时")
    ap.add_argument("--atol", type=float, default=5e-2)
    ap.add_argument("--accept", default="full",
                    help="链式接受长度 m: full(=T) / half / one / 整数。real serving 是 per-req 变长,这里取单一代表值")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "需要 CUDA"
    device = "cuda"

    # ---------------- 正确性: verify 输出 + buffer-u flush 各个 m 的 S_m ----------------
    print(f"\n=== 正确性 (max|Δ|, 阈值 {args.atol:.0e}) HV={args.HV} K={args.K} V={args.V} ===")
    print("  o:kvb-tri = verify 输出 vs triton;  Sm:flush-tri = buffer-u flush(m) 的 S_m vs triton 第 m 份中间态")
    print(f"{'N':>4} {'T':>3} | {'o:kvb-tri':>11} {'Sm@T':>10} {'Sm@half':>10} {'Sm@1':>10} | flag")
    print("-" * 60)
    ok_all = True
    for N in args.batch_sizes:
        for T in args.Ts:
            q, k, v, a, b, A_log, dt_bias, state0, indices = make_dense_inputs(
                N, T, args.H, args.HV, args.K, args.V, device)
            scale = args.K ** -0.5

            tri_call, inter = triton_call(q, k, v, a, b, A_log, dt_bias, state0.clone(), indices,
                                          scale, with_is=True, N=N, T=T, HV=args.HV, V=args.V, K=args.K)
            o_tri = tri_call().reshape(N, T, args.HV, args.V).float()

            # verify(+写 u-buffer)
            ubuf = _alloc_ubuf(N, T, args.HV, args.V, args.K, device)
            o_kvb = kvb_verify_ubuf_call(q, k, v, a, b, A_log, dt_bias, state0.clone(), indices, scale, ubuf)().float()
            d_o = (o_kvb - o_tri).abs().max().item()

            # buffer-u flush 在 m = T / half / 1 各重建 S_m,对标 triton 第 m 份中间态
            ds = {}
            for tag, mm in (("T", T), ("half", max(1, (T + 1) // 2)), ("1", 1)):
                s_f = state0.clone()
                kvb_flush_ubuf_call(s_f, indices, ubuf, mm)()
                tri_m = inter[:, mm - 1].float()
                ds[tag] = (s_f.float() - tri_m).abs().max().item()

            ok = d_o < args.atol and max(ds.values()) < args.atol
            ok_all = ok_all and ok
            print(f"{N:>4} {T:>3} | {d_o:>11.2e} {ds['T']:>10.2e} {ds['half']:>10.2e} {ds['1']:>10.2e} | "
                  f"{'OK' if ok else 'DIFF!'}")
    print("正确性:", "全部 OK" if ok_all else "有 DIFF,先查正确性!")

    if args.check or not ok_all:
        return

    # ---------------- 性能 ----------------
    print("\n=== 性能 t_graph (CUDA graph, kernel-only) us  —— 真实链式 verify→定m→commit (buffer-u) ===")
    print(f"  接受长度 m = {args.accept} (链式;real serving per-req 变长,这里取单一代表值)")
    print("  REC = triton verify(写T份态) + commit(gather第m份→池, 纯copy)")
    print("  KVB = kvbuffer verify(出输出+写紧凑u-buffer,不提交态) + flush(读u-buffer rank-m重建S_m)")
    print("  flush_rc = 旧版重算flush(对照, 看buffer-u省了多少);  spd = REC/KVB")
    hdr = (f"{'N':>4} {'T':>3} {'m':>3} | {'v_tri':>6} {'commit':>6} {'REC':>6} | "
           f"{'v_kvb':>6} {'flush':>6} {'KVB':>6} | {'flush_rc':>8} | {'spd':>6}")
    print(hdr)
    print("-" * len(hdr))
    for N in args.batch_sizes:
        for T in args.Ts:
            q, k, v, a, b, A_log, dt_bias, state0, indices = make_dense_inputs(
                N, T, args.H, args.HV, args.K, args.V, device)
            scale = args.K ** -0.5
            m = _accept_len(T, args.accept)
            ubuf = _alloc_ubuf(N, T, args.HV, args.V, args.K, device)

            # triton verify(写 T 份中间态) + 预填 inter 供 commit 计时
            tri_is, inter = triton_call(q, k, v, a, b, A_log, dt_bias, state0.clone(), indices,
                                        scale, with_is=True, N=N, T=T, HV=args.HV, V=args.V, K=args.K)
            makers = {
                "v_tri": tri_is,
                "commit": recurrent_commit_call(state0.clone(), inter, m),
                "v_kvb": kvb_verify_ubuf_call(q, k, v, a, b, A_log, dt_bias, state0.clone(), indices, scale, ubuf),
                "flush": kvb_flush_ubuf_call(state0.clone(), indices, ubuf, m),
                "flush_rc": kvb_flush_m_call(q, k, v, a, b, A_log, dt_bias, state0.clone(), indices, scale, m),
            }
            tg = {}
            for name, fn_obj in makers.items():
                try:
                    warmup(fn_obj, args.warmup)
                    tg[name] = t_graph_ms(fn_obj, 3, args.rep)
                except Exception as e:
                    tg[name] = None
                    print(f"{N:>4} {T:>3} | {name} FAIL: {str(e)[:60]}")

            def us(x):
                return f"{x * 1e3:.1f}" if x else "n/a"

            rec = (tg.get("v_tri") or 0) + (tg.get("commit") or 0)
            kvb = (tg.get("v_kvb") or 0) + (tg.get("flush") or 0)
            spd = f"{rec / kvb:.2f}x" if (rec and kvb) else "n/a"
            print(f"{N:>4} {T:>3} {m:>3} | {us(tg.get('v_tri')):>6} {us(tg.get('commit')):>6} {us(rec):>6} | "
                  f"{us(tg.get('v_kvb')):>6} {us(tg.get('flush')):>6} {us(kvb):>6} | {us(tg.get('flush_rc')):>8} | "
                  f"{spd:>6}")


if __name__ == "__main__":
    main()
