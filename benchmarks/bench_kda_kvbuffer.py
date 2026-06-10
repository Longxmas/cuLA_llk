"""KVBuffer (chunkwise) vs production recurrent vk/ws — apples-to-apples bench.

Goal: show vk-kvbuffer beats the production vk (kda_decode_mtp_small_batch
variant='vk') and ws-kvbuffer beats the production ws (kda_decode_mtp_ws), on
the SAME inputs / SAME timing harness as benchmarks/bench_kda_decode_mtp.py
(CUDA-graph capture+replay, --graph-calls amortization, HV=64, dsu).

Self-contained (inlines make_dense_inputs / to_triton_varlen / make_triton_call
/ t_graph_ms) so it survives renames of the production bench module. Triton
recurrent baseline loaded from KDA_TRITON_FILE (the standalone
fused_sigmoid_gating_recurrent.py); used only for the numerical check here.

Columns:
  tg_tri  tg_vk  tg_vkkvb [tg_ws tg_wskvb] | vk/vkkvb  ws/wskvb | vk/tri vkkvb/tri
where vk/vkkvb = speedup of vk-kvbuffer over vk (>1 = kvbuffer wins); same for ws.
"""

import argparse
import os
import sys

import torch

from cula.ops.kda_decode_mtp import (
    kda_decode_mtp_small_batch,
    kda_decode_mtp_ws,
)
from cula.ops.kda_decode_mtp_kvbuffer import kda_decode_mtp_kvbuffer

# ws-kvbuffer is added later; import is optional so the vk side still benches.
try:
    from cula.ops.kda_decode_mtp_kvbuffer import kda_decode_mtp_ws_kvbuffer
    _HAVE_WSKVB = True
except Exception:
    _HAVE_WSKVB = False

# ---------------- Triton recurrent baseline (numerical check only) ----------------
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


def make_triton_call(qt, kt, vt, at, bt, cu_seqlens, A_log, dt_bias, state, indices, scale, dsu):
    def call():
        return fused_sigmoid_gating_delta_rule_update(
            A_log=A_log, a=at, dt_bias=dt_bias, softplus_beta=1.0, softplus_threshold=20.0,
            q=qt, k=kt, v=vt, b=bt, initial_state_source=state, initial_state_indices=indices,
            scale=scale, use_qk_l2norm_in_kernel=True, cu_seqlens=cu_seqlens, is_kda=True,
            disable_state_update=dsu, intermediate_states_buffer=None, intermediate_state_indices=None,
            cache_steps=None, retrieve_parent_token=None, lower_bound=None,
        )
    return call


def warmup(fn, n):
    for _ in range(n):
        fn()
    torch.cuda.synchronize()


def t_graph_ms(fn, warmup_iters, rep, graph_calls=1):
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


_VK_BV = -1


def make_vk_call(q, k, v, a, b, A_log, dt_bias, state, indices, scale, dsu):
    """Production recurrent vk (kda_decode_mtp_small_batch variant='vk')."""
    def call():
        return kda_decode_mtp_small_batch(
            A_log=A_log, dt_bias=dt_bias, q=q, k=k, v=v, a=a, b=b,
            initial_state_source=state, initial_state_indices=indices, scale=scale,
            use_qk_l2norm_in_kernel=True, softplus_beta=1.0, softplus_threshold=20.0,
            disable_state_update=dsu, variant="vk", bv=_VK_BV,
        )
    return call


def make_vkkvb_call(q, k, v, a, b, A_log, dt_bias, state, indices, scale, dsu):
    """vk-kvbuffer (chunkwise verify; emit_output=True, no u-buffer for pure fwd perf)."""
    def call():
        return kda_decode_mtp_kvbuffer(
            A_log=A_log, dt_bias=dt_bias, q=q, k=k, v=v, a=a, b=b,
            initial_state_source=state, initial_state_indices=indices, scale=scale,
            use_qk_l2norm_in_kernel=True, softplus_beta=1.0, softplus_threshold=20.0,
            disable_state_update=dsu, emit_output=True, bv=_VK_BV,
        )
    return call


def make_ws_call(q, k, v, a, b, A_log, dt_bias, state, indices, scale, dsu):
    """Production recurrent ws (kda_decode_mtp_ws, auto tile_v/ilp/smem_v)."""
    def call():
        return kda_decode_mtp_ws(
            A_log=A_log, dt_bias=dt_bias, q=q, k=k, v=v, a=a, b=b,
            initial_state_source=state, initial_state_indices=indices, scale=scale,
            use_qk_l2norm_in_kernel=True, softplus_beta=1.0, softplus_threshold=20.0,
            disable_state_update=dsu,
        )
    return call


def make_wskvb_call(q, k, v, a, b, A_log, dt_bias, state, indices, scale, dsu):
    """ws-kvbuffer (warp-spec chunkwise) — only if implemented."""
    def call():
        return kda_decode_mtp_ws_kvbuffer(
            A_log=A_log, dt_bias=dt_bias, q=q, k=k, v=v, a=a, b=b,
            initial_state_source=state, initial_state_indices=indices, scale=scale,
            use_qk_l2norm_in_kernel=True, softplus_beta=1.0, softplus_threshold=20.0,
            disable_state_update=dsu, emit_output=True,
        )
    return call


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--Ts", type=int, nargs="+", default=[2, 3, 4, 6, 8])
    ap.add_argument("--H", type=int, default=16)
    ap.add_argument("--HV", type=int, default=64)
    ap.add_argument("--K", type=int, default=128)
    ap.add_argument("--V", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--rep", type=int, default=300)
    ap.add_argument("--graph-calls", type=int, default=4,
                    help="fire K ops per graph to amortize fixed overhead (needs idempotent dsu=1)")
    ap.add_argument("--dsu", type=int, default=1, choices=[0, 1],
                    help="disable_state_update; 1=forward-only (idempotent, default), 0=write state")
    ap.add_argument("--vk-bv", type=int, default=-1, choices=[-1, 8, 16, 32])
    ap.add_argument("--ws", type=int, default=1, choices=[0, 1], help="also bench ws / ws-kvbuffer")
    ap.add_argument("--check", action="store_true", help="numerical check only, no timing")
    ap.add_argument("--atol", type=float, default=5e-2)
    args = ap.parse_args()

    global _VK_BV
    _VK_BV = args.vk_bv
    DSU = bool(args.dsu)
    device = "cuda"
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"shape H={args.H} HV={args.HV} K={args.K} V={args.V}  dsu={DSU} "
          f"graph_calls={args.graph_calls} ws={args.ws} wskvb_impl={_HAVE_WSKVB}")

    # ---------------- numerical check (vs Triton recurrent) ----------------
    if not _HAVE_TRITON:
        print(f"[warn] Triton baseline unavailable ({_TRITON_ERR}); skipping numerical check.")
    else:
        print("\n=== numerical check (max|Δ| vs Triton recurrent, threshold "
              f"{args.atol}) ===")
        print(f"{'N':>4} {'T':>3} | {'Δ vk':>10} | {'Δ vkkvb':>10} | "
              f"{'Δ ws':>10} | {'Δ wskvb':>10} | flag")
        for N in args.batch_sizes:
            for T in args.Ts:
                q, k, v, a, b, A_log, dt_bias, state0, indices = make_dense_inputs(
                    N, T, args.H, args.HV, args.K, args.V, device)
                scale = args.K ** -0.5
                qt, kt, vt, at, bt, cu = to_triton_varlen(q, k, v, a, b)
                o_tri = make_triton_call(qt, kt, vt, at, bt, cu, A_log, dt_bias,
                                         state0.clone(), indices, scale, True)()
                o_tri = o_tri.reshape(N, T, args.HV, args.V)
                o_vk = make_vk_call(q, k, v, a, b, A_log, dt_bias,
                                    state0.clone(), indices, scale, True)()
                o_vkkvb = make_vkkvb_call(q, k, v, a, b, A_log, dt_bias,
                                          state0.clone(), indices, scale, True)()
                d_vk = (o_vk - o_tri).abs().max().item()
                d_vkkvb = (o_vkkvb - o_tri).abs().max().item()
                d_ws = float("nan")
                d_wskvb = float("nan")
                if args.ws:
                    o_ws = make_ws_call(q, k, v, a, b, A_log, dt_bias,
                                        state0.clone(), indices, scale, True)()
                    d_ws = (o_ws - o_tri).abs().max().item()
                    if _HAVE_WSKVB:
                        o_wskvb = make_wskvb_call(q, k, v, a, b, A_log, dt_bias,
                                                  state0.clone(), indices, scale, True)()
                        d_wskvb = (o_wskvb - o_tri).abs().max().item()
                cand = [x for x in (d_vk, d_vkkvb, d_ws, d_wskvb) if x == x]
                flag = "OK" if max(cand) < args.atol else "DIFF!"
                print(f"{N:>4} {T:>3} | {d_vk:>10.2e} | {d_vkkvb:>10.2e} | "
                      f"{d_ws:>10.2e} | {d_wskvb:>10.2e} | {flag}")

    if args.check:
        return

    # ---------------- timing ----------------
    print("\n=== latency (us, CUDA-graph) ===")
    hdr = (f"{'N':>4} {'T':>3} | {'tg_vk':>8} {'tg_vkkvb':>9}")
    if args.ws:
        hdr += f" {'tg_ws':>8} {'tg_wskvb':>9}"
    hdr += f" | {'vk/vkkvb':>9}"
    if args.ws:
        hdr += f" {'ws/wskvb':>9}"
    print(hdr)
    for N in args.batch_sizes:
        for T in args.Ts:
            q, k, v, a, b, A_log, dt_bias, state0, indices = make_dense_inputs(
                N, T, args.H, args.HV, args.K, args.V, device)
            scale = args.K ** -0.5
            gc = 1 if N >= 16 else args.graph_calls
            tg = {}
            fn_vk = make_vk_call(q, k, v, a, b, A_log, dt_bias, state0.clone(), indices, scale, DSU)
            fn_vkkvb = make_vkkvb_call(q, k, v, a, b, A_log, dt_bias, state0.clone(), indices, scale, DSU)
            warmup(fn_vk, 3); warmup(fn_vkkvb, 3)
            tg["vk"] = t_graph_ms(fn_vk, 3, args.rep, gc)
            tg["vkkvb"] = t_graph_ms(fn_vkkvb, 3, args.rep, gc)
            if args.ws:
                fn_ws = make_ws_call(q, k, v, a, b, A_log, dt_bias, state0.clone(), indices, scale, DSU)
                warmup(fn_ws, 3)
                tg["ws"] = t_graph_ms(fn_ws, 3, args.rep, gc)
                if _HAVE_WSKVB:
                    fn_wskvb = make_wskvb_call(q, k, v, a, b, A_log, dt_bias, state0.clone(), indices, scale, DSU)
                    warmup(fn_wskvb, 3)
                    tg["wskvb"] = t_graph_ms(fn_wskvb, 3, args.rep, gc)

            def us(x):
                return f"{x * 1e3:.1f}" if x else "n/a"

            def ratio(num, den):
                a_, b_ = tg.get(num), tg.get(den)
                return f"{a_ / b_:.2f}x" if (a_ and b_) else "n/a"

            row = f"{N:>4} {T:>3} | {us(tg.get('vk')):>8} {us(tg.get('vkkvb')):>9}"
            if args.ws:
                row += f" {us(tg.get('ws')):>8} {us(tg.get('wskvb')):>9}"
            row += f" | {ratio('vk', 'vkkvb'):>9}"  # >1 = kvbuffer faster than vk
            if args.ws:
                row += f" {ratio('ws', 'wskvb'):>9}"
            print(row)


if __name__ == "__main__":
    main()
