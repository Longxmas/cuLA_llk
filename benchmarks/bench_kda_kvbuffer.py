"""KVBuffer (chunkwise) vs production recurrent vk/ws — apples-to-apples bench.

Goal: vk-kvbuffer beats production vk (kda_decode_mtp_small_batch variant='vk')
and ws-kvbuffer beats production ws (kda_decode_mtp_ws), same inputs / timing
harness (CUDA-graph capture+replay).

Two modes:
  --verify 0 (default): pure-forward latency, compute only (no rollback I/O).
  --verify 1: fair spec-decode verify CHAIN. REC = recurrent verify (writes T·d²
    intermediate states) + commit; KVB = kvbuffer verify (emits output + writes a
    compact u-buffer) + flush (rank-m rebuild of S_m). spd = REC / KVB.

The commit step uses the REAL sglang kernel fused_mamba_state_scatter_with_mask
(loaded from KDA_SCATTER_FILE, same importlib trick as the Triton baseline) so
the recurrent rollback cost is measured with official code, not a model.

Self-contained (inlines input/timing helpers) to survive renames of the
production bench module. Triton recurrent baseline (numerical check only) from
KDA_TRITON_FILE; scatter commit from KDA_SCATTER_FILE.
"""

import argparse
import importlib.util
import os

import torch

from cula.ops.kda_decode_mtp import (
    kda_decode_mtp_small_batch,
    kda_decode_mtp_ws,
)
from cula.ops.kda_decode_mtp_kvbuffer import kda_decode_mtp_kvbuffer, kda_flush_kvbuffer

# ws-kvbuffer is optional so the vk side still benches if it is absent.
try:
    from cula.ops.kda_decode_mtp_kvbuffer import kda_decode_mtp_ws_kvbuffer
    _HAVE_WSKVB = True
except Exception:
    _HAVE_WSKVB = False


def _load_from_file(path, attr):
    """Load a single attribute from a standalone .py file via importlib."""
    spec = importlib.util.spec_from_file_location(f"_standalone_{attr}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, attr)


# Triton recurrent baseline (numerical check only).
_HAVE_TRITON, _TRITON_ERR = True, ""
fused_sigmoid_gating_delta_rule_update = None
try:
    _f = os.environ.get("KDA_TRITON_FILE", "")
    if _f and os.path.exists(_f):
        fused_sigmoid_gating_delta_rule_update = _load_from_file(
            _f, "fused_sigmoid_gating_delta_rule_update")
    else:
        from sglang.srt.layers.attention.fla.fused_sigmoid_gating_recurrent import (
            fused_sigmoid_gating_delta_rule_update,
        )
except Exception as e:
    _HAVE_TRITON, _TRITON_ERR = False, repr(e)

# Official sglang scatter commit (update_mamba_state_after_mtp_verify). Same
# importlib trick as the Triton baseline; from KDA_SCATTER_FILE.
_HAVE_SCATTER, _SCATTER_ERR = True, ""
fused_mamba_state_scatter_with_mask = None
try:
    _f = os.environ.get("KDA_SCATTER_FILE", "")
    if _f and os.path.exists(_f):
        fused_mamba_state_scatter_with_mask = _load_from_file(
            _f, "fused_mamba_state_scatter_with_mask")
    else:
        from sglang.srt.layers.attention.mamba.mamba_state_scatter_triton import (
            fused_mamba_state_scatter_with_mask,
        )
except Exception as e:
    _HAVE_SCATTER, _SCATTER_ERR = False, repr(e)


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


def make_vk_call(q, k, v, a, b, A_log, dt_bias, state, indices, scale, dsu, inter_buf=None):
    """Production recurrent vk. In verify mode (inter_buf set) it writes the T·d²
    intermediate_states_buffer — the rollback cost kvbuffer replaces with a u-buffer."""
    def call():
        return kda_decode_mtp_small_batch(
            A_log=A_log, dt_bias=dt_bias, q=q, k=k, v=v, a=a, b=b,
            initial_state_source=state, initial_state_indices=indices, scale=scale,
            use_qk_l2norm_in_kernel=True, softplus_beta=1.0, softplus_threshold=20.0,
            disable_state_update=dsu, variant="vk", bv=_VK_BV,
            intermediate_states_buffer=inter_buf,
        )
    return call


def make_vkkvb_call(q, k, v, a, b, A_log, dt_bias, state, indices, scale, dsu, ubufs=None):
    """vk-kvbuffer chunkwise verify. In verify mode (ubufs set) it writes the compact
    u-buffer (~2Td+Td, ~43x smaller than T·d² states) for flush to rebuild any S_m."""
    u_buf, kinv_buf, b_buf = (ubufs if ubufs is not None else (None, None, None))
    def call():
        return kda_decode_mtp_kvbuffer(
            A_log=A_log, dt_bias=dt_bias, q=q, k=k, v=v, a=a, b=b,
            initial_state_source=state, initial_state_indices=indices, scale=scale,
            use_qk_l2norm_in_kernel=True, softplus_beta=1.0, softplus_threshold=20.0,
            disable_state_update=dsu, emit_output=True, bv=_VK_BV,
            u_buffer=u_buf, kinv_buffer=kinv_buf, b_buffer=b_buf,
        )
    return call


def make_ws_call(q, k, v, a, b, A_log, dt_bias, state, indices, scale, dsu, inter_buf=None):
    """Production recurrent ws. In verify mode (inter_buf set) it also writes T·d² states."""
    def call():
        return kda_decode_mtp_ws(
            A_log=A_log, dt_bias=dt_bias, q=q, k=k, v=v, a=a, b=b,
            initial_state_source=state, initial_state_indices=indices, scale=scale,
            use_qk_l2norm_in_kernel=True, softplus_beta=1.0, softplus_threshold=20.0,
            disable_state_update=dsu,
            intermediate_states_buffer=inter_buf,
        )
    return call


def make_wskvb_call(q, k, v, a, b, A_log, dt_bias, state, indices, scale, dsu, ubufs=None):
    """ws-kvbuffer (warp-spec chunkwise) — only if implemented; same u-buffer as vk-kvbuffer.
    tile_v / ilp_rows overridable via env KDA_WSKVB_TILE_V / KDA_WSKVB_ILP_ROWS (-1 = auto) for tuning."""
    u_buf, kinv_buf, b_buf = (ubufs if ubufs is not None else (None, None, None))
    _tv = int(os.environ.get("KDA_WSKVB_TILE_V", "-1"))
    _ilp = int(os.environ.get("KDA_WSKVB_ILP_ROWS", "-1"))
    _smemv = int(os.environ.get("KDA_WSKVB_SMEM_V", "-1"))
    def call():
        return kda_decode_mtp_ws_kvbuffer(
            A_log=A_log, dt_bias=dt_bias, q=q, k=k, v=v, a=a, b=b,
            initial_state_source=state, initial_state_indices=indices, scale=scale,
            use_qk_l2norm_in_kernel=True, softplus_beta=1.0, softplus_threshold=20.0,
            disable_state_update=dsu, emit_output=True,
            u_buffer=u_buf, kinv_buffer=kinv_buf, b_buffer=b_buf,
            tile_v=_tv, ilp_rows=_ilp, use_smem_v=_smemv,
        )
    return call


# ---- verify-chain components: commit (recurrent rollback) & flush (kvbuffer) ----
def make_scatter_commit_call(state_pool, inter_buf, m, N, T, HV, V, K):
    """Recurrent rollback via the OFFICIAL sglang fused_mamba_state_scatter_with_mask:
    gather each request's accepted-step state from the intermediate cache into the pool
    (num_layers=1; step = m-1 for all requests)."""
    dst = state_pool.view(1, N, HV, V, K)            # [layers, cache, *state]
    src = inter_buf.view(1, N, T, HV, V, K)          # [layers, req, step, *state]
    dst_idx = torch.arange(N, device=state_pool.device, dtype=torch.int32)
    step_idx = torch.full((N,), m - 1, device=state_pool.device, dtype=torch.int32)
    def call():
        fused_mamba_state_scatter_with_mask(dst, src, dst_idx, step_idx)
        return state_pool
    return call


def make_gather_commit_call(state_pool, inter_buf, m):
    """Recurrent rollback, strided gather model: copy inter_buf[:,m-1] (a T-strided view)
    into the pool. Less coalesced than the official kernel — kept for sensitivity only."""
    midx = m - 1
    def call():
        state_pool.copy_(inter_buf[:, midx])
        return state_pool
    return call


def make_flush_call(state_pool, indices, ubufs, m):
    """KVBuffer flush: read the compact u-buffer, rank-m rebuild S_m (no recompute)."""
    u_b, kinv_b, b_b = ubufs
    def call():
        return kda_flush_kvbuffer(state_pool, indices, u_b, kinv_b, b_b, m)
    return call


def _accept_len(T, accept, N=0):
    if accept == "full":
        return T
    if accept == "half":
        return max(1, (T + 1) // 2)
    if accept == "one":
        return 1
    if accept == "random":
        # Deterministic per-(N,T) accept length in [1,T] (real serving is per-req variable).
        g = torch.Generator().manual_seed(1000 * N + T)
        return int(torch.randint(1, T + 1, (1,), generator=g).item())
    return max(1, min(int(accept), T))


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
                    help="ops per graph to amortize fixed overhead (needs idempotent dsu=1)")
    ap.add_argument("--dsu", type=int, default=1, choices=[0, 1],
                    help="disable_state_update; 1=forward-only (idempotent, default), 0=write state")
    ap.add_argument("--vk-bv", type=int, default=-1, choices=[-1, 8, 16, 32])
    ap.add_argument("--ws", type=int, default=1, choices=[0, 1], help="also bench ws / ws-kvbuffer")
    ap.add_argument("--verify", type=int, default=0, choices=[0, 1],
                    help="fair spec-decode verify-CHAIN mode (REC=verify+commit, KVB=verify+flush, "
                         "spd=REC/KVB). 0=pure forward (compute only).")
    ap.add_argument("--accept", default="random",
                    help="chain accept length m: full(=T)/half/one/random/<int>; drives commit/flush.")
    ap.add_argument("--commit", default="scatter", choices=["scatter", "gather"],
                    help="recurrent commit model: scatter=official sglang "
                         "fused_mamba_state_scatter_with_mask (coalesced N·d², default); "
                         "gather=strided copy (sensitivity). kvbuffer flush always counted.")
    ap.add_argument("--check", action="store_true", help="numerical check only, no timing")
    ap.add_argument("--atol", type=float, default=5e-2)
    args = ap.parse_args()

    global _VK_BV
    _VK_BV = args.vk_bv
    DSU = bool(args.dsu)
    VERIFY = bool(args.verify)
    device = "cuda"
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"shape H={args.H} HV={args.HV} K={args.K} V={args.V}  dsu={DSU} verify={VERIFY} "
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

    def us(x):
        return f"{x * 1e3:.1f}" if x else "n/a"

    if VERIFY:
        _timing_verify_chain(args, DSU, device)
        return

    # ---------------- timing: pure forward (compute only) ----------------
    print("\n=== latency (us, CUDA-graph) — pure forward (compute only) ===")
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


def _timing_verify_chain(args, DSU, device):
    """Fair spec-decode verify CHAIN (each segment timed in its own CUDA graph, summed):
    REC = recurrent verify (writes T·d² states) + commit; KVB = kvbuffer verify (emit +
    write u-buffer) + flush (rank-m rebuild S_m). spd = REC / KVB. Accept length m = --accept."""
    def us(x):
        return f"{x * 1e3:.1f}" if x else "n/a"

    if args.commit == "scatter" and not _HAVE_SCATTER:
        raise RuntimeError(
            f"commit=scatter needs the official sglang kernel; set KDA_SCATTER_FILE to "
            f"mamba_state_scatter_triton.py (load error: {_SCATTER_ERR})")

    print(f"\n=== verify-CHAIN latency (us, CUDA-graph, kernel-only) — accept m={args.accept} "
          f"commit={args.commit} ===")
    if args.commit == "scatter":
        print("  REC = verify (writes T·d² states) + commit (official sglang "
              "fused_mamba_state_scatter_with_mask: coalesced N·d² gather→pool)")
    else:
        print("  REC = verify (writes T·d² states) + commit (strided gather copy, sensitivity)")
    print("  KVB = kvbuffer verify (emit output + write compact u-buffer) + flush (rank-m rebuild S_m)")
    hdr = (f"{'N':>4} {'T':>3} {'m':>3} | {'vk_v':>6} {'cmt':>5} {'REC_vk':>7} | "
           f"{'kvb_v':>6} {'flush':>6} {'KVB_vk':>7} | {'spd_vk':>7}")
    if args.ws:
        hdr += f" || {'ws_v':>6} {'REC_ws':>7} {'wskvb_v':>7} {'KVB_ws':>7} | {'spd_ws':>7}"
    print(hdr)
    print("-" * len(hdr))
    for N in args.batch_sizes:
        for T in args.Ts:
            q, k, v, a, b, A_log, dt_bias, state0, indices = make_dense_inputs(
                N, T, args.H, args.HV, args.K, args.V, device)
            scale = args.K ** -0.5
            m = _accept_len(T, args.accept, N)
            gc = 1  # segments timed separately (large buffers, no amortization)
            inter_buf = torch.empty(N, T, args.HV, args.V, args.K, dtype=torch.float32, device=device)
            ubufs = (
                torch.empty(N, T, args.HV, args.V, dtype=torch.float32, device=device),
                torch.empty(N, T, args.HV, args.K, dtype=torch.float32, device=device),
                torch.empty(N, T, args.HV, args.K, dtype=torch.float32, device=device),
            )
            tg = {}
            # vk REC: verify (fills inter_buf) + commit
            fn_vk = make_vk_call(q, k, v, a, b, A_log, dt_bias, state0.clone(), indices, scale, DSU, inter_buf)
            warmup(fn_vk, 3)
            tg["vk_v"] = t_graph_ms(fn_vk, 3, args.rep, gc)
            if args.commit == "scatter":
                fn_cmt = make_scatter_commit_call(state0.clone(), inter_buf, m,
                                                  N, T, args.HV, args.V, args.K)
            else:
                fn_cmt = make_gather_commit_call(state0.clone(), inter_buf, m)
            warmup(fn_cmt, 3)
            tg["cmt"] = t_graph_ms(fn_cmt, 3, args.rep, gc)
            # vk KVB: verify (fills ubufs) + flush
            fn_vkkvb = make_vkkvb_call(q, k, v, a, b, A_log, dt_bias, state0.clone(), indices, scale, DSU, ubufs)
            warmup(fn_vkkvb, 3)
            tg["kvb_v"] = t_graph_ms(fn_vkkvb, 3, args.rep, gc)
            fn_flush = make_flush_call(state0.clone(), indices, ubufs, m)
            warmup(fn_flush, 3)
            tg["flush"] = t_graph_ms(fn_flush, 3, args.rep, gc)

            rec_vk = tg["vk_v"] + tg["cmt"]
            kvb_vk = tg["kvb_v"] + tg["flush"]
            spd_vk = f"{rec_vk / kvb_vk:.2f}x" if kvb_vk else "n/a"
            row = (f"{N:>4} {T:>3} {m:>3} | {us(tg['vk_v']):>6} {us(tg['cmt']):>5} {us(rec_vk):>7} | "
                   f"{us(tg['kvb_v']):>6} {us(tg['flush']):>6} {us(kvb_vk):>7} | {spd_vk:>7}")

            if args.ws:
                fn_ws = make_ws_call(q, k, v, a, b, A_log, dt_bias, state0.clone(), indices, scale, DSU, inter_buf)
                warmup(fn_ws, 3)
                tg["ws_v"] = t_graph_ms(fn_ws, 3, args.rep, gc)
                rec_ws = tg["ws_v"] + tg["cmt"]
                if _HAVE_WSKVB:
                    fn_wskvb = make_wskvb_call(q, k, v, a, b, A_log, dt_bias, state0.clone(), indices, scale, DSU, ubufs)
                    warmup(fn_wskvb, 3)
                    tg["wskvb_v"] = t_graph_ms(fn_wskvb, 3, args.rep, gc)
                    kvb_ws = tg["wskvb_v"] + tg["flush"]
                    spd_ws = f"{rec_ws / kvb_ws:.2f}x" if kvb_ws else "n/a"
                    wskvb_v_s, kvb_ws_s = us(tg["wskvb_v"]), us(kvb_ws)
                else:
                    spd_ws, wskvb_v_s, kvb_ws_s = "n/a", "n/a", "n/a"
                row += f" || {us(tg['ws_v']):>6} {us(rec_ws):>7} {wskvb_v_s:>7} {kvb_ws_s:>7} | {spd_ws:>7}"
            print(row)


if __name__ == "__main__":
    main()
