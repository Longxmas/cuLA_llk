"""No-IS compare: KVBuffer (chunkwise) vs cuLA recurrent vk/kv (compute-only picture).

Secondary bench. Validates kvbuffer output/state against the recurrent vk op and
times kvbuffer vs vk vs kv with NO intermediate-state writes (dsu=True). This is
the "compute-only" regime where KVBuffer has no memory advantage — useful to see
the raw chunkwise compute overhead. The production-faithful comparison (vs Triton
recurrent WITH intermediate states) is in bench_kvbuffer_is.py.

SELF-CONTAINED: inlines input/timing helpers (the benchmark module gets renamed by
the cleanup agent); only op dep = cula.ops.* . Run on GB200:

  python kvbuffer_dev/bench_kvbuffer.py --check
  python kvbuffer_dev/bench_kvbuffer.py --batch-sizes 1 2 4 8 16 --Ts 2 3 4 6 --HV 32
"""

import argparse
import os
import pathlib
import sys

import torch

os.environ.setdefault("FLA_USE_FAST_OPS", os.getenv("CULA_USE_FAST_MATH", "1"))
_here = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_here.parent))  # cuLA repo root

from cula.ops.kda_decode_mtp import kda_decode_mtp_small_batch  # noqa: E402
from cula.ops.kda_decode_mtp_kvbuffer import kda_decode_mtp_kvbuffer  # noqa: E402


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


def warmup(fn, n):
    for _ in range(n):
        fn()
    torch.cuda.synchronize()


def t_graph_ms(fn, warmup_iters, rep):
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


def call_kvbuffer(q, k, v, a, b, A_log, dt_bias, state, indices, scale, dsu):
    def call():
        return kda_decode_mtp_kvbuffer(
            A_log, dt_bias, q, k, v, a, b, state, indices, scale=scale,
            use_qk_l2norm_in_kernel=True, disable_state_update=dsu,
        )
    return call


def call_vk(q, k, v, a, b, A_log, dt_bias, state, indices, scale, dsu):
    def call():
        return kda_decode_mtp_small_batch(
            A_log, dt_bias, q, k, v, a, b, state, indices, scale=scale,
            use_qk_l2norm_in_kernel=True, disable_state_update=dsu, variant="vk", bv=-1,
        )
    return call


def call_kv(q, k, v, a, b, A_log, dt_bias, state, indices, scale, dsu):
    def call():
        return kda_decode_mtp_small_batch(
            A_log, dt_bias, q, k, v, a, b, state, indices, scale=scale,
            use_qk_l2norm_in_kernel=True, disable_state_update=dsu, variant="kv", k_split=-1,
        )
    return call


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--Ts", type=int, nargs="+", default=[2, 3, 4, 6])
    ap.add_argument("--H", type=int, default=16)
    ap.add_argument("--HV", type=int, default=32)
    ap.add_argument("--K", type=int, default=128)
    ap.add_argument("--V", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--rep", type=int, default=300)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--atol", type=float, default=5e-2)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "需要 CUDA"
    device = "cuda"

    print(f"\n=== 正确性 (max|Δ|, 阈值 {args.atol:.0e}) HV={args.HV} K={args.K} V={args.V} ===")
    print(f"{'N':>4} {'T':>3} | {'o:kvb-vk':>11} {'state:kvb-vk':>13} | flag")
    print("-" * 45)
    ok_all = True
    for N in args.batch_sizes:
        for T in args.Ts:
            q, k, v, a, b, A_log, dt_bias, state0, indices = make_dense_inputs(
                N, T, args.H, args.HV, args.K, args.V, device)
            scale = args.K ** -0.5
            s_kvb = state0.clone()
            o_kvb = call_kvbuffer(q, k, v, a, b, A_log, dt_bias, s_kvb, indices, scale, False)().float()
            s_vk = state0.clone()
            o_vk = call_vk(q, k, v, a, b, A_log, dt_bias, s_vk, indices, scale, False)().float()
            d_o = (o_kvb - o_vk).abs().max().item()
            d_s = (s_kvb - s_vk).abs().max().item()
            ok = d_o < args.atol and d_s < args.atol
            ok_all = ok_all and ok
            print(f"{N:>4} {T:>3} | {d_o:>11.2e} {d_s:>13.2e} | {'OK' if ok else 'DIFF!'}")
    print("正确性:", "全部 OK" if ok_all else "有 DIFF!")

    if args.check or not ok_all:
        return

    print("\n=== 性能 t_graph (kernel-only, dsu=True 无态写) us ===")
    hdr = f"{'N':>4} {'T':>3} | {'kvbuffer':>9} {'vk':>9} {'kv':>9} | {'kvb/vk':>8} {'best':>9}"
    print(hdr)
    print("-" * len(hdr))
    for N in args.batch_sizes:
        for T in args.Ts:
            q, k, v, a, b, A_log, dt_bias, state0, indices = make_dense_inputs(
                N, T, args.H, args.HV, args.K, args.V, device)
            scale = args.K ** -0.5
            makers = {
                "kvbuffer": call_kvbuffer(q, k, v, a, b, A_log, dt_bias, state0.clone(), indices, scale, True),
                "vk": call_vk(q, k, v, a, b, A_log, dt_bias, state0.clone(), indices, scale, True),
                "kv": call_kv(q, k, v, a, b, A_log, dt_bias, state0.clone(), indices, scale, True),
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

            def r(num, den):
                return f"{den / num:.2f}x" if (num and den) else "n/a"

            cands = {nm: t for nm, t in tg.items() if t}
            best = min(cands, key=cands.get) if cands else "-"
            print(f"{N:>4} {T:>3} | {us(tg.get('kvbuffer')):>9} {us(tg.get('vk')):>9} {us(tg.get('kv')):>9} | "
                  f"{r(tg.get('kvbuffer'), tg.get('vk')):>8} {best:>9}")


if __name__ == "__main__":
    main()
