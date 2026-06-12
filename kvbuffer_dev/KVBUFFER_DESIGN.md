# KDA MTP — KVBuffer / Chunkwise Parallel-Verification Operator (Design)

Follow-up to issue 17. The initial recurrent operators (`kda_decode_mtp.py`: vk
production + kv contrast) are done and being prepped into a PR by a separate
cleanup agent. **This document designs a NEW operator** that implements the
KVBuffer paper's chunkwise / parallel-verification form, to be benchmarked
against the existing recurrent operators.

Paper: `speculative_decoding/KVBuffer.pdf` — *"KVBuffer: IO-aware Serving for
Linear Attention"* (Zou & Zhong, Yale). Implemented in SGLang for Qwen3-Next
(GDN). The directly relevant section is **§3.3 Parallel Verification for
Speculative Decoding**.

---

## 1. What KVBuffer changes (and what it does NOT, for us)

**Existing serving + our recurrent op:** verify `T` draft tokens *recurrently* —
run the gated-delta-rule recurrence token-by-token, and (for rollback) snapshot
`T` full linear-attention states, each `d×d`. That snapshot is the paper's
"384 MB/request" cost (`intermediate_states_buffer [N,T,HV,V,K]` in our kernel).

**KVBuffer:** buffer the `T` draft KVs, compute their outputs in **chunkwise
form** (Eq. 7) `O = Q·S_in + ((QKᵀ)⊙M)·V`, and after acceptance update the state
once with **only the accepted tokens** (Eq. 8) `S_t = S_{t-j} + K_accᵀ·V_acc`.

Reported wins: (a) store KV (`2d`) instead of state (`d×d`) → ~`64×` less
rollback memory at `d=128` → "≈5× more concurrent requests"; (b) verify-latency
speedup Eq. 9 ≈ `(m+1)/3` when `d ≫ m`.

### Honest nuance for OUR setting — keep the comparison fair

The paper's recurrent **baseline reads/writes the `d×d` state from HBM every
decode step** (Table 1: recurrent read `= 4d² + 6d` per token). **Our recurrent
operator does not** — the state is register-resident across all `T` tokens in a
single fused launch, read from HBM once. So in our regime:

- **HBM-traffic win → mostly absent.** Both forms read `S_0` once.
- **Compute (d²) → roughly a wash.** Recurrent ≈ `4T·d²` (decay + s + rank-1 + o,
  per token). Chunkwise ≈ `3T·d²` (`S_0@kdec`, `S_0@qdec`, final outer — each
  `T·d²`, plus one-time `d²`) `+ ~4T²·d` (the `A`,`P`,`Pu`,`Au` `T×T` work).
  For `T≤8, d=128`, `T²·d ≪ T·d²`, so chunkwise is marginally *less* d² work but
  not dramatically.
- **Where chunkwise can actually win latency for us:**
  1. **Rollback memory** — store `2·T·d` KVs instead of `T·d²` states. Real, and
     it maps onto our `intermediate_states_buffer`. This is the headline serving
     win and is genuine even when latency is a tie.
  2. **Parallelism / ILP** — the recurrent form is a strict serial dependency
     chain of length `T` (token `t`'s rank-1 update must complete before token
     `t+1`'s `s = S@k`). Chunkwise replaces that with `T` **independent**
     `S_0`-matvecs (`S_0@kdec_t`, `S_0@qdec_t`) plus a tiny `T×T` triangular
     solve. The expensive `d²` matvecs no longer sit on a serial chain → more
     eligible warps / better latency hiding. **This directly targets the issue-17
     open problem** (vk@BV=32 ~10% slow at N=4/8; ncu Eligible Warps/Sched
     0.43 vs Triton 0.64 — a serial-chain stall).

**Conclusion:** we expect the clear win to be rollback memory; the latency
question (does breaking the serial chain beat the recurrent kernel at small
batch?) is exactly what the benchmark must answer. We design so the comparison
is apples-to-apples (same grid, same lane=K layout, same load/writeback infra).

---

## 2. Chunkwise gated-delta-rule for a single chunk of size T — DERIVED & VERIFIED

Verified to machine precision (`< 1e-15`) against the recurrent reference in
`kvbuffer_dev/verify_chunkwise_kda.py` (pure python, no deps).

State `S[v,k]` (vk layout, `V×K`), input chunk state `S0 = S_{t-j}`. Per token
`t` the recurrence is (decay-first, matching the production kernel):

```
g_t[k] = exp(-exp(A_log) * softplus(a_t[k] + dt_bias[k]))   # per-channel decay
S[v,k] *= g_t[k]                          # 1 decay
s[v]    = Σ_k S[v,k]·k_t[k]               # 2 reduce-K
v_new[v]= sigmoid(b_t)·(v_t[v] - s[v])    # 3 delta
S[v,k] += v_new[v]·k_t[k]                 # 4 rank-1
o_t[v]  = Σ_k S[v,k]·q_t[k]               # 5 reduce-K (post-update state)
```

Define per-channel **cumulative decay** `b_t[k] = Π_{i≤t} g_i[k]` (prefix product
over the chunk, `b ≤ 1`), and decayed feature maps:

```
kdec_t = k_t · b_t          # key, decayed
kinv_t = k_t / b_t          # key, inverse-decayed
qdec_t = q_t · b_t · scale  # query, decayed (scale folded in)
```

Then the chunk is computed as:

```
A[t,i] = <kdec_t, kinv_i>      strict-lower (i<t)      # T×T, reduce-K
P[t,i] = <qdec_t, kinv_i>      lower incl. diag (i≤t)  # T×T, reduce-K
Skdec_t[v] = Σ_k S0[v,k]·kdec_t[k]                     # reduce-K vs fixed S0
Sqdec_t[v] = Σ_k S0[v,k]·qdec_t[k]                     # reduce-K vs fixed S0

# triangular solve (forward substitution over t), per v-column:
u_t[v] = beta_t · ( v_t[v] - Skdec_t[v] - Σ_{i<t} A[t,i]·u_i[v] )

# outputs:
o_t[v] = Sqdec_t[v] + Σ_{i≤t} P[t,i]·u_i[v]

# final chunk state (full accept):
S_T[v,k] = b_{T-1}[k] · ( S0[v,k] + Σ_i u_i[v]·kinv_i[k] )
```

`u_t` is the WY-representation "pseudo-value" — `u_t = beta_t (v_t - decayed
state · k_t)`. `g_t, beta_t` are computed identically to the recurrent kernel.

**Partial accept (rollback):** if only the first `m ≤ T` tokens are accepted,
recompute `b`, `Σ_i u_i kinv_i`, and `S_T` over `i < m` only — same formulas,
truncated. The buffered KVs are sufficient; no per-token state snapshot needed.
(For the operator + benchmark we do full accept = all `T`, matching the recurrent
op's contract; partial-accept is a serving-layer concern.)

**Numerical note:** `kinv = k/b` can grow as `b→0`. For chunk size `T ≤ 8` and
realistic decays this is well-conditioned (the production chunk kernels use this
within chunks of 64). Everything is fp32 inside the kernel. We still validate
against the fp32 oracle at `atol 3e-2 / rtol 2e-2` (same as recurrent).

---

## 3. Kernel plan (`cula/ops/kda_decode_mtp_kvbuffer.py`)

Mirror the production vk kernel so the comparison is fair and ~80% of infra is
reused.

- **Public entry:** `kda_decode_mtp_kvbuffer(...)` — identical signature & IO to
  `kda_decode_mtp` (q/k/v/a/b `[N,T,H/HV,K/V]`, state pool, indices, scale,
  `use_qk_l2norm_in_kernel`, softplus params, `out [N,T,HV,V]`, in-place state
  update, `disable_state_update`). Drop-in for `bench_kda_mtp_small_batch.py`.
- **Grid / block:** `N*HV*(V/BV)`, one CTA = 1 warp (32 lanes), **lane=K**
  (each lane owns `vec_size=4` contiguous K-channels across `BV` V-cols) — same
  as production vk. Reuse: float4 coalesced `LDG.128` load, butterfly
  shuffle-reduce over K, coalesced float4 state writeback.
- **Register state:** `S0` slice `[BV × 4]` per lane (same footprint as vk).
- **SMEM:** per-token `q,k,a,b` for all `T` (and `v`); `A`,`P` as `[T,T]` fp32;
  cumulative `b_t[k]` (reuse the K-channel lane layout — each lane holds its 4
  channels' `b_t` for all `T`).

**Kernel steps:**
1. Load all `T` tokens' `q,k,v,a,b`; compute `g_t[k]` (per channel), `beta_t`;
   optional q/k l2norm — same as vk.
2. Prefix-product over `T` to get `b_t[k]` (per lane, in-register, serial in `T`
   but cheap); form `kdec,kinv,qdec` per token.
3. `A`,`P` (`T×T`): for each `(t,i)` a per-lane partial dot over its 4 channels +
   butterfly reduce → store to SMEM `[T,T]`. `≤ T²` reductions (`T≤8` → ≤64).
4. `Skdec_t[v]`, `Sqdec_t[v]`: `T·BV` reductions vs the fixed register `S0`
   (same reduce pattern as vk's `s = S@k`). Independent across `t` — no serial
   chain.
5. Triangular solve for `u_t[v]` (forward subst over `t`, per v-col, in-register
   using `A` from SMEM). Only serial part; `T` tiny.
6. `o_t[v] = Sqdec_t[v] + Σ_{i≤t} P[t,i]·u_i[v]`; write outputs (bf16).
7. Final state `S_T[v,k] = b_{T-1}[k]·(S0[v,k] + Σ_i u_i[v]·kinv_i[k])`; update
   register state; coalesced writeback (same as vk epilogue).

**Tunables:** `BV` (split V, reuse `_select_vk_bv`-style heuristic); start
`opt_level=3 + fast_math=True` (B200-tuned, matching ws/inline default).

**Out of scope for v1:** kv-layout variant (lane=V), partial-accept path,
multi-chunk (`T>chunk`). v1 = single chunk, vk layout, full accept.

---

## 4. Validation & benchmark

- **Correctness:** new test mirroring `tests/test_kda_decode.py` MTP path —
  compare kvbuffer output & updated state to (a) fp32 torch oracle (the recurrent
  reference) and (b) the existing recurrent vk op, at `atol 3e-2 / rtol 2e-2`.
- **Benchmark:** extend `benchmarks/bench_kda_mtp_small_batch.py` to add a
  `kvbuffer` variant next to `vk`/`kv`, sweeping `--batch-sizes 1 2 4 8 16 64 128
  --Ts 2 3 4 6 --HV 32` (and `--HV 64`). Real Kimi-Linear KDA: HV=32, K=V=128.
- **Profiling:** ncu **Eligible Warps/Scheduler** kvbuffer vs recurrent vk —
  this is the metric the ILP hypothesis (§1) predicts should improve.

Build/test/profile on GB200 (SM100) via Long or `./remote_exec.sh
aistudio-workflow_58650011-ssctl "<cmd>"` (repo `/ossfs/workspace/xiangwan/cuLA_llk`,
venv `/ossfs/workspace/xiangwan/venv`).

---

## 5. RESULTS (GB200, 2026-06-08) — buffer-u implemented, KVBuffer wins

Correctness: verify output vs Triton ≤1.5e-5 (bf16); buffer-u flush S_m vs Triton's
m-th cached state = 1.8e-7 (machine precision) at m = T / half / 1.

Comparison is the **real SGLang chain** (verify → pick accept length m → rollback),
see `SGLANG_VERIFY_ROLLBACK_FLOW.md`:
- **REC** (recurrent) = Triton verify in target_verify mode (writes T·d² intermediate
  states) + commit = pure gather `ssm[idx]=intermediate_state_cache[:,idx,m-1]`
  (`hybrid_linear_attn_backend.update_mamba_state_after_mtp_verify`).
- **KVB** (kvbuffer) = verify (outputs + write compact u-buffer `u/kinv/b`, ~43× <
  T-states, no state commit) + **buffer-u flush** (pure Phase-D rank-m rebuild of S_m,
  no recompute).

`bench_kvbuffer_is.py --accept full|half|one`. **KVBuffer wins across ALL accept
lengths and configs: 1.0–1.53×** (HV=32, K=V=128). The buffer-u flush is cheap and
m-independent (4–12µs) vs the old recompute-flush (26–61µs at large T, ~5× worse) —
this is what unlocks the partial-accept win (not just full-accept). The recurrent
commit (gather) costs 4–22µs (grows with N) — saved by KVBuffer.

Implementation: `kda_decode_mtp_kvbuffer(...)` (verify, `u_buffer/kinv_buffer/
b_buffer` out-args + `write_ubuf`) + `kda_flush_kvbuffer(state, indices, u/kinv/
b_buffer, accept_len)` (flush). Both in `cula/ops/kda_decode_mtp_kvbuffer.py`.

**Open / next:** at large N×T the verify itself (88µs@N16T8) slightly exceeds Triton
verify (78µs) — chunkwise T² intra-chunk + register spill; KVB wins there only via
the saved commit. Cutting register pressure (SMEM for kdec/kinv/qdec, shrink u
storage, software-pipeline Phase A) would widen the lead. Also: per-request ragged m
in flush (currently one scalar m per launch); deferred-update / KV-only short-context
forms (paper §3.2/§3.4); the memory/throughput "5× requests" angle (not yet measured).
