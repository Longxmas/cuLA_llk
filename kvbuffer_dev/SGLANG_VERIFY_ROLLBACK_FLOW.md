# SGLang KDA spec-decode: verify → accept → rollback flow (research notes)

Researched on remote GB200, sglang source at `/ossfs/workspace/xiangwan/SGLang/python/sglang`
(inside the xiangwan whitelist; the other `/ossfs/workspace/sglang` is out of scope).
Purpose: model the **real** spec-decode chain (verify → pick accept length m →
rollback/commit) so the KVBuffer-vs-recurrent benchmark reflects production, not
just the bare verify kernel.

## The three stages

### 1. verify — `KDAAttnBackend.forward_extend`, `is_target_verify` branch
(file `kvbuffer_dev/kda_backend.py`, the local copy Long shared)

- `causal_conv1d_update(...)` also caches the conv window into
  `intermediate_conv_window` (conv has its own rollback, separate & small).
- calls `kernel_dispatcher.target_verify(...)` → the **Triton** operator
  (`benchmarks/fused_sigmoid_gating_recurrent.py`) with
  `intermediate_states_buffer = intermediate_state_cache`,
  `intermediate_state_indices`, `cache_steps = draft_token_num`,
  `retrieve_parent_token` (tree attention).
- So verify = compute T per-token outputs **+ cache all T per-token states**
  (`intermediate_states_buffer [N, T, HV, V, K]` fp32). The operator itself does
  **NOT** decide acceptance (confirmed: no accept/compare/argmax logic in it).

### 2. acceptance — `srt/speculative/eagle_worker*.py`, `eagle_info_v2.py`
- `verify_tree_greedy_func(...)` → `accept_length`, `accept_index` per request.
- TREE-based (eagle): draft is a tree, "accepted" = a path through it.
- **NOT key for our study** (Long's call): we model **chain** (linear) verify and
  just pick an accept length `m` (random or fixed) for evaluation.

### 3. rollback / commit — `srt/layers/attention/hybrid_linear_attn_backend.py:959`
`update_mamba_state_after_mtp_verify(self, accepted_indices, model)` — the core:

```python
# accepted_indices[req] = index of last accepted token (= last_steps), shape [N]
valid_state_indices = state_indices_tensor[valid_mask]      # pool slots
last_steps          = accepted_indices[valid_mask]          # per-req m-1
# pick the accepted intermediate state and scatter into the persistent pool:
ssm_states[:, valid_state_indices, :]  = intermediate_state_cache[:, valid_state_indices, last_steps]
conv_states[:, valid_state_indices, :] = intermediate_conv_window_cache[:, valid_state_indices, last_steps]
```

**This is a pure gather + copy — no compute.** The recurrent path pre-stored all T
candidate states in verify; commit just indexes out the accepted one
(`intermediate_state_cache[:, idx, m-1]`) into the live `ssm_states` pool. Cost is
~one `d²` copy per request. `intermediate_ssm` cache is allocated in
`srt/mem_cache/memory_pool.py` (`MambaPool.SpeculativeState`, ~line 198).

## Cost model: recurrent vs KVBuffer (chain, accept length m)

| stage | recurrent (Triton) | KVBuffer |
|---|---|---|
| **verify** (all T) | T outputs **+ write T·d² intermediate states** | T outputs, **no state write** (+ keep KV / compact u) |
| **commit** (given m) | **gather state[m-1] → ssm pool** (cheap copy, m-independent) | **flush: chunkwise rebuild S_m from first m KVs → pool** (compute, reads 2Td) |

- `recurrent_total(m) = verify_tri+IS + gather_commit`   (≈ constant in m: verify
  writes all T states regardless, commit is a cheap copy)
- `kvbuffer_total(m)  = verify_kvb + flush(m)`            (grows with m: flush does
  chunkwise work over the m accepted tokens)

Implications:
- KVBuffer looks **best at small m** (few accepted → cheap flush; recurrent still
  paid the full T-state write in verify).
- At **m = T (full accept)** KVBuffer should fuse verify+flush into ONE kernel
  (the `full` column, already shown to win at small batch) instead of
  verify + a separate flush that recomputes.
- The current separate flush **recomputes Phase A+B**; the **buffer-u** optimization
  (verify writes the compact `u_i/kinv_i/b`, 2Td; flush = cheap Phase-D rank-m,
  no recompute) makes the split cheap across all m — the next kernel work item.

## Benchmark model (what `bench_kvbuffer_is.py` now does)

For each (N, T) and accept length m:
- **recurrent**: `verify` = Triton + intermediate_states_buffer (writes T states);
  `commit` = torch gather `ssm[idx] = inter[:, idx, m-1]`. total = verify + commit.
- **kvbuffer**: `verify` = kda_decode_mtp_kvbuffer(emit_output=True, dsu=True);
  `flush(m)` = kda_decode_mtp_kvbuffer over inputs sliced to first m tokens
  (emit_output=False, dsu=False). total = verify + flush(m). Also report `full`
  (single fused kernel) as the m=T best case.
- m chosen via `--accept` (full / half / one / int). Real serving = per-request
  ragged m; we use a single representative m per run (chain).
