# Arm — `--context-shift` hybrid-state correctness (Qwen3.8-27B, GDN hybrid)

Design/spec phase only — not yet executed. Nothing in this doc assumes code
changes; the binary under test is the **baseline** build (clean v0.4.0 + PR
#110's admission-gate port, `d50efc6f0`).

## Background and premise

Qwen3.8-27B is a hybrid architecture: Gated DeltaNet (GDN / recurrent
linear-attention) layers mixed with normal attention layers. Production runs
the 747.4 pin with `context_shift: on` — context-shift works by discarding old
tokens from the cache once `n_ctx` fills, so the model can keep generating
indefinitely. For plain attention layers this is a clean per-token KV slice.
For the recurrent/GDN layers it is **not** — the recurrent state is a single
running vector per sequence; it is not a per-token array, so "delete tokens
[p0, p1)" has a meaning only through a bounded per-token snapshot window.

### Confirmed code facts (verified on baseline @ `d50efc6f0`)

Recurrent rollback path — `src/llama-memory-recurrent.cpp:161-215`:

```cpp
// models like Mamba or RWKV can't have a state partially erased at the end
// of the sequence because their state isn't preserved for previous tokens
...
// partial rollback via per-token snapshot index (bounded by n_rs_seq)
if (0 < p0 && p0 <= cell.pos && p1 > cell.pos) {
    const llama_pos rollback = cell.pos - (p0 - 1);
    // pending rollback is single-use
    const bool pending = rs_idx[seq_id] != 0;
    if (!pending && rollback >= 1 && rollback <= (llama_pos) n_rs_seq) {
        set_rs_idx(seq_id, (uint32_t) rollback);
        cell.pos = p0 - 1;
        return true;
    }
    return false;   // rollback exceeds the snapshot window -> silent failure
}
```

Constraints on whether a recurrent-layer rollback can succeed:

1. `rollback <= n_rs_seq` — the snapshot window. With the production spec
   config (`--spec-type draft-mtp`), `n_rs_seq` is set to
   `params.speculative.need_n_rs_seq()` = `draft.n_max` (`common/common.h:394-400`,
   default `n_max = 3`, `common/common.h:326`) — **a 3-token snapshot window**.
2. The pending rollback (`rs_idx != 0`) is **single-use** — it is consumed by
   the next decode. Because spec-decode draft/apply calls `seq_rm` on the
   recurrent cache between decodes, the window may be pending/burned at the
   exact moment the shift fires, making even a 1-3 token rollback fail.
3. Hybrid dispatch — `src/llama-memory-hybrid.cpp:143-150`:

```cpp
bool llama_memory_hybrid::seq_rm(...) {
    // Try removing from the recurrent cache first since it may fail.
    if (!mem_recr->seq_rm(seq_id, p0, p1)) {
        return false;      // <-- attention seq_rm is NOT reached (short-circuit)
    }
    return mem_attn->seq_rm(seq_id, p0, p1);
}
```

4. **The server swallows the failure** — `tools/server/server-context.cpp:2968-2971`,
   the context-shift call site:

```cpp
SLT_WRN(slot, "slot context shift, n_keep = %d, n_left = %d, n_discard = %d\n",
        n_keep, n_left, n_discard);

slot.mem.seq_rm (slot.id, n_keep, n_keep + n_discard);   // return value IGNORED
slot.mem.seq_add(slot.id, n_keep + n_discard, ..., -n_discard);
```

   The shift then proceeds unconditionally: positions of surviving tokens are
   shifted by `-n_discard` and the token buffer is physically trimmed
   (`server-context.cpp:2976-2987`). So the expected failure mode is:

   - recurrent `seq_rm` fails (n_discard = `n_left/2`, typically hundreds to
     thousands >> n_rs_seq = 3) → **whole hybrid seq_rm returns false**,
   - the server ignores this, so even the attention layer's tokens are not
     removed from the hybrid cache, yet the server still calls `seq_add` and
     trims its own token buffer — attention cache positions and the trimmed
     token stream desync, while the recurrent state is untouched,
   - the server records only that a shift *was attempted* (`SLT_WRN` above).
     There is **no log anywhere** that the underlying `seq_rm` failed
     (re-verified on baseline @ `d50efc6f0`: nothing else in
     `pre_decode()` checks the return value, and
     `llama_memory_recurrent::seq_rm` only logs for `invalid seq_id`,
     `llama-memory-recurrent.cpp:172-174`).

   Upstream #24786's `n_discard` clamp (`server-context.cpp:2966`) prevents an
   outright crash/negative indexing, so the failure mode is **silent state
   corruption, not a crash**. This is the same bug class this fork already
   found twice in this model family — hydra_vortex findings #469 ("dishonest
   RPC PREFILL checkpoints corrupted hybrid/recurrent state") and #641
   ("hybrid checkpoint rewind") — plain-attention KV semantics assumed by code
   paths that silently mishandle GDN/recurrent state. Upstream has related
   open hybrid/cache-consistency issues too: ggml-org/llama.cpp #22384,
   #24055, #20428.

### Critical empirical fact: production probably never shifted at all

Every 747.x production boot shows at init
(`docs/investigations/740-results-report.md` §747.0/747.1/747.3, and the
PR105.0 arm boot log on the same binary family):

```
W cmn  common_init_: KV cache shifting is not supported for this context, disabling KV cache shifting
```

That line is emitted at `common/common.cpp:1457-1459`:

```cpp
if (params.ctx_shift && !llama_memory_can_shift(llama_get_memory(lctx))) {
    COM_WRN("KV cache shifting is not supported for this context, "
            "disabling KV cache shifting\n");
    params.ctx_shift = false;
}
```

i.e. for the production shape, production's `--context-shift on` has been
**silently converted to OFF at every boot**. This means:

- we have near-zero production evidence that context-shift even fires for
  Qwen3.8, let alone that it is correct;
- this arm MUST run a Gate-0 boot-log check and, because of the mechanism
  pinned below, needs a **test-only patch** to exercise the gate at all
  (see "Gate 0 revision" and the patch file).

### Root cause of the boot-time disable (pinned to one line, independently verified)

Traced end to end on baseline @ `d50efc6f0`:

1. `src/llama-model.cpp`, `llama_model_rope_type()` — `LLM_ARCH_QWEN35`
   and `LLM_ARCH_QWEN35MOE` are grouped with `QWEN3VL`/`QWEN3VLMOE` and
   return `LLAMA_ROPE_TYPE_IMROPE` **unconditionally** (not gated on
   mmproj/vision being loaded). This is baked into the architecture table.
2. `llama_hparams::n_pos_per_embd()` — returns `4` for
   `MROPE`/`IMROPE` rope types (all 4 M-RoPE axes), so qwen35 has
   `n_pos_per_embd() = 4`.
3. `src/llama-kv-cache.cpp`, `llama_kv_cache::get_can_shift()` —
   `if (hparams.n_pos_per_embd() > 1) { return false; }` fires
   unconditionally for every qwen35/qwen35moe boot, regardless of ctx size,
   kv_unified, SWA shape, or any launch flag.
4. Confirmed against the actual production GGUF: the GGUF header at
   `/mnt/SSD/Qwen3.8-27B-UD-Q5_K_M.gguf` declares
   `general.architecture = qwen35` (read directly from the GGUF header).

**Conclusion: no config lever exists to enable the shift for this model —
it is an architectural hard gate, not a runtime toggle.** This resolves the
arm's H0 in favor of "config no-op is real, and now has a precise cause".
Consequences for the arm:

- Gate 0 as originally scoped (find a bootable shape with shift enabled)
  can never succeed as written, since no shape change affects the RoPE
  type. Gate 0 is therefore revised below to use a **test-only local
  patch** that bypasses this specific check.
- One more gate is discovered by this trace: even with `get_can_shift()`
  bypassed, `llama_kv_cache::seq_add()` and `llama_kv_cache::seq_div()`
  carry `GGML_ASSERT(hparams.n_pos_per_embd() == 1)`
  (`llama-kv-cache.cpp` seq_add/seq_div) — the very first shift would
  hard-abort. The test patch must (and does) cover all three sites.

## Hypothesis

After a shift fires on Qwen3.8:

- H1 (spotless): both attention and recurrent layers shift correctly; the
  model genuinely forgets only tokens older than the eviction boundary and
  recalls everything else — shift is safe for Qwen3.8.
- H2 (silent corruption): the hybrid `seq_rm` fails silently; either the
  post-shift output is garbled/incoherent (desynced attention × recurrent),
  or the model's recall pattern contradicts the eviction boundary (e.g.
  post-shift-marker recall is flaky even though those tokens were never
  evicted — the shifted attention positions no longer align with the
  untouched recurrent state).
- H0 (null): the shift never actually fires because the init disable above
  applies to this arm's shape too — then the finding is "ctx_shift is
  config-no-op for Qwen3.8 in every shape we can boot" (diagnosability/config
  bug, still worth a finding; the correctness question stays open).
  **Status: effectively confirmed and root-caused** — see "Root cause"
  above (`n_pos_per_embd() = 4` from unconditional IMROPE rope type). With
  the test-only patch (Gate 0 revised below) the arm still answers H1/H2
  on the patch-enabled shape; the null-hypothesis resolution stands
  separately for vanilla production configs.

## Rig and launch spec

Base launch = **747.4 production pin** (`infra/llama-baseline/params/
747.4-baseline-nokvu-p2-pool262k-cap164k-ctxsame.yml` in the hydra_vortex
worktree) with two deltas: small `ctx` and UM off (see rationale). Topology
stays the production RPC split (server on CUDA0 = 5060 Ti, `rpc-server` on
CUDA1 = 3060, `tensor_split 27,38` — RPC0 gets 27, CUDA0 gets 38; the
PR105.0 doc's `-ts` order flip applies only to the single-machine in-process
topology, not the RPC topology we use here).

| Item | Production 747.4 | This arm | Why |
|---|---|---|---|
| `-np` / parallel | 2 | 2 | production parity (the `parallel: 2` confound is the point) |
| `ctx` (total) | 262144 | **16384** | 2 × 8192. Small so the shift actually fires quickly and cheaply: sessions reach per-slot `n_ctx` in ~10 chat turns of ~800 tokens, not ~55 min of prefill. Upstream's PoCs for context-shift bugs use the same scale. 8192/slot keeps the probe wall-clock < 1 h/cell and boots on this rig with **no UM oversubscription** (KV is ~30 MB here, vs 262 K cells in prod — that is the only reason UM can be dropped) |
| `GGML_CUDA_ENABLE_UNIFIED_MEMORY` | 1 | **unset** | UM exists in prod only to boot the oversized 262 K-pool shape; at 16 K total cells it is a pure confound (and per PR105.0 §rig-validation, UM + imbalanced in-process split has its own paging landmine). No oversubscription at 8K/slot to mask |
| RPC `tensor_split` | 27,38 | 27,38 | unchanged |
| KV types | q8_0 / q5_1 (+ draft q8_0/q5_1) | same | parity |
| MTP | draft-mtp on | **on in cells A*, off in cells C*** | MTP-on is production-realistic AND it is what plants `n_rs_seq = 3` (vs 0 without spec) — cell C isolates the MTP/n_rs_seq interaction |
| YaRN | yarn scale 5, orig 32768 | same | parity |
| cache-prompt / checkpoints / idle slots / cache-ram | 24576 MiB | same flags, `cache_ram_mib: 1024` | checkpoint stores scale with context size; a 24 GB host-RAM reservoir is pointless at 8K/slot. Keep the *mechanisms* on (they interact with shift via checkpoint restore/rollback) |
| admission gate | `--parallel-ctx-threshold 100000` | same flag value, but will not bind at 8192/slot | harmless to keep for boot-spec parity | 
| port | 18081 | **8080** (prod pod on 18081 stays up, untouched) | arm etiquette per PR105.0/PR103.0 |

Launch (cell A):

```bash
# test-only patch first, then rebuild (build flags below the command block)
git apply docs/arms/arm-context-shift-hybrid-testpatch.patch

export LLAMA_TEST_FORCE_SHIFT_QWEN35=1   # TEST HARNESS ONLY — never set in production
# topology per 747.4: rpc-server on CUDA1 (3060), server on CUDA0 (5060 Ti)
ggml-rpc-server --host 127.0.0.1 --port 50052 -d 1 2>&1 &

./build/bin/llama-server \
  -m /mnt/SSD/Qwen3.8-27B-UD-Q5_K_M.gguf \
  --rpc 127.0.0.1:50052 -ts 27,38 -ngl 99 \
  --rope-scaling yarn --rope-scale 5 --yarn-orig-ctx 32768 \
  -fa on -ctk q8_0 -ctv q5_1 -ctkd q8_0 -ctvd q5_1 \
  --no-kv-unified --cache-prompt --cache-reuse 64 --cache-idle-slots \
  --cache-ram 1024 --ubatch-size 512 --cont-batching \
  -np 2 -c 16384 \
  --parallel-ctx-threshold 100000 --spec-type draft-mtp \
  --context-shift --prio-batch 1 \
  --jinja --host 0.0.0.0 --port 8080 --metrics --slots --log-verbosity 4
```

(Cells B/C vary only the lines marked. Build flags as usual for this fork:
`-DGGML_CUDA=ON -DGGML_RPC=ON -DGGML_CUDA_FA_ALL_QUANTS=ON
-DGGML_CUDA_FORCE_CUBLAS=OFF -DCMAKE_CUDA_ARCHITECTURES="86;120"
-DCUDAToolkit_ROOT=/opt/software/cuda/13.2.2 -DCMAKE_BUILD_TYPE=Release`.)

Pre-flight hardware checks: `nvidia-smi` free memory, `/health` 200, both
devices in boot log, then **Gate 0** below.

## Test-only patch: `docs/arms/arm-context-shift-hybrid-testpatch.patch`

To run Gate 0 / A1 / A2 / C1 / C2 at all, the runner applies this patch to
the baseline tree (`git apply docs/arms/arm-context-shift-hybrid-testpatch.patch`)
and rebuilds. Spec:

- **Env gate**: `LLAMA_TEST_FORCE_SHIFT_QWEN35` (set only in the arm's test
  launcher; unset = zero behavior change vs vanilla baseline — the patch is
  also fully dormant at runtime).
- **Scope, tightly architectural**: only `rope_type == IMROPE &&
  (arch == LLM_ARCH_QWEN35 || arch == LLM_ARCH_QWEN35MOE)`. All three
  three hard-gate sites are covered:
  1. `llama_kv_cache::get_can_shift()` — inside the
     `n_pos_per_embd() > 1` branch, only for the two qwen35 arches,
     env-gated; every other IMROPE arch (qwen3vl, etc.) stays prohibited.
  2. `llama_kv_cache::seq_add()` — the `GGML_ASSERT(n_pos_per_embd()==1)`
     is downgraded to a warning under the env gate (the binary would
     otherwise abort at the first shift despite the gate above),
     proceeding with a scalar cell-pos shift.
  3. `llama_kv_cache::seq_div()` — same downgrade, so the cache-reuse
     divide path can't abort either.
- **Why the scalar-pos shift is defensible for this probe (and only
  there)**: with text-only chats, all four M-RoPE axes hold the same
  position value; a single scalar shift equals a per-axis shift.
  Mixed-media content would silently corrupt non-temporal axes — hence
  the "test harness only" warning, and this is exactly the second,
  separate risk asked to be flagged: **IMROPE K-shift correctness for the
  4-axis position case is itself unverified and is a second, separate risk
  beyond what this arm measures.**
- **Easy to revert**: 37 inserted lines in one file
  (`src/llama-kv-cache.cpp`), no headers touched, no behavior change with
  the env unset; `git checkout -- src/llama-kv-cache.cpp` reverts it.
  The PR carries the patch un-applied — the arm tree applies it, builds,
  runs, and drops it in close-out.

## Gate 0 (revised — before any probe)

With the test patch applied and `LLAMA_TEST_FORCE_SHIFT_QWEN35=1` set,
verify from the boot/probe logs:

- [ ] With the patch applied but `LLAMA_TEST_FORCE_SHIFT_QWEN35` unset:
      boot must still print the disabling warning (proves the gate is
      dormant and the vanilla behavior is unchanged — a negative control
      on the patch itself).
- [ ] With `LLAMA_TEST_FORCE_SHIFT_QWEN35=1`: the disabling warning line
      (`common/common.cpp:1457`) is ABSENT (i.e. `common_init_` sees
      shift-enabled memory) and `llama_kv_cache::get_can_shift()` printed
      its TEST HARNESS ONLY warning.
- [ ] Optional shape-lever tracking from the original Gate 0 drafting:
      try booting one config WITHOUT the patch but with (a) `--no-spec`,
      (b) kv_unified on, (c) `-ctk f16 -ctv f16`, (d) `-np 1 -c 8192` and
      record that the disabling warning still fires in every case —
      confirming the root-cause claim that no config lever exists.
- [ ] `n_rs_seq` reported in the `llama_context` init INFO line is the
      expected 3 for spec-on cells and 0 for spec-off cells (`ctx.cpp` logs
      `n_rs_seq = %u`). This confirms the snapshot-window premise.
- [ ] Boot INFO shows `n_parallel = 2`, `n_ctx_slot = 8192`.
- [ ] No Xid errors / OOM, no `GGML_ABORT` from `seq_add`/`seq_div` at the
      first shift event (if one aborts, the patch's third site is
      incomplete — stop and fix the patch, do not proceed).

If any Gate-0 checklist item fails, STOP: record it and fix the patch (or
file the bug) before any probe run. Do not observe a probe on a server
where the shift is silently disabled or abort-prone — all probe verdicts
would be uninterpretable.

## Test cells

All cells share the probe harness (below). Per cell, run ≥ 2 full probe
sessions of ~20 K tokens each (concurrency = 2 means the two sessions run
concurrently; concurrency = 1 means the same two-session harness runs
sequentially as a control).

| Cell | MTP | Concurrency | What it isolates |
|---|---|---|---|
| **A1** | on | 1 | production-realistic shift under no concurrency — the "works alone" bar |
| **A2** | on | 2 | the production confound: 2 sessions each shift while the other is resident; cross-session rs-pending/rollback interference |
| **C1** | off (`--no-spec`) | 1 | n_rs_seq = 0 → recurrent partial rollback can NEVER succeed on this config; a Clean verdict here means the attention-only desync is tolerable, while a fail isolates that plain attention+recurrent coexistence (not MTP interaction) is already broken |
| **C2** | off | 2 | same, concurrent |

(Right-branch cells B* = A* with prior `docs/arms/pr103-gdn-cache-cpy-fusion`
fusion enabled, if that arm lands first — the fusion redirects snapshot
writes through the same `cpy` path that interacts with rollback state; test
it only after the A* cells have a verdict, as fusion should not change
correctness, only perf. Do not run B* before A*.)

## Probe design (needle-in-haystack two-marker recall)

The probe has to distinguish (a) clean eviction — model should NOT remember
everything evicted and SHOULD remember everything past the boundary — from
(b) silent state corruption — a desynced GDN layer shows up as random-quality
noise: garbling, inconsistent recall of never-evicted tokens, or
cross-session state contamination.

1. **Session skeleton** (`session_lengths ~ 2.4 × n_ctx_slot`, i.e. ~20 K
   tokens per session — enough to trigger ≥ 2 shifts per session):

   - Turn 1 (marker-plant **pre-shift marker M1**): "Here is a story to keep
     in mind. Once, there was a very small pig named Wilbur whose favorite
     color was **chartreuse**, and the pig lived with a man named
     **Borzoi-san**. Please remember this story." followed by 30 turns of
     neutral filler ("Continue writing a story about the sea", "How many
     legs does a cat have, and why do they have that number on this planet?
     Keep a natural tone.") — enough cumulative filler to bring
     `n_tokens + 1 >= n_ctx_slot` on one of these turns. M1 sits early →
     `chartreuse` pig / `Borzoi-san` are older than the eviction boundary.
   - Turn T-shift (marker-plant **post-shift marker M2**): placement rule
     — plant M2 at a turn whose tokens will be past the eviction boundary of
     the just-fired shift and any later shift (i.e. inject the M2 story
     immediately after the turn where the first shift fired, then keep the
     session running past a second shift). If the first shift fires mid-fill
     between turns, M2 lands safely inside the surviving window; verify
     against the logged `n_keep`/`n_discard` from the shift line that M2's
     turn is strictly after `n_keep + n_discard` of every shift in the
     session.

   Example M2 plant (runner may randomize, but the strings must be disjoint
   from session 2's): "Here is another story to keep in mind: a squirrel
   named **Zurnif-8** lived with a beekeeper named **Pavdeel** and spoke
   only in a rare accent, **Felarn**. Please remember this story."
   - Post-shift recall probes (run across the remainder of the session,
     phrased differently each time so prefix-cache hits do not mask state
     effects):
     - P1: "What was the pig's favorite color? What was its name?"  (and the
       pig's owner) — target M1 = chartreuse/Borzoi-san. **Expected
       honest-forget**: no identity-coherent recall allowed; any confident
       M1 recall is a red flag (eviction did not actually happen for this
       state) — but only if M2 recall is reliable.
     - P2: "What was the squirrel's name? Who did it live with? What rare
       accent did it speak in?" — target M2 = Zurnif-8 / Pavdeel / Felarn.
       **Expected: correct, consistently answerable 5× in a row**.
     - P3 (desync detector, new question, no marker reliance): 3 open
       questions ("Describe the pig's farm in detail"; 2 neutral long-form
       prompts) — scored for **coherence**: no garbling, no token soup, no
       contradictions mid-answer, no cross-session leakage.
     - P4 (repeat-determinism): identical P2 at temp 0 twice, byte-diff
       equality required (PR #110 self-determinism standard; repeated runs
       through the same post-shift recurrent state must give byte-identical
       output. Non-identical = recurrent-state nondeterminism between the
       two repeated runs = corruption signature).
2. **Concurrency**: cells A2/C2 run **two concurrent sessions**: the runner
   drives the sessions as two concurrent threads (same technique PR105.0
   used — `threading.Thread` + per-turn wall-clock overlap asserted ≥ 80% of
   total session wall), and each session plants **completely disjoint
   marker strings** (session 1: pig Wilbur / chartreuse / Borzoi-san;
   session 2: use a different animal, name, color, and keeper — zero shared
   tokens between the two casts) so cross-session leakage is measurable:
   session 2 must never answer with session 1's marker string, and vice
   versa. This also exercises the shared recurrent-state bookkeeping under
   `n_parallel = 2`: `rs_idx[seq_id]` is per-seq-id, but both sessions'
   shifts and spec-decode checkpoint rollbacks interleave inside the same
   `llama_memory_recurrent` instance, so any cross-seq accounting bug in the
   snapshot machinery will show up as cross-contamination.
3. **Both sessions trigger their own shift** — the harness grows each
   session to ~20 K tokens so each fires ≥ 2 shifts. Confirm from the
   `SLT_WRN` `slot context shift` lines per slot in the server log
   (`grep "slot context shift" server.log | sort | uniq -c`): ≥ 4 shift
   lines per slot (2 sessions × 2 shifts). Pass convention: 2 shift events
   per session over ~20 K tokens at 8192/slot.

### Scoring

For each session, run 10 probe turns across the post-shift remainder (5
targeting M1, 5 targeting M2, phrased differently each time so prefix-cache
hits do not mask state effects), plus the P3/P4 observations:

| Verdict | M1 (pre-shift, should be forgotten) | M2 (post-shift, should be recalled) | Coherence (P3) | Determinism (P4) |
|---|---|---|---|---|
| **Clean** | Forgets both facts (≤ 1/5 confident recall) OR admits not knowing | Recalls ALL facts ≥ 5/5 across cycles | 0 incoherent passages | byte-identical |
| **Silent-corruption (fail)** | "Flaky recall" (different confident answers to the same M1 fact across probes) or "hologram recall" (confident facts contradicting the planted fact) | Recalls < 5/5, or differing answers to the same fact across cycles (nondeterministic), or garbled | ≥ 1 garbled / contradictory passage | non-identical output |
| **Cross-contamination (fail)** | (either mode) | Session 2 recites session 1's cast (or vice versa) | — | — |

"Flaky recall" = same question asked across the 5 probes gives ≥ 3
different confident answers (nonzero colors/keepers). "Hologram recall" =
confident facts that contradict the planted fact (state partially retained,
reconstructed wrong — a classic desynced-hybrid signature).

## Log-level verification plan (separate scoring line)

**Known, documented diagnosability gap (do NOT fix in this arm, this is
design doc reporting):**

- `tools/server/server-context.cpp:2970` ignores `slot.mem.seq_rm`'s return.
  There is no log when the recurrent layer's rollback fails — the only
  server-side signal is the attempt itself (`SLT_WRN "slot context shift"`).
- `llama_memory_recurrent::seq_rm` only logs for the invalid-seq-id
  rejection (`llama-memory-recurrent.cpp:173`); a bounded-window failure is
  a bare `return false`.
- Therefore, today, **there is no way to see from logs alone whether the
  recurrent layer accepted or refused the shift**, or how many refusals a
  production session incurs. The arm measures this `gap` behaviorally (via
  the probe bars) and the review issue candidate (follow the project's
  `review-finding` protocol) should propose:
  1. server logs the `seq_rm` return per shift, and
  2. an optional `LLAMACPP_ ...=1`-style verbose line exposing the hybrid
     layer breakdown (which physical sub-cache refused).

Additionally, at run time, count for the record:

- server log: total `slot context shift` lines per slot vs the expected
  shifts; any `GGML_ABORT "The current KV cache / model configuration does
  not support K-shift"`-adjacent aborts (should not appear post-#24786);
- `llama_memory_recurrent::seq_rm` failures are invisible today — treat
  "shift attempt count >> expected" (e.g. per-slot > 5 attempts for a 2-shift
  session) as a soft red flag for retry/checkpoint-rollback churn in
  combination with checkpoint lines (`restored context checkpoint`,
  `created context checkpoint`) from the interaction of spec-decode's
  RS-type checkpoint path (`server-context.cpp:3091-3092`,
  `draft.size() > llama_n_rs_seq`) burning the single-use pending state
  (`rs_idx`) and the shift hitting the `pending` rejection.

## Bars — pass/fail in concrete terms

**PASS ("safe to keep enabled for Qwen3.8")** — requires ALL of:

1. Gate 0: shift confirmed enabled at boot in the eval cell, and observed
   `slot context shift` events ≥ 2 per session, with no aborts.
2. Coherence across the full ~20 K session chain in each executed cell:
   P3 clean and P4 byte-determinism confirmed.
3. **Marker separation**: M2 (post-shift) correct on all 5 probes; M1
   forgotten on all but ≤ 1 probe, with no confident-but-false M1 answer
   anywhere in the session.
4. Concurrency equivalence: **cell A2 vs A1 and C2 vs C1: same verdict
   verdict class**, and **zero cross-session marker leakage** in A2/C2 on
   any probe (10 probes × 2 sessions = 20 opportunities).
5. Log echo: everything scored above retained in the arm report, and the
   `slot context shift` attempt counts match the expected shifts (not 10×).

**FAIL ("disable / flag as broken")** — ANY of:

1. Gate 0 fails after the patch (negative control broken, disabling
   warning still fires under the env, or `seq_add`/`seq_div` aborts at the
   first shift) — verdict is
   "ctx_shift is NOT actionable for Qwen3.8 in any bootable/repaired form;
   needs an upstream fix (IMROPE-shift correctness or an architecture-level
   shift path) before it can be a safe production toggle" + diagnosability
   finding. NOTE: the original "Gate 0 blocked" version (no bootable
   shape) is now superseded by the pinned root cause — no shape ever boots
   with shift enabled without the test patch.
2. **Recall does not separate by boundary** (the classic silent-corruption
   signature): M2 (never evicted) and M1 (evicted) recall are statistically
   indistinguishable — M2 flaky while M1 gets confident "hologram" hits —
   and/or P4 non-determinism confirmed. ⇒ the shift leaves inconsistent
   hybrid state; recommend removing `context_shift` from the next 747.5 pin
   + open a `review-finding` issue.
3. **Coherence crack**: ≥ 1 of the 10 P3 passages shows garbling
   (token soup / broken words), a mid-answer self-contradiction, or the
   session's off-marker knowledge contradicting itself after the shift.
4. Cross-session marker leakage in A2/C2: any inverted cast mention.
5. Shift count mismatch: > 10 shift attempts in a session where 2 are
   expected, combined with checkpoint-restore churn — indicates the
   rollback bookkeeping is thrashing, not functioning.

**Asymmetric concurrency result** (A1 FAIL / A2 PASS, or C1 FAIL / C2 PASS,
or vice versa) is itself a critical finding — it means the concurrent
interference, not the isolated shift logic, changes correctness under
production's concurrent workload — and is scored as a hard fail in either
direction.

## Correctness gates (hard, per fork convention)

- Baseline run (a session without a shift, n_tokens never reaching
  n_ctx_slot, ~1/3 of the session length) must score **Clean** — otherwise
  the probe harness itself, not context-shift, is the confounder and the
  session design must be revised before any shift-context verdict.
- Greedy repeat determinism (P4) within-cell: byte-identical, both pre-shift
  and post-shift repeats.

## Sequencing / execution notes (for the runner)

1. Build & boot per cell; capture full boot log per cell (workspace
   convention: `restore-log/` artifacts listing).
2. Run Gate 0; STOP if blocked — log it as the verdict.
3. Run probe harness for cells in order A1 → C1 → A2 → C2 (+ control
   baseline-no-shift run first). Each cell ends with
   `pkill llama-server && podman pod start pod_llama-baseline && /health`
   restore per arm etiquette (PR105.0/PR103.0).
4. Record results and verdict in a `## Results` section in this doc;
   `review-finding` issue for the log-gap (seq_rm silent) per project
   close-out, then hand a `747.5` config recommendation to the
   hydra_vortex `review-finding` backlog.
5. Do NOT merge this PR without a rig execute pass + explicit user
   confirmation; never merge live-infra verify to `main` directly.
