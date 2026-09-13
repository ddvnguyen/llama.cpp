# llama-route-trace

Dumps the per-position MoE expert selection of every routed layer, in the
`ROUTE_TRACE` text format used by [colibri](https://github.com/JustVugg/colibri)'s
offline routing tools. It is a measurement-only tool: it never changes routing,
weights or output.

## Build

```sh
cmake -B build
cmake --build build --target llama-route-trace
```

## Run

```sh
LLAMA_ROUTE_TRACE=/tmp/route.txt ./build/bin/llama-route-trace \
  -m model.gguf -c 2048 -n 400 -p "Write a detailed essay about photosynthesis."
```

Set `-n 0` to trace the prompt only. The prompt is evaluated one token at a time
and the optional greedy continuation is appended, so every row maps to exactly
one position.

## Output format

One line per `(position, routed layer)`:

```
<call> <pos> <layer> <expert>:<gate> <expert>:<gate> ...
```

`call` is an opaque counter, `gate` is a placeholder (`1`) that the offline
consumers ignore. Example:

```
0 0 0 238:1 112:1 120:1 43:1 56:1 106:1 200:1 6:1
1 0 1 214:1 182:1 143:1 86:1 112:1 1:1 19:1 167:1
```

The implementation reuses the existing backend eval callback: the graph already
names the selected-expert tensor `ffn_moe_topk-<il>`
(`llama_context::graph_get_cb`), so no core changes are required.

## Analysis

Clone colibri and run its offline calculators on the trace:

```sh
python3 c/tools/route_coupling_report.py route.txt   # dependence + prefetch recall
python3 c/tools/route_pairs.py route.coli_pairs route.txt
```

`route_coupling_report.py` reports, for a held-out split:

- `[1]` cross-layer co-activation lift vs independence, and
- `[2]` simulated prefetch recall of the true top-K at budgets of 8/16/32
  experts per layer, comparing a marginal-frequency predictor against a
  coupled `(layer, expert) -> next-layer experts` predictor.

The coupled predictor uses the observed layer-L routing set, i.e. a short
lookahead available once L's router has run. It is not the longer
stale-state prediction used by colibri's `PILOT`.

## Measured (2026-09-13, RTX 5060 Ti + RTX 3060)

Decoder-only greedy generation, `--flash-attn on`, `top-K = 8`.

Qwen4-exp (Qwen3.8-Flash-Next apex, 48 layers, 512 experts, 437 positions):

| prefetch | budget 8/layer | budget 16/layer | budget 32/layer |
| --- | --- | --- | --- |
| marginal | 28.9% | 38.3% | 49.9% |
| coupled  | 37.2% (+8.3pp) | 51.2% (+13.0pp) | 64.7% (+14.8pp) |

L->L+1 lift vs independence: median 2.77x, p90 16.94x, p99 101.67x.

Qwopus3.6-35B-A3B (40 layers, 256 experts, 837 positions):

| prefetch | budget 8/layer | budget 16/layer | budget 32/layer |
| --- | --- | --- | --- |
| marginal | 25.0% | 36.1% | 51.9% |
| coupled  | 41.3% (+16.3pp) | 57.2% (+21.1pp) | 72.4% (+20.5pp) |

L->L+1 lift vs independence: median 2.17x, p90 10.83x, p99 58.50x.

Both models show genuine cross-layer routing structure, so a coupling-table
prefetch has real recall to work with. Whether it pays off end-to-end still
depends on the cost of a wrong prefetch and on how many of the predicted
experts are already resident; the same conclusion is recorded in colibri's own
decode-failure ledger for resident decode.
