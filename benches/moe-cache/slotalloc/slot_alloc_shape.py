"""Why the non-uniform gain is negligible: shape of the uniform miss curve (near-linear =>
 Jenson gap ~ 0), per-slot price in MiB, grid-constrained optima, simple two-level allocations.
"""
import collections
import heapq
import re
import statistics
import sys

import numpy as np

sys.path.insert(0, '/mnt/WorkDisk/harness/multiturn-ctx/slotalloc')
import moe_cache as M

TRACE = '/mnt/WorkDisk/harness/multiturn-ctx/demand-trace-dt42.txt'
LEDGER = '/mnt/WorkDisk/harness/multiturn-ctx/moe-steps-dt42.log'
NLAY, BUDGET, MAXC = 48, 2016, 320
layers, nstep = M.parse_trace(TRACE)
nxts = [M.next_use_tables(s, nstep) for s in layers]
tps = lambda mps: 1000.0 / (28.3 + 0.2842 * mps)

# ---- ledger: MiB per missed expert (grounding for the cost of a slot) ------------
need = res = mis = 0
mib = 0.0
for line in open(LEDGER):
    kv = dict(p.split('=') for p in line.split() if '=' in p)
    if 'moe-step' not in line or 'misses' not in kv:
        continue
    mis += int(kv['misses']); mib += float(kv['miss_mib'])
    need += int(kv['need']); res += int(kv['resident'])
mib_per_miss = mib / mis
print("== VRAM price of one resident expert (ledger) ==")
print("  ledger totals: need=%d resident=%d misses=%d miss_mib=%.1f" % (need, res, mis, mib))
print("  MiB per miss = %.3f  -> one expert slot = %.3f MiB (all 3 matrices)" % (mib_per_miss, mib_per_miss))
print("  cache at 42 slots/layer = %d experts = %.1f MiB (%.2f GiB)"
      % (BUDGET, BUDGET * mib_per_miss, BUDGET * mib_per_miss / 1024))
print("  +1 uniform slot/layer (+48 experts) = +%.1f MiB" % (48 * mib_per_miss))

# ---- uniform curve shape around 42 ----------------------------------------------
curve = {}
for C in range(1, MAXC + 1):
    curve[C] = sum(M.sim_min_b(layers[L], nxts[L], nstep, C) for L in range(NLAY)) / nstep
print("\n== uniform miss curve shape (per-step) ==")
print("  C: misses/step  (delta vs previous)")
for C in range(36, 50):
    print("  %2d: %7.2f   %+.2f" % (C, curve[C], curve[C] - curve[C - 1]))
print("  C=60: %.2f  C=80: %.2f  C=100: %.2f" % (curve[60], curve[80], curve[100]))
d = [curve[C] - curve[C - 1] for C in range(2, 150)]
print("  marginal (per +48 slots, i.e. per +1 uniform) at 42: %.3f, at 60: %.3f, at 100: %.3f misses/step"
      % (curve[43] - curve[42], curve[61] - curve[60], curve[101] - curve[100]))
print("  convexity of the uniform curve: |second difference| max over C=5..100 = %.4f (0 => linear)"
      % max(abs(d[i + 1] - d[i]) for i in range(len(d) - 1)))

# ---- how skew-tolerant is the optimum? uniform is the max-entropy allocation ----
uni42 = curve[42]

# ---- grid-constrained optimum: every C_L a multiple of k, sum = 2016 -----------
def greedy_grid(curves, budget, k):
    C = [k] * NLAY
    spent = k * NLAY

    def gain(L, c):
        hi = min(c + k, MAXC)
        return curves[L][c - 1] - curves[L][hi - 1]

    if spent > budget:
        return None
    h = [(-gain(L, k), L, k) for L in range(NLAY)]
    heapq.heapify(h)
    while spent + k <= budget:
        g, L, c = heapq.heappop(h)
        C[L] = c + k
        spent += k
        heapq.heappush(h, (-gain(L, C[L]), L, C[L]))
    return C

curves_all = {L: np.array([M.sim_min_b(layers[L], nxts[L], nstep, C) for C in range(1, MAXC + 1)], float)
              for L in range(NLAY)}
print("\n== granularity-constrained optima (C_L must be a multiple of k, sum = 2016) ==")
for k in (1, 2, 3, 6, 7, 21, 42):
    C = greedy_grid(curves_all, BUDGET, k)
    if C is None:
        continue
    tot = sum(curves_all[L][C[L] - 1] for L in range(NLAY)) / nstep
    print("  k=%-2d: %7.2f misses/step  %.3f t/s  (min=%d max=%d)  gain vs uniform42 = %+.3f"
          % (k, tot, tps(tot), min(C), max(C), uni42 - tot))

# ---- simple two-level allocations: only layer 0 gets more, rest stay at 42 -------
print("\n== simple 'give the hottest layer more, take from everyone else equally' ==")
for extra in (6, 12, 18, 24, 48):
    C = [42] * NLAY
    take = extra // (NLAY - 1)
    C[0] = 42 + extra - take * (NLAY - 1) - take * 0   # layer 0 gets extra, others give take
    for L in range(1, NLAY):
        C[L] = 42 - take
    C[0] = BUDGET - sum(C[1:])
    tot = sum(curves_all[L][C[L] - 1] for L in range(NLAY)) / nstep
    print("  layer0 -> %3d, others -> %d : %7.2f misses/step  %.3f t/s  gain %+.3f"
          % (C[0], C[1], tot, tps(tot), uni42 - tot))

# ---- stability of the marginal value per layer at 42 (why hold-out fails) --------
mg = [curves_all[L][41] - curves_all[L][42] for L in range(NLAY)]
print("\n== per-layer marginal value of the 43rd slot (absolute misses over %d steps) ==" % nstep)
print("  min=%d max=%d mean=%.2f sd=%.2f  -> spread of %.0f%% around the mean"
      % (min(mg), max(mg), statistics.mean(mg), statistics.pstdev(mg),
         100 * statistics.pstdev(mg) / statistics.mean(mg)))
print("  per-step: %.4f misses/step per slot (best layer) vs %.4f (worst)" % (max(mg) / nstep, min(mg) / nstep))
print("  => a perfect reallocation can only move slots from a %.4f to a %.4f value: the whole" % (min(mg) / nstep, max(mg) / nstep))
print("     spread over all 48 layers buys at most %.2f misses/step ((max-min)*48/2/%d)"
      % ((max(mg) - min(mg)) * NLAY / 2 / nstep, nstep))

# ---- per-layer structure: distinct experts and compulsory misses -----------------
distinct = [len({e for t in range(nstep) for e in layers[L][t]}) for L in range(NLAY)]
comp = [curves_all[L][MAXC - 1] for L in range(NLAY)]
print("\n== layer-level structure (why layer 0 is different) ==")
top = sorted(range(NLAY), key=lambda L: -distinct[L])[:6]
for L in top:
    print("  layer %2d: distinct=%3d compulsory=%3d  C=42 misses=%.2f/step  C=42 miss/distinct dens=%.2f"
          % (L, distinct[L], comp[L], curves_all[L][41] / nstep, curves_all[L][41] / distinct[L]))
print("  layer 0 distinct=%d vs median layer distinct=%d (%.1fx)"
      % (distinct[0], int(statistics.median(distinct)), distinct[0] / statistics.median(distinct)))
print("  layer 0 needs C=%d to reach its compulsory floor %d; uniform 42 is far below that"
      % (distinct[0], comp[0]))
