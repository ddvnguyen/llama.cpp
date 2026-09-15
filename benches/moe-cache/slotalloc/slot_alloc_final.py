"""Final numbers: exact DP argmin allocation, and why a trace-fitted allocation cannot transfer
(noise in the per-layer marginal estimates vs the spread they are meant to exploit)."""
import heapq
import statistics
import sys

import numpy as np

sys.path.insert(0, '/mnt/WorkDisk/harness/multiturn-ctx/slotalloc')
import moe_cache as M

TRACE = '/mnt/WorkDisk/harness/multiturn-ctx/demand-trace-dt42.txt'
NLAY, BUDGET, MAXC = 48, 2016, 320
layers, nstep = M.parse_trace(TRACE)
nxts = [M.next_use_tables(s, nstep) for s in layers]
tps = lambda mps: 1000.0 / (28.3 + 0.2842 * mps)


def win(a, b):
    seqs = [layers[L][a:b] for L in range(NLAY)]
    nxt = [M.next_use_tables(s, len(s)) for s in seqs]
    return {L: np.array([M.sim_min_b(seqs[L], nxt[L], len(seqs[L]), C) for C in range(1, MAXC + 1)], float)
            for L in range(NLAY)}


full = win(0, nstep)

# ---- exact DP with backpointers ------------------------------------------------
INF = 1e18
prev = np.full(BUDGET + 1, INF); prev[0] = 0.0
choice = np.zeros((NLAY, BUDGET + 1), dtype=np.int16)
for L in range(NLAY):
    cur = np.full(BUDGET + 1, INF)
    best = np.zeros(BUDGET + 1, dtype=np.int16)
    for c in range(1, MAXC + 1):
        if c > BUDGET:
            break
        v = prev[:BUDGET + 1 - c] + full[L][c - 1]
        upd = v < cur[c:]
        cur[c:][upd] = v[upd]
        best[c:][upd] = c
    choice[L] = best
    prev = cur

dp_val = prev[BUDGET]
alloc = [0] * NLAY
b = BUDGET
for L in range(NLAY - 1, -1, -1):
    c = int(choice[L][b])
    alloc[L] = c
    b -= c
print("== exact optimum (DP over the 48 per-layer Belady curves) ==")
print("  dp value  : %.2f misses/step  %.4f t/s" % (dp_val / nstep, tps(dp_val / nstep)))
print("  dp alloc  : min=%d max=%d mean=%.2f sd=%.2f  sum|C-42|=%d" % (
    min(alloc), max(alloc), statistics.mean(alloc), statistics.pstdev(alloc), sum(abs(c - 42) for c in alloc)))
print("  vector    : " + " ".join(str(c) for c in sorted(alloc)))
print("  per-layer : " + " ".join("%d:%d" % (L, alloc[L]) for L in range(NLAY)))
u42 = sum(full[L][41] for L in range(NLAY))
u43 = sum(full[L][42] for L in range(NLAY))
print("  uniform42 : %.2f misses/step  %.4f t/s" % (u42 / nstep, tps(u42 / nstep)))
print("  uniform43 : %.2f misses/step  %.4f t/s  (%d slots, +%.1f MiB)" % (u43 / nstep, tps(u43 / nstep), 43 * NLAY, 48 * 1.796))
print("  gain optimal-realloc vs uniform42 : %.2f misses/step (%+.4f t/s, %+.2f%%)"
      % ((u42 - dp_val) / nstep, tps(dp_val / nstep) - tps(u42 / nstep), 100 * (tps(dp_val / nstep) / tps(u42 / nstep) - 1)))
print("  gain uniform 42->43              : %.2f misses/step (%+.4f t/s, %+.2f%%)"
      % ((u42 - u43) / nstep, tps(u43 / nstep) - tps(u42 / nstep), 100 * (tps(u43 / nstep) / tps(u42 / nstep) - 1)))

# ---- how estimable is each layer's curve? split-half marginal comparison --------
print("\n== split-half estimability of the per-layer marginal value at C=42 ==")
A = win(0, 100); B = win(100, nstep)
nA, nB = 100, nstep - 100
mgA = np.array([(A[L][41] - A[L][42]) / nA for L in range(NLAY)])
mgB = np.array([(B[L][41] - B[L][42]) / nB for L in range(NLAY)])
mgF = np.array([(full[L][41] - full[L][42]) / nstep for L in range(NLAY)])
print("  marginal value of slot #43 [misses/step per slot]")
print("    full trace : mean %.4f  sd %.4f  range [%.4f, %.4f]" % (mgF.mean(), mgF.std(), mgF.min(), mgF.max()))
print("    first half : mean %.4f  sd %.4f" % (mgA.mean(), mgA.std()))
print("    second half: mean %.4f  sd %.4f" % (mgB.mean(), mgB.std()))
print("    corr(mgA, mgB) = %.3f ; mean |mgA-mgB| = %.4f (vs spread max-min = %.4f)"
      % (np.corrcoef(mgA, mgB)[0, 1], np.abs(mgA - mgB).mean(), mgF.max() - mgF.min()))
print("    => the *ordering* of layers is stable (corr %.2f) but the *differences* that a" % np.corrcoef(mgA, mgB)[0, 1])
print("       reallocation exploits (range %.4f) are only %.1fx the half-to-half disagreement (%.4f)."
      % (mgF.max() - mgF.min(), (mgF.max() - mgF.min()) / np.abs(mgA - mgB).mean(), np.abs(mgA - mgB).mean()))

# ---- hold-out in per-step terms, both directions --------------------------------
def greedy(curves, budget):
    C = [1] * NLAY; spent = NLAY
    g = lambda L, c: curves[L][c - 1] - curves[L][c]
    h = [(-g(L, 1), L, 1) for L in range(NLAY)]; heapq.heapify(h)
    while spent < budget:
        _, L, c = heapq.heappop(h); C[L] = c + 1; spent += 1
        heapq.heappush(h, (-g(L, c + 1), L, c + 1))
    return C

aA, aB = greedy(A, BUDGET), greedy(B, BUDGET)
print("\n== hold-out transfer (budget %d both sides) ==" % BUDGET)
for name, fit, test, n in (("fit first half -> test second half", aA, B, nB),
                           ("fit second half -> test first half", aB, A, nA)):
    u = sum(test[L][41] for L in range(NLAY)) / n
    o = sum(test[L][fit[L] - 1] for L in range(NLAY)) / n
    oracle = sum(test[L][greedy(test, BUDGET)[L] - 1] for L in range(NLAY)) / n
    print("  %-34s uniform42 %.2f | fitted %.2f | in-sample oracle %.2f | transfer gain %+.2f misses/step"
          % (name, u, o, oracle, u - o))
