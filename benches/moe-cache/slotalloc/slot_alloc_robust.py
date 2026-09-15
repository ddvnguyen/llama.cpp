"""Cross-checks for the non-uniform slot allocation result:
  1. exact DP optimum (numpy min-plus convolution) at the real budget,
  2. granularity-constrained allocations (slots must move in blocks),
  3. hold-out: fit allocation on the first half of the steps, evaluate on the second half,
  4. stability of the per-layer skew between the two halves.
"""
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


def curves_for(win_a, win_b, maxc=MAXC):
    """Per-layer MIN miss counts (absolute, over the window) for C=1..maxc."""
    out = {}
    for L in range(NLAY):
        seq = layers[L][win_a:win_b]
        nxt = M.next_use_tables(seq, len(seq))
        out[L] = np.array([M.sim_min_b(seq, nxt, len(seq), C) for C in range(1, maxc + 1)], float)
    return out


def greedy(curves, budget):
    C = [1] * NLAY
    spent = NLAY

    def gain(L, C_):
        return curves[L][C_ - 1] - curves[L][C_]      # curve index C-1 holds m(C)

    h = [(-gain(L, 1), L, 1) for L in range(NLAY)]
    heapq.heapify(h)
    while spent < budget:
        g, L, c = heapq.heappop(h)
        C[L] = c + 1
        spent += 1
        heapq.heappush(h, (-gain(L, c + 1), L, c + 1))
    return C


def total(curves, C):
    return sum(curves[L][C[L] - 1] for L in range(NLAY))


def tps(mps):
    return 1000.0 / (28.3 + 0.2842 * mps)


def dp_exact(curves, budget, maxc=MAXC):
    """Exact min of sum_L curve_L[C_L] with sum C_L = budget, C_L >= 1. Min-plus DP."""
    INF = 1e18
    prev = np.full(budget + 1, INF)
    prev[0] = 0.0
    for L in range(NLAY):
        cur = np.full(budget + 1, INF)
        for c in range(1, min(maxc, budget) + 1):
            v = prev[:budget + 1 - c] + curves[L][c - 1]
            np.minimum(cur[c:], v, out=cur[c:])
        prev = cur
    return prev[budget]


print("== 1. exact DP optimum at budget %d ==" % BUDGET)
curves_all = curves_for(0, nstep)
g_alloc = greedy(curves_all, BUDGET)
g_tot = total(curves_all, g_alloc)
print("greedy   : %.2f misses/step  %.3f t/s" % (g_tot / nstep, tps(g_tot / nstep)))
d_tot = dp_exact(curves_all, BUDGET)
print("DP exact : %.2f misses/step  %.3f t/s" % (d_tot / nstep, tps(d_tot / nstep)))
print("uniform42: %.2f misses/step  %.3f t/s" % (total(curves_all, [42] * NLAY) / nstep,
                                                 tps(total(curves_all, [42] * NLAY) / nstep)))
print("greedy == DP optimum:", abs(g_tot - d_tot) < 1e-6)

print("\n== 2. granularity constraints (slots move in blocks of k) ==")
for k in (1, 2, 3, 6, 12):
    # coarse grid: C in {1} U {k, 2k, ...} -- approximate by rounding the greedy result onto the
    # same grid, then re-running greedy restricted to grid multiples of k.
    base = 2016 // (NLAY)  # 42; keep total slots == 2016 exactly by construction below
    # grid allocation: start every layer at 42 rounded down to a multiple of k, redistribute the
    # leftover in +k chunks to the layer with the best chunk gain.
    C = [(42 // k) * k] * NLAY
    left = BUDGET - sum(C)

    def chunk_gain(L, c, k):
        hi = min(c + k, MAXC)
        return curves_all[L][c - 1] - curves_all[L][hi - 1], hi - c

    h = []
    for L in range(NLAY):
        g, d = chunk_gain(L, C[L], k)
        h.append((-g, L, C[L], d))
    heapq.heapify(h)
    while left > 0:
        g, L, c, d = heapq.heappop(h)
        if d == 0:
            continue
        C[L] = c + d
        left -= d
        g2, d2 = chunk_gain(L, C[L], k)
        heapq.heappush(h, (-g2, L, C[L], d2))
    tot = total(curves_all, C) / nstep
    print("  block k=%-2d -> %7.2f misses/step  %.3f t/s   alloc min/max %d/%d" % (
        k, tot, tps(tot), min(C), max(C)))

print("\n== 3. hold-out: fit on steps 0-99, evaluate on steps 100-198 ==")
A = curves_for(0, 100)
B = curves_for(100, nstep)
tail = nstep - 100
alloc_AB = greedy(A, BUDGET)
print("uniform42 on window B : %7.2f misses/step  %.3f t/s" % (total(B, [42] * NLAY) / tail,
                                                               tps(total(B, [42] * NLAY) / tail)))
print("greedy(A) on window B : %7.2f misses/step  %.3f t/s" % (total(B, alloc_AB) / tail,
                                                               tps(total(B, alloc_AB) / tail)))
alloc_B = greedy(B, BUDGET)
print("greedy(B) on window B : %7.2f misses/step  %.3f t/s   (oracle on that window)"
      % (total(B, alloc_B) / tail, tps(total(B, alloc_B) / tail)))
print("transferable gain of the first-half allocation: %.2f misses/step"
      % ((total(B, [42] * NLAY) - total(B, alloc_AB)) / tail))

print("\n== 4. is the per-layer skew stable between halves? ==")
mA = [A[L][41] / 100 for L in range(NLAY)]
mB = [B[L][41] / tail for L in range(NLAY)]
print("per-layer MIN misses/step at C=42:  window A mean %.3f sd %.3f | window B mean %.3f sd %.3f"
      % (statistics.mean(mA), statistics.pstdev(mA), statistics.mean(mB), statistics.pstdev(mB)))
r = np.corrcoef(mA, mB)[0, 1]
print("Pearson correlation of per-layer C=42 misses across halves: %.3f" % r)
print("window-A top-8 heaviest layers:", sorted(range(NLAY), key=lambda L: -mA[L])[:8])
print("window-B top-8 heaviest layers:", sorted(range(NLAY), key=lambda L: -mB[L])[:8])
print("allocation from window A (per layer):")
print("  " + " ".join("%d:%d" % (L, alloc_AB[L]) for L in range(NLAY)))
print("allocation from window B (per layer):")
print("  " + " ".join("%d:%d" % (L, alloc_B[L]) for L in range(NLAY)))
