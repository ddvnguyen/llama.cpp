"""Offline: non-uniform per-layer slot allocation for the MoE expert cache.

Budget is fixed at 42*48 = 2016 slots. Question: does redistributing slots across the 48
independent per-layer caches (Belady/MIN optimum per layer) beat uniform 42?
Throwaway analysis script; no C++ touched.
"""
import heapq
import json
import statistics
import sys

sys.path.insert(0, '/mnt/WorkDisk/harness/multiturn-ctx/slotalloc')
import moe_cache as M

TRACE = '/mnt/WorkDisk/harness/multiturn-ctx/demand-trace-dt42.txt'
BUDGET = 42 * 48
MAXC = 320

layers, nstep = M.parse_trace(TRACE)
nlay = len(layers)
nxts = [M.next_use_tables(s, nstep) for s in layers]

# ---- per-layer demand statistics -------------------------------------------------
demands = [sum(len(layers[L][t]) for t in range(nstep)) for L in range(nlay)]
distinct = [len({e for t in range(nstep) for e in layers[L][t]}) for L in range(nlay)]
compulsory = [  # first-touch count == misses with unlimited capacity
    M.sim_min_b(layers[L], nxts[L], nstep, MAXC) for L in range(nlay)]

# ---- per-layer Belady/MIN miss curve ---------------------------------------------
curves = {}
for L in range(nlay):
    curves[L] = {C: M.sim_min_b(layers[L], nxts[L], nstep, C) for C in range(1, MAXC + 1)}
    # saturate beyond MAXC at the compulsory count
    for C in range(MAXC + 1, MAXC + 64):
        curves[L][C] = compulsory[L]

def m(L, C):
    return curves[L][C]

def tps(misses_per_step):
    return 1000.0 / (28.3 + 0.2842 * misses_per_step)

def total(alloc):
    return sum(m(L, alloc[L]) for L in range(nlay)) / nstep

def gains_for(L, C):
    return m(L, C) - m(L, C + 1)

# ---- uniformity of the demand ----------------------------------------------------
print("== per-layer demand skew ==")
print("distinct experts/layer: min=%d max=%d mean=%.1f sd=%.1f" % (
    min(distinct), max(distinct), statistics.mean(distinct), statistics.pstdev(distinct)))
print("demands/layer:          min=%d max=%d mean=%.1f (expected %d)" % (
    min(demands), max(demands), statistics.mean(demands), 10 * nstep))
print("compulsory misses/layer: min=%d max=%d mean=%.2f" % (
    min(compulsory), max(compulsory), statistics.mean(compulsory)))
print("compulsory total/step: %.2f" % (sum(compulsory) / nstep))

# ---- uniformity of the miss curve at uniform 42 ----------------------------------
u42 = [m(L, 42) for L in range(nlay)]
print("\n== per-layer MIN misses at C=42 ==")
print("min=%d max=%d mean=%.2f sd=%.2f  (sum/step=%.2f)" % (
    min(u42), max(u42), statistics.mean(u42), statistics.pstdev(u42), sum(u42) / nstep))
print("sorted per-layer C=42 misses (layer:misses):")
print("  " + " ".join("%d:%d" % (L, u42[L]) for L in sorted(range(nlay), key=lambda L: -u42[L])))

# marginal gain of a slot at 42, per layer
gain42 = [gains_for(L, 42) for L in range(nlay)]
print("\nmarginal gain of the 43rd slot per layer: min=%d max=%d mean=%.3f sd=%.2f" % (
    min(gain42), max(gain42), statistics.mean(gain42), statistics.pstdev(gain42)))
print("  " + " ".join("%d:%d" % (L, gain42[L]) for L in sorted(range(nlay), key=lambda L: -gain42[L])))

# ---- concavity check (greedy optimality precondition) ----------------------------
viol = 0
worst = 0
for L in range(nlay):
    for C in range(1, MAXC):
        g0, g1 = gains_for(L, C), gains_for(L, C + 1)
        if g1 > g0:
            viol += 1
            worst = max(worst, g1 - g0)
print("\n== concavity of per-layer curves ==\nstrictly increasing marginal gains (violations): %d, worst=+%d" % (viol, worst))

# ---- uniform baselines -----------------------------------------------------------
print("\n== uniform baselines ==")
uni = {}
for C in (28, 36, 40, 41, 42, 43, 44, 48, 56, 64):
    tot = sum(m(L, C) for L in range(nlay)) / nstep
    uni[C] = tot
    print("  uniform C=%2d (%5d slots): %7.1f misses/step  %6.2f t/s" % (C, C * nlay, tot, tps(tot)))
d1 = uni[43] - uni[42]
print("  marginal value of +1 uniform slot/layer (43 each, 2064 slots): %.2f misses/step (%.3f t/s)" % (d1, tps(uni[43]) - tps(uni[42])))
print("  slot cost: 48 slots = %.1f MiB/bank/layer if 1 expert slot = 2.05MiB/3 ... (geometry: see leader)" % 0)

# ---- greedy marginal allocation --------------------------------------------------
alloc = [1] * nlay
spent = nlay
heap = [(-gains_for(L, 1), L, 1) for L in range(nlay)]
heapq.heapify(heap)
series = {spent: total(alloc)}
while spent < BUDGET:
    ng, L, C = heapq.heappop(heap)
    assert alloc[L] == C, (L, C, alloc[L])
    alloc[L] = C + 1
    spent += 1
    series[spent] = total(alloc)
    heapq.heappush(heap, (-gains_for(L, C + 1), L, C + 1))
opt_total = total(alloc)
print("\n== greedy marginal allocation, budget %d slots ==" % BUDGET)
print("optimal total: %.2f misses/step  %.2f t/s   (uniform42: %.2f / %.2f t/s)"
      % (opt_total, tps(opt_total), uni[42], tps(uni[42])))
print("gain over uniform42: %.2f misses/step  (+%.3f t/s, +%.2f%%)"
      % (uni[42] - opt_total, tps(opt_total) - tps(uni[42]), 100 * (tps(opt_total) / tps(uni[42]) - 1)))
print("allocation: min=%d max=%d mean=%.3f sd=%.2f" % (min(alloc), max(alloc), statistics.mean(alloc), statistics.pstdev(alloc)))
print("departure from 42: max |C-42| = %d, sum |C-42| = %d" % (max(abs(c - 42) for c in alloc), sum(abs(c - 42) for c in alloc)))
print("sorted allocation vector:")
print("  " + " ".join(str(c) for c in sorted(alloc)))
print("per-layer allocation (layer:C, sorted by C desc):")
print("  " + " ".join("%d:%d" % (L, alloc[L]) for L in sorted(range(nlay), key=lambda L: -alloc[L])))
# equalisation check: marginal gains should all be equal at the optimum
mg = [gains_for(L, alloc[L]) for L in range(nlay)]
print("marginal gain at the optimum: min=%d max=%d (should be tight)" % (min(mg), max(mg)))
print("  " + " ".join("%d:%d" % (L, mg[L]) for L in sorted(range(nlay), key=lambda L: -mg[L])))

# which layers gained/lost
print("\nlayers moved (-=lost slots vs 42):")
print("  " + " ".join("%d:%+d" % (L, alloc[L] - 42) for L in range(nlay)))

# ---- curve of achievable misses vs budget (non-uniform) --------------------------
print("\n== budget curve (greedy/non-uniform vs uniform-equivalent) ==")
for b in (2016, 2064, 2112, 2208, 2304, 2400, 2688, 3072, 4032):
    # cheap: recompute greedy up to b from the series if available
    if b in series:
        print("  budget %5d (%4.2f slots/layer avg): %7.2f misses/step  %6.2f t/s" % (b, b / nlay, series[b], tps(series[b])))
series_out = {k: v for k, v in series.items() if k % 48 == 0 or k in (2016, 2064)}

# ---- what uniform C matches the optimal non-uniform budget -----------------------
lo, hi = 1, MAXC
while lo < hi:
    mid = (lo + hi) // 2
    if sum(m(L, mid) for L in range(nlay)) / nstep > opt_total:
        lo = mid + 1
    else:
        hi = mid
print("\nequivalent uniform C for the same total misses: C=%d (%.2f slots/layer)" % (lo, lo))

# ---- DP cross-check (exact optimum, small budget on the same curves) -------------
def dp_opt(budget):
    NEG = float('inf')
    prev = [NEG] * (budget + 1)
    prev[0] = 0
    for L in range(nlay):
        cur = [NEG] * (budget + 1)
        for used in range(budget + 1):
            if prev[used] == NEG:
                continue
            for C in range(1, min(MAXC, budget - used) + 1):
                v = prev[used] + m(L, C)
                if v < cur[used + C]:
                    cur[used + C] = v
        prev = cur
    return prev[budget]

# DP is O(layers * budget * MAXC) -- run it at a reduced budget to validate greedy
for b in (192, 288, 480):
    g = None
    a2 = [1] * nlay
    h = [(-gains_for(L, 1), L, 1) for L in range(nlay)]
    heapq.heapify(h)
    sp = nlay
    while sp < b:
        ng, L, C = heapq.heappop(h)
        a2[L] = C + 1
        sp += 1
        heapq.heappush(h, (-gains_for(L, C + 1), L, C + 1))
    g = sum(m(L, a2[L]) for L in range(nlay))
    d = dp_opt(b)
    print("DP cross-check budget=%d: greedy=%d dp=%d  %s" % (b, g, d, "MATCH" if g == d else "MISMATCH"))

json.dump({"nstep": nstep, "uniform42": uni[42], "uniform43": uni[43], "opt": opt_total,
           "alloc": alloc, "curves": {str(L): curves[L] for L in range(nlay)},
           "distinct": distinct, "demands": demands, "compulsory": compulsory,
           "u42": u42, "gain42": gain42},
          open('/mnt/WorkDisk/harness/multiturn-ctx/slotalloc/slot_alloc_results.json', 'w'))
print("\nwrote slot_alloc_results.json")
