"""Throwaway offline analysis helpers: per-layer Belady/MIN miss curves for a MoE expert cache.

Model: 48 independent per-layer caches. At each decode step every layer demands 10 experts
(ids are distinct within a (step,layer) line). A cache of C slots at a layer holds C experts.
Misses = demands whose expert is not resident at the moment of the demand.

Policies:
  min_b : textbook Belady/MIN. The fetched expert occupies a slot and the victim is chosen among
          the *other* residents (farthest next use). This is the physically correct model for a
          fixed slot array, hence the achievable optimum.
  min_a : "keep the C experts with the nearest next use out of residents+demands", allowing the
          just-fetched expert to be dropped (needs C+1 physical slots transiently). Included only
          to show the calibration difference.
  lru   : plain LRU.
"""
import heapq

INF = 10 ** 9


def parse_trace(path):
    steps = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            kv = dict(p.split('=') for p in line.split())
            steps.setdefault(int(kv['step']), {})[int(kv['layer'])] = [
                int(x) for x in kv['ids'].split(',')]
    nstep = max(steps) + 1
    nlayer = max(max(v) for v in steps.values()) + 1
    return [[steps[t][L] for t in range(nstep)] for L in range(nlayer)], nstep


def next_use_tables(seq, nstep):
    """nxt[t][k] = step of the next demand of the same expert strictly after t, else INF."""
    last = {}
    out = [[INF] * len(seq[t]) for t in range(nstep)]
    for t in range(nstep - 1, -1, -1):
        for k, e in enumerate(seq[t]):
            out[t][k] = last.get(e, INF)
        for e in seq[t]:
            last[e] = t
    return out


def sim_min_b(seq, nxt, nstep, C):
    live = {}
    heap = []
    misses = 0
    for t in range(nstep):
        row = nxt[t]
        for k, e in enumerate(seq[t]):
            w = row[k]
            if e in live:
                live[e] = w
                heapq.heappush(heap, (-w, e))
                continue
            misses += 1
            if len(live) >= C:
                while True:
                    nw, ve = heapq.heappop(heap)
                    if live.get(ve) == -nw:
                        break
                del live[ve]
            live[e] = w
            heapq.heappush(heap, (-w, e))
    return misses


def sim_min_a(seq, nxt, nstep, C):
    """Optimal for the relaxed model where cache_{t} = C nearest-next-use of (cache_{t-1} + D_t)."""
    live = {}
    heap = []
    misses = 0
    for t in range(nstep):
        row = nxt[t]
        for k, e in enumerate(seq[t]):
            w = row[k]
            if e not in live:
                misses += 1
                live[e] = w
                heapq.heappush(heap, (-w, e))
            else:
                live[e] = w
                heapq.heappush(heap, (-w, e))
            while len(live) > C:
                while True:
                    nw, ve = heapq.heappop(heap)
                    if live.get(ve) == -nw:
                        break
                del live[ve]
    return misses


def sim_lru(seq, nstep, C):
    from collections import OrderedDict
    live = OrderedDict()
    misses = 0
    for t in range(nstep):
        for e in seq[t]:
            if e in live:
                live.move_to_end(e)
            else:
                misses += 1
                live[e] = None
                if len(live) > C:
                    live.popitem(last=False)
    return misses
