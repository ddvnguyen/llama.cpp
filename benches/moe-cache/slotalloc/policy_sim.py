#!/usr/bin/env python3
"""Offline replay of the per-layer expert cache over the captured demand trace.

Geometry: 48 independent caches (one per MoE layer), C slots each, 10 demanded
expert ids per (step, layer). A miss is a demanded id not resident when the step
starts (equivalently: one host fetch). t/s from the validated model
    t/s = 1000 / (28.3 + 0.2842 * misses_per_step).

Two access models are supported, and the difference matters:

  engine    - the model of the SHIPPED plan kernel (moe_grouped_plan_decode in
              ggml/src/ggml-cuda/moe-cache.cu). The 10 demands of a step are a
              batch: hits are counted against the step-start residency, victims
              are drawn only from slots NOT holding an expert demanded this step
              (the kernel marks those route_storage=1 before victim selection),
              and last_use/frequency are committed for all 10 at the end of the
              step, in demand order. This model reproduces the measured per-step
              ledger (moe-steps-dt42.log) EXACTLY for the shipped LFU-16 policy.
  textbook  - plain sequential LRU/LFU over each layer's flattened access
              sequence: each of the 10 ids is one access, a hit bumps recency and
              a miss evicts the globally coldest resident. This reproduces the
              owner's LRU reference number.

Both models are reported because they differ by ~5 misses/step on the same trace,
and any new policy would run inside the `engine` structure.
"""
import argparse
import bisect
import collections
import json
import random
import re
import statistics
import sys

TRACE_DEFAULT = "/mnt/WorkDisk/harness/multiturn-ctx/demand-trace-dt42.txt"
LEDGER_DEFAULT = "/mnt/WorkDisk/harness/multiturn-ctx/moe-steps-dt42.log"

BASE_MS = 28.3
MS_PER_MISS = 0.2842
N_EXPERTS = 512


def tps(misses):
    return 1000.0 / (BASE_MS + MS_PER_MISS * misses)


# ---------------------------------------------------------------------------
# trace / ledger
# ---------------------------------------------------------------------------

def load_trace(path):
    raw = collections.defaultdict(dict)
    n_layers = 0
    for line in open(path):
        m = re.match(r"step=(\d+) layer=(\d+) ids=([\d,]+)\s*$", line)
        if not m:
            continue
        step, layer = int(m.group(1)), int(m.group(2))
        raw[step][layer] = [int(x) for x in m.group(3).split(",")]
        n_layers = max(n_layers, layer + 1)
    steps = max(raw) + 1
    return [[raw[t][l] for l in range(n_layers)] for t in range(steps)], n_layers


def load_ledger(path):
    out = []
    for line in open(path):
        m = re.search(r"step=(\d+) need=(\d+) resident=(\d+) misses=(\d+)", line)
        if m:
            out.append(int(m.group(4)))
    return out


# ---------------------------------------------------------------------------
# offline optimum
# ---------------------------------------------------------------------------

def belady(trace, cap):
    """MIN over each layer's flattened access sequence: evict the resident whose
    next use is farthest away (never used again = +inf). Optimal offline."""
    T, L = len(trace), len(trace[0])
    total = 0
    for l in range(L):
        seq = [x for t in range(T) for x in trace[t][l]]
        n = len(seq)
        nxt = [n] * n
        seen = {}
        for i in range(n - 1, -1, -1):
            nxt[i] = seen.get(seq[i], n)
            seen[seq[i]] = i
        dist = {}
        miss = 0
        for i, eid in enumerate(seq):
            if eid in dist:
                dist[eid] = nxt[i]
                continue
            miss += 1
            if len(dist) < cap:
                dist[eid] = nxt[i]
                continue
            victim = max(dist, key=lambda r: dist[r])
            del dist[victim]
            dist[eid] = nxt[i]
        total += miss
    return total / T


# ---------------------------------------------------------------------------
# policy engine
# ---------------------------------------------------------------------------

class Layer:
    """One layer's cache. Frequency state is attached to the EXPERT (as the
    shipped kernel does), so a re-admitted expert keeps its decayed history;
    the 'lfuslot' base keeps it on the slot instead."""

    __slots__ = ("slots", "where", "age", "freq", "fepoch", "clock",
                 "hist", "sfreq", "s_seen")

    def __init__(self, cap):
        self.slots = [-1] * cap
        self.where = {}
        self.age = [0] * cap
        self.freq = {}
        self.fepoch = {}
        self.clock = 0
        self.hist = {}          # sliding-window base: expert -> ascending step list
        self.sfreq = [0] * cap  # slot-attached counters ('lfuslot')
        self.s_seen = [0] * cap


def eff_freq(lay, eid, epoch):
    """Shipped decay law: one right shift per elapsed epoch, zero after 32."""
    if eid < 0:
        return 0
    stored = lay.freq.get(eid, 0)
    if stored == 0:
        return 0
    elapsed = epoch - lay.fepoch.get(eid, 0)
    if elapsed <= 0:
        return stored
    return 0 if elapsed >= 32 else stored >> elapsed


def eff_slot(lay, slot, halflife, epoch):
    """'lfuslot' decay law: same right-shift law, counter lives on the slot."""
    stored = lay.sfreq[slot]
    if stored == 0:
        return 0
    elapsed = epoch - lay.s_seen[slot]
    if elapsed <= 0:
        return stored
    return 0 if elapsed >= 32 else stored >> elapsed


class Policy:
    """base    : 'lru' | 'lfu' | 'win' | 'lfuslot'
    halflife : epochs per halving of the demand counter (LFU); for 'win' it is
               the window length in steps
    protect  : None | 'pred' | 'recent' | 'pred+recent'
    admit    : only install when the new expert outranks the victim
    model    : 'engine' | 'textbook'
    """

    def __init__(self, cap, base="lfu", halflife=16, protect=None, protect_arg=None,
                 admit=False, model="engine", demand_guard=True):
        self.cap, self.base, self.halflife = cap, base, halflife
        self.protect, self.protect_arg, self.admit, self.model = protect, protect_arg, admit, model
        # demand_guard=True is the shipped kernel's rule: a slot holding an expert
        # demanded this step is never a victim. False allows evicting it right
        # after the demand was served (what textbook MIN does).
        self.demand_guard = demand_guard
        self.layers = None
        self.recent = None
        self.t = 0

    def init_layers(self, n_layers):
        self.layers = [Layer(self.cap) for _ in range(n_layers)]
        if self.protect in ("recent", "pred+recent"):
            self.recent = [collections.deque(maxlen=self.protect_arg) for _ in range(n_layers)]

    # lower key = colder = better victim
    def key(self, lay, slot, epoch):
        eid = lay.slots[slot]
        if self.base == "lru":
            return (0, lay.age[slot], slot)
        if self.base == "win":
            h = lay.hist.get(eid)
            n = 0 if not h else len(h) - bisect.bisect_left(h, self.t - self.halflife + 1)
            # hoist the protected flag above recency, not above the count
            return (n, lay.age[slot], slot)
        if self.base == "lfuslot":
            return (eff_slot(lay, slot, self.halflife, epoch), lay.age[slot], slot)
        return (eff_freq(lay, eid, epoch), lay.age[slot], slot)

    def _protected(self, l, Dset, predicted_by_layer):
        prot = set()
        if self.protect in ("pred", "pred+recent") and predicted_by_layer is not None:
            prot |= predicted_by_layer[l]
        if self.protect in ("recent", "pred+recent"):
            for s in self.recent[l]:
                prot |= s
        return prot - Dset

    def _pick_victim(self, lay, epoch, pool, protected_ids):
        """Hard protection: never evict a resident that is in the protected set.
        If the unprotected pool is too small, fall back to the unprotected ones
        that are coldest, then (only if protection cannot be honoured) to the
        whole pool."""
        if protected_ids:
            pref = [s for s in pool if lay.slots[s] not in protected_ids]
            if pref:
                pool = pref
        return min(pool, key=lambda s: self.key(lay, s, epoch))

    def _install(self, lay, eid, epoch, Dset, predicted_by_layer, l, t):
        pool = [s for s in range(self.cap) if not self.demand_guard or lay.slots[s] not in Dset]
        prot = self._protected(l, Dset, predicted_by_layer)
        slot = self._pick_victim(lay, epoch, pool, prot)
        if self.admit and lay.slots[slot] >= 0:
            new_f = eff_freq(lay, eid, epoch) + 1
            old_f = eff_freq(lay, lay.slots[slot], epoch)
            if new_f <= old_f:
                return  # fetch and use, but do not retain
        old = lay.slots[slot]
        if old >= 0:
            del lay.where[old]
        lay.slots[slot] = eid
        lay.where[eid] = slot
        lay.sfreq[slot] = 1
        lay.s_seen[slot] = epoch
        return slot

    def _bump(self, lay, eid, epoch, t):
        if self.base == "lfu":
            lay.freq[eid] = eff_freq(lay, eid, epoch) + 1
            lay.fepoch[eid] = epoch
        elif self.base == "lfuslot":
            slot = lay.where.get(eid)
            if slot is None:
                return
            lay.sfreq[slot] = eff_slot(lay, slot, self.halflife, epoch) + 1
            lay.s_seen[slot] = epoch
        elif self.base == "win":
            h = lay.hist.get(eid)
            if h is None:
                lay.hist[eid] = [t]
            else:
                h.append(t)
                if len(h) > 8 * self.halflife:      # keep memory bounded
                    del h[:len(h) - 4 * self.halflife]

    def step(self, t, demand_by_layer, predicted_by_layer=None):
        self.t = t
        epoch = t // self.halflife if self.halflife else 0
        misses = 0
        for l, D in enumerate(demand_by_layer):
            Dset = set(D)
            if self.model == "engine":
                misses += self._engine_batch(l, D, Dset, epoch, predicted_by_layer, t)
            else:
                misses += self._textbook(l, D, Dset, epoch, predicted_by_layer, t)
            if self.protect in ("recent", "pred+recent"):
                self.recent[l].append(Dset)
        return misses

    def _engine_batch(self, l, D, Dset, epoch, predicted, t):
        lay = self.layers[l]
        lay.clock += len(D)
        base = lay.clock - len(D)
        missing = [x for x in D if x not in lay.where]
        for eid in missing:
            self._install(lay, eid, epoch, Dset, predicted, l, t)
        for i, eid in enumerate(D):
            slot = lay.where.get(eid)
            if slot is not None:
                lay.age[slot] = base + i + 1
            self._bump(lay, eid, epoch, t)
        return len(missing)

    def _textbook(self, l, D, Dset, epoch, predicted, t):
        lay = self.layers[l]
        miss = 0
        for eid in D:
            lay.clock += 1
            if eid in lay.where:
                lay.age[lay.where[eid]] = lay.clock
                self._bump(lay, eid, epoch, t)
                continue
            miss += 1
            pool = list(range(self.cap))
            prot = self._protected(l, Dset, predicted)
            slot = self._pick_victim(lay, epoch, pool, prot)
            if self.admit and lay.slots[slot] >= 0:
                new_f = eff_freq(lay, eid, epoch) + 1
                if new_f <= eff_freq(lay, lay.slots[slot], epoch):
                    self._bump(lay, eid, epoch, t)
                    continue
            old = lay.slots[slot]
            if old >= 0:
                del lay.where[old]
            lay.slots[slot] = eid
            lay.where[eid] = slot
            lay.age[slot] = lay.clock
            lay.sfreq[slot] = 1
            lay.s_seen[slot] = epoch
            self._bump(lay, eid, epoch, t)
        return miss


class Oracle(Policy):
    """Offline reference family: a PERFECT predictor with horizon H steps.

    The victim is the resident whose next demand lies farthest away, where a
    next demand beyond H steps counts as never. Ties (all residents beyond the
    horizon) fall back to the base policy's key, so H = 1 is exactly "protect
    next step's demands, else LRU/LFU". H = 1,000,000 is the true Belady/MIN
    optimum. Not implementable; used to price a deeper predictor.
    """

    def __init__(self, cap, horizon, base="lru", halflife=16):
        super().__init__(cap, base=base, halflife=halflife, model="engine")
        self.horizon = horizon
        self.dist = None

    def init_layers(self, n_layers):
        super().init_layers(n_layers)
        self.t = 0

    def build_dist(self, trace):
        T, L = len(trace), len(trace[0])
        self.dist = []
        for l in range(L):
            seen = {}
            d = [None] * T
            for t in range(T - 1, -1, -1):
                cur = {e: nxt + 1 for e, nxt in seen.items()}
                for e in trace[t][l]:
                    cur[e] = 0
                d[t] = cur
                seen = cur
            self.dist.append(d)

    def key(self, lay, slot, epoch):
        eid = lay.slots[slot]
        if eid < 0:
            return (-10 ** 9, (0, 0, slot))       # free slot: always the best victim
        d = self.dist[self._l][self.t].get(eid)
        # d is the step distance to the next demand (0 = demanded this step, but
        # those slots are guarded out of the victim pool). A next demand beyond
        # the horizon counts as "never again", i.e. the best victim.
        far = self.horizon + 1 if d is None else min(d, self.horizon + 1)
        base_key = super().key(lay, slot, epoch)
        return (-far, base_key)                   # farthest next use = best victim

    def step(self, t, demand_by_layer, predicted_by_layer=None):
        self.t = t
        epoch = t // self.halflife if self.base == "lfu" else 0
        misses = 0
        for l, D in enumerate(demand_by_layer):
            self._l = l
            misses += self._engine_batch(l, D, set(D), epoch, None, t)
            if self.protect in ("recent", "pred+recent"):
                self.recent[l].append(set(D))
        return misses


# ---------------------------------------------------------------------------
# predictors
# ---------------------------------------------------------------------------

def predict_perfect(trace, t, rng=None):
    if t + 1 >= len(trace):
        return [set() for _ in trace[0]]
    return [set(ids) for ids in trace[t + 1]]


def predict_width(trace, t, rng, width, precision):
    """Model of the look-ahead filter output, `width` ids per layer.

    Each of the `width` emitted ids is correct with probability `precision` (a
    uniformly drawn, not yet used, member of the true next-step set) and
    otherwise a uniformly drawn non-member of the 512 experts. With
    width 8 / precision 0.8621 this reproduces BOTH measured quantities:
    E[correct] = 8 * 0.8621 = 6.897 -> recall 86.21% of the 8 emitted ids and
    coverage 68.97% of the 10 used. width 1 / 0.955 is the other measured point.
    """
    if t + 1 >= len(trace):
        return [set() for _ in trace[0]]
    out = []
    for truth in trace[t + 1]:
        w = min(width, len(truth))
        picked = set()
        pool = list(truth)
        for _ in range(w):
            if rng.random() < precision and pool:
                picked.add(pool.pop(rng.randrange(len(pool))))
            else:
                picked.add(rng.randrange(N_EXPERTS))
        out.append(picked)
    return out


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def replay(policy, trace, predictor=None, rng=None):
    """Returns (per_step_misses, total_misses)."""
    policy.init_layers(len(trace[0]))
    per_step = []
    total = 0
    for t in range(len(trace)):
        pred = predictor(t, rng) if predictor is not None else None
        m = policy.step(t, trace[t], pred)
        per_step.append(m)
        total += m
    return per_step, total


def summarize(label, per_step, extra=None):
    n = len(per_step)
    m = sum(per_step) / n
    row = {"policy": label, "misses": m, "tps": tps(m), "steps": n,
           "misses_warm": sum(per_step[1:]) / (n - 1)}
    if extra:
        row.update(extra)
    return row


def fmt(row, base_tps=10.76):
    return "%-48s %8.2f  %6.2f  %+7.1f%%" % (
        row["policy"], row["misses"], row["tps"], 100.0 * (row["tps"] / base_tps - 1.0))


def stochastic(policy, trace, width, precision, trials, seed):
    acc = [0.0] * len(trace)
    means = []
    for k in range(trials):
        rng = random.Random(seed + 7919 * k)
        per_step, tot = replay(policy, trace,
                               predictor=lambda t, r, w=width, p=precision: predict_width(trace, t, r, w, p),
                               rng=rng)
        for i, v in enumerate(per_step):
            acc[i] += v
        means.append(tot / len(trace))
    per_step = [v / trials for v in acc]
    return per_step, statistics.pstdev(means), means


def verify(args, trace, n_layers):
    """Reproduces every external number this evaluation is calibrated against."""
    ok = True

    # 1. per-step, bit-exact replay of every measured ledger on disk
    cases = [("moe-steps-dt42.log", 42, "lfu", 16), ("moe-steps-n28.log", 28, "lfu", 16),
             ("moe-steps-n40.log", 40, "lfu", 16), ("moe-steps-n53.log", 53, "lfu", 16),
             ("moe-steps-pol16.log", 53, "lfu", 16), ("moe-steps-pol256.log", 53, "lfu", 256),
             ("moe-steps-pol2048.log", 53, "lfu", 2048), ("moe-steps-pollru.log", 53, "lru", 0)]
    print("-- ledger replay (per-step, 199 steps each)")
    for name, cap, base, h in cases:
        path = name if "/" in name else "/mnt/WorkDisk/harness/multiturn-ctx/" + name
        exp = load_ledger(path)
        if not exp:
            print("   SKIP %s (missing)" % name)
            continue
        per_step, _ = replay(Policy(cap, base=base, halflife=h or 16), trace)
        worst = max(abs(a - b) for a, b in zip(per_step, exp))
        bad = sum(1 for a, b in zip(per_step, exp) if a != b)
        print("   %-24s cap=%-3d %-4s h=%-5d sim=%7.2f ledger=%7.2f  steps differing=%d (max %d)  %s"
              % (name, cap, base, h, sum(per_step) / len(per_step), sum(exp) / len(exp), bad, worst,
                 "EXACT" if bad == 0 else "MISMATCH"))
        ok = ok and bad == 0

    # 2. Belady against the offline table in the plan document
    print("-- Belady/MIN vs the plan document's offline table")
    want = {8: 305.1, 10: 281.6, 20: 214.5, 42: 147.8, 64: 115.8, 128: 77.9, 256: 67.2, 512: 67.1}
    for cap, w in want.items():
        got = belady(trace, cap)
        same = abs(got - w) < 0.06
        print("   C=%-3d got %7.2f  doc %7.2f  %s" % (cap, got, w, "OK" if same else "DIFFERS"))
        ok = ok and same

    # 3. LRU: the document's 242.1 is a textbook sequential LRU, not the engine's
    seq = Policy(42, base="lru", model="textbook")
    per_step, _ = replay(seq, trace)
    print("-- LRU models: textbook sequential %.2f (document: 242.1) vs engine-semantics %.2f"
          % (sum(per_step) / len(per_step), sum(replay(Policy(42, base="lru"), trace)[0]) / len(trace)))

    # 4. predictor model reproduces both measured statistics
    print("-- predictor model (width 8 / 0.8621 and width 1 / 0.955)")
    trace_rng = random.Random(999)
    for width, prec in ((8, 0.8621), (1, 0.955)):
        r_tot = c_tot = p_tot = n = 0
        for t in range(len(trace) - 1):
            P = predict_width(trace, t, trace_rng, width, prec)
            for l in range(len(trace[0])):
                truth = set(trace[t + 1][l])
                r_tot += len(P[l] & truth)
                p_tot += len(P[l])
                c_tot += len(P[l] & truth)
                n += len(truth)
        print("   width %d: recall %.4f  coverage %.4f  (measured 8: .8621/.6897, 1: .955/.0955)"
              % (width, r_tot / p_tot, c_tot / n))

    # 5. the two independent implementations of "protect the next step" agree
    p1 = Oracle(42, 1, base="lfu"); p1.build_dist(trace)
    a, _ = replay(p1, trace)
    b, _ = replay(Policy(42, base="lfu", halflife=16, protect="pred"),
                  trace, predictor=lambda t, r: predict_perfect(trace, t))
    print("-- horizon-1 oracle %.2f vs protect-next-step %.2f  %s"
          % (sum(a) / len(a), sum(b) / len(b), "AGREE" if abs(sum(a) - sum(b)) < 0.01 else "DIFFER"))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default=TRACE_DEFAULT)
    ap.add_argument("--ledger", default=LEDGER_DEFAULT)
    ap.add_argument("--cap", type=int, default=42)
    ap.add_argument("--model", default="engine", choices=("engine", "textbook"))
    ap.add_argument("--arms", default="core")
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20260915)
    ap.add_argument("--json", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    trace, n_layers = load_trace(args.trace)
    T = len(trace)
    cap, model = args.cap, args.model
    rows = []
    perfect = lambda t, r=None: predict_perfect(trace, t)
    if args.arms == "verify":
        ok = verify(args, trace, n_layers)
        print("VERIFY: %s" % ("ALL EXACT" if ok else "SOME CHECKS FAILED"))
        return rows

    def emit(row):
        rows.append(row)
        if not args.quiet:
            print(fmt(row), flush=True)
        return row

    def run_arm(label, policy, predictor=None, extra=None):
        per_step, _ = replay(policy, trace, predictor=predictor)
        return emit(summarize(label, per_step, extra))

    def run_pred(label, policy, width, precision, extra=None):
        per_step, sd, _ = stochastic(policy, trace, width, precision, args.trials, args.seed)
        e = {"trials": args.trials, "trial_sd": sd}
        if extra:
            e.update(extra)
        return emit(summarize(label, per_step, e))

    if not args.quiet:
        print("cap=%d layers=%d steps=%d model=%s trace=%s" % (cap, n_layers, T, model, args.trace))
        print("%-48s %8s  %6s  %7s" % ("policy", "miss/step", "t/s", "vs 10.76"))

    if args.ledger and T == len(load_ledger(args.ledger)):
        emit(summarize("shipped ledger, MEASURED on GPU (LFU-16)",
                       load_ledger(args.ledger), {"measured": True}))

    if args.arms in ("core", "all"):
        run_arm("LRU", Policy(cap, base="lru", model=model))
        emit(summarize("Belady/MIN offline optimum (model-independent)",
                       [belady(trace, cap)] * T))
        for h in (4, 8, 16):
            run_arm("LFU decay half-life %d (shipped law)" % h,
                    Policy(cap, base="lfu", halflife=h, model=model))

    if args.arms in ("all", "tune"):
        for h in (32, 64, 128, 256, 512, 100000):
            run_arm("LFU decay half-life %d" % h,
                    Policy(cap, base="lfu", halflife=h, model=model))

    if args.arms in ("core", "all"):
        # ---- (3) prediction-guided victim selection -------------------------
        run_arm("PRED perfect + LRU",
                Policy(cap, base="lru", protect="pred", model=model), predictor=perfect)
        run_arm("PRED perfect + LFU-16",
                Policy(cap, base="lfu", halflife=16, protect="pred", model=model), predictor=perfect)
        run_pred("PRED width8 r86.21% + LRU",
                 Policy(cap, base="lru", protect="pred", model=model), 8, 0.8621)
        run_pred("PRED width8 r86.21% + LFU-16",
                 Policy(cap, base="lfu", halflife=16, protect="pred", model=model), 8, 0.8621)
        run_pred("PRED width1 r95.5% + LRU",
                 Policy(cap, base="lru", protect="pred", model=model), 1, 0.955)
        run_pred("PRED width1 r95.5% + LFU-16",
                 Policy(cap, base="lfu", halflife=16, protect="pred", model=model), 1, 0.955)

        # ---- (4) recency-window protection ----------------------------------
        for k in (1, 2, 4):
            run_arm("RECENT last %d step(s) + LRU" % k,
                    Policy(cap, base="lru", protect="recent", protect_arg=k, model=model))
        for k in (1, 2, 4):
            run_arm("RECENT last %d step(s) + LFU-16" % k,
                    Policy(cap, base="lfu", halflife=16, protect="recent", protect_arg=k, model=model))

    if args.arms in ("all", "tune", "oracle"):
        # ---- how much is a perfect predictor worth, by horizon ---------------
        for H, tie in ((1, "lru"), (1, "lfu"), (2, "lfu"), (4, "lfu"),
                       (8, "lfu"), (16, "lfu"), (10 ** 6, "lfu")):
            pol = Oracle(cap, H, base=tie, halflife=16)
            pol.build_dist(trace)
            per_step, _ = replay(pol, trace)
            label = "ORACLE perfect horizon %s + %s tie-break" % (
                "inf (=Belady)" if H > 10 ** 5 else H, tie.upper())
            emit(summarize(label, per_step, {"horizon": H, "tiebreak": tie}))

    if args.arms in ("all", "tune"):
        # ---- extra families, hunting for the best implementable policy ------
        run_arm("LRU + admission (no colder newcomer)",
                Policy(cap, base="lru", admit=True, model=model))
        run_arm("LFU-16 + admission (no colder newcomer)",
                Policy(cap, base="lfu", halflife=16, admit=True, model=model))
        for h in (64, 256, 100000):
            run_arm("LFU h=%d + admission" % h,
                    Policy(cap, base="lfu", halflife=h, admit=True, model=model))
        for w in (4, 8, 16, 32, 64):
            run_arm("WINDOW last %d steps count" % w,
                    Policy(cap, base="win", halflife=w, model=model))
        for h in (8, 16, 32):
            run_arm("LFU-slot (counter reset on install) h=%d" % h,
                    Policy(cap, base="lfuslot", halflife=h, model=model))
        for h in (32, 64, 128, 256, 512, 100000):
            run_arm("LFU h=%d + RECENT last 1 step" % h,
                    Policy(cap, base="lfu", halflife=h, protect="recent", protect_arg=1, model=model))
        for h in (64, 256, 100000):
            run_arm("LFU h=%d + PRED perfect" % h,
                    Policy(cap, base="lfu", halflife=h, protect="pred", model=model), predictor=perfect)
            run_pred("LFU h=%d + PRED width8" % h,
                     Policy(cap, base="lfu", halflife=h, protect="pred", model=model), 8, 0.8621)
        run_arm("LFU h=256 + PRED perfect + RECENT1",
                Policy(cap, base="lfu", halflife=256, protect="pred+recent", protect_arg=1, model=model), predictor=perfect)
        run_arm("LFU h=256 + PRED perfect + admission",
                Policy(cap, base="lfu", halflife=256, protect="pred", admit=True, model=model), predictor=perfect)

    if args.arms in ("all", "tune", "best"):
        # ---- combinations around the shipped policy -------------------------
        run_pred("LFU-16 + RECENT1 + PRED width8",
                 Policy(cap, base="lfu", halflife=16, protect="pred+recent", protect_arg=1, model=model),
                 8, 0.8621)
        run_pred("LFU-16 + RECENT2 + PRED width8",
                 Policy(cap, base="lfu", halflife=16, protect="pred+recent", protect_arg=2, model=model),
                 8, 0.8621)
        run_arm("LFU-16 + RECENT1 + PRED perfect",
                Policy(cap, base="lfu", halflife=16, protect="pred+recent", protect_arg=1, model=model),
                predictor=perfect)
        run_arm("LFU-16 + admission + RECENT1",
                Policy(cap, base="lfu", halflife=16, protect="recent", protect_arg=1,
                       admit=True, model=model))
        run_arm("WINDOW last 16 + PRED perfect",
                Policy(cap, base="win", halflife=16, protect="pred", model=model), predictor=perfect)
        run_pred("WINDOW last 16 + PRED width8",
                 Policy(cap, base="win", halflife=16, protect="pred", model=model), 8, 0.8621)

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rows, fh, indent=1)
    if not args.quiet:
        print("(rows json: %s)" % args.json if args.json else "")
    return rows


if __name__ == "__main__":
    main()
