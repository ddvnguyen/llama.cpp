#!/usr/bin/env python3
"""join_steps.py - join the per-step MoE miss ledger with the per-token arrival trace.

Usage: join_steps.py <tag> [bandwidth_gbs]

Reads  <script_dir>/moe-steps-<tag>.log   server stderr, one "moe-step:" line per decode step
       <script_dir>/perftok-<tag>.tsv     per-token arrival trace written by stream.sh
Writes <script_dir>/steptrace-<tag>.tsv   joined per-step table

Wire format parsed from the ledger (only lines containing "moe-step:" are examined,
so timestamps/levels/other server logs may be interleaved freely):

  moe-step: step=<n> need=<u> resident=<u> misses=<u> miss_mib=<f>

Model column (what the misses should cost on the link):

  gather_ms = miss_mib * 1048576 / (bandwidth_gbs * 1e9) * 1000

Stdlib only. All output is ASCII. Paths resolve relative to this script's directory,
so the script may be invoked from any cwd.
"""

import json
import math
import os
import re
import sys

MIB = 1048576.0
DEFAULT_BW_GBS = 5.94
USAGE = "usage: join_steps.py <tag> [bandwidth_gbs]   (bandwidth_gbs default %.2f)\n" % DEFAULT_BW_GBS

KV_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([-+0-9.eE]+)")
INT_RE = re.compile(r"^[-+]?[0-9]+$")

# Offsets probed when cross-checking the alignment rule against the data.
OFFSET_SCAN = (-3, -2, -1, 0, 1, 2, 3)


def fail(msg, code=2):
    """Exit with a one-line error instead of a traceback."""
    sys.stderr.write("ERROR: %s\n" % msg)
    sys.exit(code)


def as_float(text, what, where):
    try:
        return float(text)
    except (TypeError, ValueError):
        fail("non-numeric %s %r in %s" % (what, text, where))


def as_int(text, what, where):
    if INT_RE.match(text or ""):
        return int(text)
    f = as_float(text, what, where)
    return int(round(f))


def percentile(sorted_vals, p):
    """Nearest-rank percentile; same convention as stream.sh (index int(n*p))."""
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    k = int(n * p)
    if k >= n:
        k = n - 1
    return sorted_vals[k]


def pearson(xs, ys):
    """Pearson correlation; None when undefined (n < 2 or a series has zero variance)."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0.0 or syy <= 0.0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / math.sqrt(sxx * syy)


def load_ledger(path):
    """Parse moe-step lines; returns (steps, info). Duplicate steps keep the first."""
    steps = []
    seen = set()
    duplicates = 0
    malformed = 0
    total_lines = 0
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            total_lines += 1
            pos = line.find("moe-step:")
            if pos < 0:
                continue
            payload = line[pos + len("moe-step:"):]
            kv = dict(KV_RE.findall(payload))
            if "step" not in kv or "misses" not in kv or "miss_mib" not in kv:
                malformed += 1
                continue
            step = as_int(kv["step"], "step", path)
            if step in seen:
                duplicates += 1
                continue
            seen.add(step)
            steps.append({
                "step": step,
                "need": as_int(kv["need"], "need", path) if "need" in kv else 0,
                "resident": as_int(kv["resident"], "resident", path) if "resident" in kv else 0,
                "misses": as_int(kv["misses"], "misses", path),
                "miss_mib": as_float(kv["miss_mib"], "miss_mib", path),
            })
    if not steps:
        fail("no 'moe-step:' lines found in %s (%d lines scanned)" % (path, total_lines))
    steps.sort(key=lambda s: s["step"])
    lo, hi = steps[0]["step"], steps[-1]["step"]
    missing = sorted(set(range(lo, hi + 1)) - seen)
    info = {
        "lines_scanned": total_lines,
        "duplicates": duplicates,
        "malformed": malformed,
        "first_step": lo,
        "last_step": hi,
        "missing": missing,
    }
    return steps, info


def load_trace(path):
    """Parse the per-token trace; returns (gaps, info).

    gaps[i] is the arrival delta (ms) of the i-th DECODE step: the trace chunk that
    carried a new token, i.e. the interval between the arrival of token i+1 and the
    arrival of token i+2. The prefill chunk (no server timings) and any trailing
    flush chunk (no predicted_n increase) are not decode steps and are excluded.
    """
    with open(path, "r", errors="replace") as fh:
        raw = fh.read().split("\n")
    lines = [ln for ln in raw if ln.strip()]
    if len(lines) < 2:
        fail("per-token trace has no data rows: %s" % path)
    header = lines[0].split("\t")
    col = {}
    for i, name in enumerate(header):
        col[name.strip()] = i
    if "delta_ms" not in col:
        fail("per-token trace has no 'delta_ms' column: %s (header: %s)"
             % (path, " | ".join(header)))

    i_delta = col["delta_ms"]
    i_st = col.get("server_timings")
    rows = []
    malformed = 0
    for line in lines[1:]:
        fields = line.split("\t")
        if len(fields) <= i_delta:
            malformed += 1
            continue
        delta = as_float(fields[i_delta], "delta_ms", path)
        pn = None
        if i_st is not None and len(fields) > i_st and fields[i_st].strip():
            try:
                parsed = json.loads(fields[i_st])
            except ValueError:
                parsed = None
            if isinstance(parsed, dict):
                pn = parsed.get("predicted_n")
                if not isinstance(pn, int):
                    try:
                        pn = int(pn)
                    except (TypeError, ValueError):
                        pn = None
        rows.append({"delta": delta, "pn": pn})
    if not rows:
        fail("no usable data rows in %s (%d malformed)" % (path, malformed))

    # Preferred rule: a decode step is a chunk whose predicted_n advanced.
    gaps = []
    prev_pn = None
    for r in rows:
        if r["pn"] is not None and prev_pn is not None and r["pn"] > prev_pn:
            gaps.append({"delta": r["delta"], "pn": r["pn"]})
        if r["pn"] is not None:
            prev_pn = r["pn"]

    basis = "predicted_n increase"
    dropped_tail = 0
    dropped_head = 0
    if not gaps:
        # Fallback: the trace carries no server timings. Skip the prefill chunk and
        # treat every remaining row as one decode step, dropping sub-ms head/tail
        # flush chunks (chunking artifacts, not decode steps).
        basis = "positional fallback (no predicted_n in trace)"
        body = rows[1:]
        ref = body if body else rows
        limit = min(1.0, 0.25 * sorted(r["delta"] for r in ref)[len(ref) // 2])
        while len(body) > 1 and body[-1]["delta"] < limit:
            body.pop()
            dropped_tail += 1
        while len(body) > 1 and body[0]["delta"] < limit:
            body.pop(0)
            dropped_head += 1
        gaps = [{"delta": r["delta"], "pn": None} for r in body]

    info = {
        "header": header,
        "data_rows": len(rows),
        "malformed": malformed,
        "basis": basis,
        "dropped_tail": dropped_tail,
        "dropped_head": dropped_head,
        "first_pn": gaps[0]["pn"] if gaps else None,
        "last_pn": gaps[-1]["pn"] if gaps else None,
    }
    return gaps, info


def gather_ms(miss_mib, bw_gbs):
    return miss_mib * MIB / (bw_gbs * 1e9) * 1000.0


def pair(steps, gaps, offset):
    """Join ledger steps to trace gaps by trace_index = step + offset."""
    joined = []
    for s in steps:
        j = s["step"] + offset
        if 0 <= j < len(gaps):
            g = gaps[j]
            gm = gather_ms(s["miss_mib"], BW)
            joined.append({
                "step": s["step"],
                "misses": s["misses"],
                "miss_mib": s["miss_mib"],
                "gather_ms": gm,
                "observed_ms": g["delta"],
                "residual_ms": g["delta"] - gm,
            })
    return joined


def stats_line(vals, digits, ints=False):
    sv = sorted(vals)
    if not sv:
        return "n=0"
    if ints:
        return ("min=%d mean=%.1f p50=%d p95=%d max=%d"
                % (sv[0], sum(sv) / len(sv), percentile(sv, 0.50), percentile(sv, 0.95), sv[-1]))
    fmt = "min=%%.%df mean=%%.%df p50=%%.%df p95=%%.%df max=%%.%df" % (digits, digits, digits, digits, digits)
    return fmt % (sv[0], sum(sv) / len(sv), percentile(sv, 0.50), percentile(sv, 0.95), sv[-1])


def main(argv):
    if len(argv) >= 2 and argv[1] in ("-h", "--help"):
        sys.stdout.write(USAGE)
        return 0
    if len(argv) != 2 and len(argv) != 3:
        sys.stderr.write(USAGE)
        return 2

    tag = argv[1]
    global BW
    BW = DEFAULT_BW_GBS
    bw_source = "default"
    if len(argv) == 3:
        BW = as_float(argv[2], "bandwidth_gbs", "command line")
        bw_source = "argv"
    if not BW > 0.0:
        fail("bandwidth_gbs must be > 0, got %s" % argv[2])

    here = os.path.dirname(os.path.realpath(os.path.abspath(__file__)))
    led_path = os.path.join(here, "moe-steps-%s.log" % tag)
    tok_path = os.path.join(here, "perftok-%s.tsv" % tag)
    out_path = os.path.join(here, "steptrace-%s.tsv" % tag)

    if not os.path.isfile(led_path):
        hint = ""
        alt = os.path.join(here, "server-%s.log" % tag)
        if os.path.isfile(alt):
            hint = ("\n       note: %s exists - if the emitter wrote its stderr there, copy or symlink it to %s"
                    % (alt, led_path))
        fail("ledger not found: %s%s" % (led_path, hint))
    if not os.path.isfile(tok_path):
        fail("per-token trace not found: %s" % tok_path)

    try:
        steps, linfo = load_ledger(led_path)
        gaps, tinfo = load_trace(tok_path)
    except OSError as exc:
        fail("cannot read input file: %s" % exc)

    # Alignment rule: the first ledger step owns the first trace gap, so a ledger
    # numbered 0-based (0..n-1) pairs step s with gap s, and a 1-based ledger
    # (1..n) pairs step s with gap s-1. Both give offset = -first_step.
    offset = -linfo["first_step"]
    joined = pair(steps, gaps, offset)

    # Cross-check the rule against the data: which offset best explains observed ms?
    scan = []
    for off in OFFSET_SCAN:
        pj = pair(steps, gaps, off)
        if len(pj) < 8:
            continue
        r = pearson([x["misses"] for x in pj], [x["observed_ms"] for x in pj])
        scan.append((off, len(pj), r))
    used_r = pearson([x["misses"] for x in joined], [x["observed_ms"] for x in joined]) if len(joined) >= 8 else None
    best = None
    for off, n, r in scan:
        if r is None:
            continue
        if best is None or abs(r) > abs(best[2]):
            best = (off, n, r)

    out = sys.stdout
    out.write("=== join_steps: tag=%s ===\n" % tag)
    out.write("dir       : %s\n" % here)
    out.write("ledger    : %s\n" % os.path.basename(led_path))
    out.write("trace     : %s\n" % os.path.basename(tok_path))
    out.write("bandwidth : %.4f GB/s (%s)\n" % (BW, bw_source))
    out.write("\n")

    out.write("-- alignment --\n")
    out.write("ledger steps        : %d (step %d..%d)\n"
              % (len(steps), linfo["first_step"], linfo["last_step"]))
    out.write("trace decode gaps   : %d (basis: %s%s)\n"
              % (len(gaps), tinfo["basis"],
                 ", predicted_n %s..%s" % (tinfo["first_pn"], tinfo["last_pn"]) if tinfo["first_pn"] is not None else ""))
    out.write("trace data rows     : %d (columns: %s)\n"
              % (tinfo["data_rows"], " ".join(tinfo["header"])))
    out.write("lengths differ      : %s\n" % ("YES" if len(steps) != len(gaps) else "no"))
    out.write("offset used         : %d  (trace_index = step + offset = step %+d)\n" % (offset, offset))
    if tinfo["first_pn"] is not None:
        out.write("mapping             : step -> trace row predicted_n = step %+d (the decode pass that produced that token)\n"
                  % (offset + 2))
    out.write("overlap used        : %d paired steps\n" % len(joined))
    if len(steps) > len(joined):
        out.write("ledger steps dropped: %d (outside the trace gap range)\n" % (len(steps) - len(joined)))
    if len(gaps) > len(joined):
        out.write("trace gaps unused   : %d (no ledger step mapped to them)\n" % (len(gaps) - len(joined)))
    if linfo["duplicates"]:
        out.write("duplicate step lines: %d (kept first occurrence)\n" % linfo["duplicates"])
    if linfo["malformed"]:
        out.write("malformed step lines: %d (skipped)\n" % linfo["malformed"])
    if linfo["missing"]:
        shown = " ".join(str(x) for x in linfo["missing"][:8])
        out.write("missing step numbers: %d (%s%s)\n"
                  % (len(linfo["missing"]), shown, " ..." if len(linfo["missing"]) > 8 else ""))
    if tinfo["malformed"]:
        out.write("malformed trace rows: %d (skipped)\n" % tinfo["malformed"])
    if tinfo["dropped_tail"] or tinfo["dropped_head"]:
        out.write("flush rows dropped  : %d head, %d tail (sub-ms arrival deltas, not decode steps)\n"
                  % (tinfo["dropped_head"], tinfo["dropped_tail"]))
    if scan:
        out.write("offset scan (r = corr(misses, observed_ms)):\n")
        for off, n, r in scan:
            marks = []
            if off == offset:
                marks.append("used")
            if best is not None and off == best[0]:
                marks.append("best")
            out.write("  offset %+d n=%3d r=%s %s\n"
                      % (off, n, "   n/a " if r is None else "%+.4f" % r, ",".join(marks)))
        if best is not None and best[0] != offset and used_r is not None and abs(best[2]) > abs(used_r) + 0.05:
            out.write("WARNING: offset %+d fits the data better than the rule-derived offset %+d "
                      "(%+.4f vs %+.4f) - check the emitter's step numbering\n"
                      % (best[0], offset, best[2], used_r))
    out.write("\n")

    miss_vals = [s["misses"] for s in steps]
    mib_vals = [s["miss_mib"] for s in steps]
    out.write("-- ledger per-step miss statistics (%d steps) --\n" % len(steps))
    out.write("misses   : %s\n" % stats_line(miss_vals, 0, ints=True))
    out.write("miss_mib : %s\n" % stats_line(mib_vals, 2))
    out.write("totals   : need=%d resident=%d misses=%d miss_mib=%.2f\n"
              % (sum(s["need"] for s in steps), sum(s["resident"] for s in steps),
                 sum(miss_vals), sum(mib_vals)))
    out.write("\n")

    out.write("-- model: gather_ms = miss_mib * 1048576 / (%.4f GB/s * 1e9) * 1000 --\n" % BW)
    gv = [j["gather_ms"] for j in joined]
    out.write("per-step : %s\n" % stats_line(gv, 3))
    out.write("total    : %.3f ms over %d steps\n" % (sum(gv), len(gv)))
    out.write("\n")

    out.write("-- comparison (aligned steps) --\n")
    ov = [j["observed_ms"] for j in joined]
    mv = [j["misses"] for j in joined]
    r = pearson(mv, ov)
    sum_mib = sum(j["miss_mib"] for j in joined)
    sum_obs = sum(ov)
    out.write("pearson(misses, observed_ms) : %s\n" % ("n/a" if r is None else "%+.4f" % r))
    out.write("mean observed_ms             : %.3f  (min %.2f max %.2f)\n"
              % (sum(ov) / len(ov) if ov else 0.0, min(ov) if ov else 0.0, max(ov) if ov else 0.0))
    out.write("mean gather_ms               : %.3f\n" % (sum(gv) / len(gv) if gv else 0.0))
    out.write("mean residual_ms             : %.3f  (observed - gather)\n"
              % (sum(j["residual_ms"] for j in joined) / len(joined) if joined else 0.0))
    out.write("sum miss_mib                 : %.2f\n" % sum_mib)
    out.write("sum observed_ms              : %.3f\n" % sum_obs)
    if sum_obs > 0.0:
        implied_gbs = sum_mib * MIB / (sum_obs / 1000.0) / 1e9
        out.write("implied bandwidth            : %.4f GB/s  (sum(miss_mib)*1048576 / (sum(observed_ms)/1000) / 1e9)\n"
                  % implied_gbs)
    else:
        out.write("implied bandwidth            : n/a (sum observed_ms is 0)\n")
    out.write("\n")

    with open(out_path, "w") as fh:
        fh.write("step\tmisses\tmiss_mib\tgather_ms\tobserved_ms\tresidual_ms\n")
        for j in joined:
            fh.write("%d\t%d\t%.2f\t%.3f\t%.2f\t%.3f\n"
                     % (j["step"], j["misses"], j["miss_mib"], j["gather_ms"],
                        j["observed_ms"], j["residual_ms"]))

    head = 40
    out.write("-- per-step table (first %d of %d rows) --\n" % (min(head, len(joined)), len(joined)))
    out.write("%6s %8s %9s %10s %11s %11s\n"
              % ("step", "misses", "miss_mib", "gather_ms", "observed_ms", "residual_ms"))
    for j in joined[:head]:
        out.write("%6d %8d %9.2f %10.3f %11.2f %11.3f\n"
                  % (j["step"], j["misses"], j["miss_mib"], j["gather_ms"],
                     j["observed_ms"], j["residual_ms"]))
    if len(joined) > head:
        out.write("... %d more rows\n" % (len(joined) - head))
    out.write("full table (%d rows) written to %s\n" % (len(joined), out_path))
    return 0


BW = DEFAULT_BW_GBS

if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except KeyboardInterrupt:
        sys.exit(130)
