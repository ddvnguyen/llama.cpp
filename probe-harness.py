#!/usr/bin/env python3
"""Arm: --context-shift hybrid-state correctness (Qwen3.5/3.8 GDN hybrid).

Two-marker needle-in-haystack probe per
docs/arms/arm-context-shift-hybrid-correctness.md.

Usage:
  python3 probe-harness.py --cells A1 --concurrency 1 --server-log srv-A1.log --port 8080
  python3 probe-harness.py --cells A1 --concurrency 2 --server-log srv-A2.log --port 8080
  python3 probe-harness.py --cells CTRL --concurrency 1 --filler-cap 10 ...
"""

import argparse
import datetime
import json
import os
import re
import threading
import time
import urllib.request

# ------------------------------- markers -----------------------------------

# Session-1 cast (from the doc example). Session-2 cast is by construction
# disjoint: different animal, names, color, keeper — zero token overlap.

S1 = {
    "m1": {
        "plant": (
            "Here is a story to keep in mind. Once, there was a very small "
            "pig named Wilbur whose favorite color was chartreuse, and the "
            "pig lived with a man named Borzoi-san. Please remember this "
            "story."
        ),
        "facts": ["Wilbur", "chartreuse", "Borzoi-san"],
    },
    "m2": {
        "plant": (
            "Here is another story to keep in mind: a squirrel named "
            "Zurnif-8 lived with a beekeeper named Pavdeel and spoke only "
            "in a rare accent, Felarn. Please remember this story."
        ),
        "facts": ["Zurnif-8", "Pavdeel", "Felarn"],
    },
    "animal": "pig",
    "second_animal": "crab",
}

S2 = {
    "m1": {
        "plant": (
            "Here is a story to keep in mind. Once, there was a very tall "
            "heron named Plimblad whose favorite color was saffron, and the "
            "heron lived with a boatwright named Kestral. Please remember "
            "this story."
        ),
        "facts": ["Plimblad", "saffron", "Kestral"],
    },
    "m2": {
        "plant": (
            "Here is another story to keep in mind: a crab named Gromvex-3 "
            "lived with a ferryman named Undshade and spoke only in a rare "
            "accent, Birser. Please remember this story."
        ),
        "facts": ["Gromvex-3", "Undshade", "Birser"],
    },
    "animal": "heron",
    "second_animal": "squirrel",
}

CASTS = {"s1": S1, "s2": S2}

GROWTH_SUFFIX = (
    " Answer at length in full flowing prose: at least fourteen complete "
    "sentences, two paragraphs minimum."
)

GROWTH_FILLER = [
    "Continue writing a story about the sea. Three paragraphs.",
    "How many legs does a cat have, and why do they have that number on "
    "this planet? Keep a natural tone.",
    "Name five rivers famous for their width and explain why each has "
    "that reputation. One paragraph.",
    "What makes a bridge feel solid or unsafe from a pedestrian's "
    "intuition, not engineering? One paragraph.",
    "Name plausible-sounding villages on two coasts and briefly justify "
    "the feel of each name.",
    "When do street markets in large cities open and how does climate "
    "change that? Two sentences.",
    "Describe how bread smells at three distinct baking stages — "
    "specific, sensory.",
    "Predict one believable change in daily life five years out and keep "
    "the claim measured.",
    "One paragraph of dialogue between a tired bus driver and a regular "
    "passenger. Natural, no drama.",
    "Why do satellite-view maps look different colors over farmland vs "
    "city in the same season? Plain.",
    "Why do some metal pans ring when struck and others just thud? A "
    "simple explanation.",
    "Which gets dirtier faster: windows on a busy road or on a quiet "
    "garden wall? One paragraph.",
    "Name three machines that fail slowly with warning instead of "
    "suddenly, and what the warning looks like.",
    "Three sentences on why some words sound soft and others hard, with "
    "an example of each.",
    "What do sheep do in prolonged heavy rain and how do farmers account "
    "for it? Plain, brief.",
    "Name three things that feel cold to your touch though they are at "
    "the same temperature as you.",
    "Describe a living room that reads quietly wealthy without naming "
    "wealth or prices.",
    "Why do some gym floors squeak and others don't? A few sentences.",
    "Which household toolbox items wear out fastest, and what visible "
    "wear shows up first?",
    "Explain in plain language why rain can fall while the far horizon "
    "still looks sharp.",
    "Two-sentence comparison: how a pianist and a chess player each stay "
    "sharp off their main craft.",
    "You get an unfamiliar spice — how would you figure out its origin "
    "and pairings? Brief.",
    "Why do some painted walls look patchy in evening light but even in "
    "daylight? One concrete reason.",
    "What tasks become worse when scheduled rather than done "
    "spontaneously? One example each.",
    "Three features of old houses that feel annoying but exist for a "
    "reason — with the reason.",
    "Three visual signs a pot is a minute from boiling over, and why the "
    "signs appear then.",
    "Which street signs vary most across a city, and what confusion "
    "results? Concrete detail.",
    "Write a small believable change to office meeting scheduling and "
    "argue for it briefly.",
    "Three old office devices that do their job better than their modern "
    "replacements sometimes.",
    "Why do old barns lean, and how do you spot it at a glance? One "
    "paragraph.",
]

M1_PROBE_FMT = [
    "What was the {animal}'s favorite color? What was its name? And who "
    "did it live with?",
    "Quick recall from the first story: what color did the {animal} "
    "favor?",
    "Remind me — the {animal} from the opening story: give its name, "
    "favorite color, and the person it lived with.",
    "Who did the {animal} from the story live with? And what was the "
    "{animal}'s name?",
    "From the story at the start of our chat: what color did the "
    "{animal} want around it?",
]
M2_PROBE_FMT = [
    "What was the {animal2}'s name? Who did it live with? What rare "
    "accent did it speak in?",
    "Second story check: the {animal2} — who did it live with, and what "
    "accent did it have?",
    "Name the {animal2} from the second story and the person it lived "
    "with.",
    "Recall the later story: the {animal2} that spoke in a rare accent — "
    "name, accent, and keeper.",
    "The keeper in the later story: what {animal2} was with them and "
    "what language quirk did it have?",
]

P3_OPEN_Q = [
    "Describe the pig's farm in detail.",
    "Write a short essay on why desert night skies feel different from "
    "city night skies.",
    "Invent a small believable change to how offices schedule meetings "
    "and argue for it briefly.",
]

# Marker-injection probability, at high temp these surface only if model
# reproduces them fromression state; they rise only in cross-cast leakage.
S1_ALL_FACTS = S1["m1"]["facts"] + S1["m2"]["facts"]
S2_ALL_FACTS = S2["m1"]["facts"] + S2["m2"]["facts"]


def now():
    return datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def chat(port, messages, temp, max_tokens, timeout=900):
    url = f"http://localhost:{port}/v1/chat/completions"
    payload = {
        "messages": messages,
        "temperature": temp,
        "max_tokens": max_tokens,
        "stream": False,
    }
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.loads(r.read())
            return {
                "ok": True,
                "content": out["choices"][0]["message"]["content"],
                "wall_s": round(time.time() - t0, 2),
                "prompt_tokens": out.get("usage", {}).get("prompt_tokens"),
                "completion_tokens": out.get("usage", {}).get(
                    "completion_tokens"
                ),
            }
    except Exception as e:
        return {
            "ok": False,
            "error": repr(e),
            "wall_s": round(time.time() - t0, 2),
        }


class ServerLogWatcher:
    """Watches the server log for shift lines and abort signatures."""

    SHIFT_RE = re.compile(r"slot context shift, n_keep = (\d+), n_left = (\d+)")
    ABORT_SIGS = (
        "GGML_ABORT",
        "Abort trap",
        "Segmentation fault",
        "does not support K-shift",
    )

    def __init__(self, path):
        self.path = path

    def refresh(self):
        self.shift_count = 0
        self.shift_events = []
        self.aborts = []
        if not os.path.exists(self.path):
            return 0
        with open(self.path, errors="replace") as f:
            for line in f:
                m = self.SHIFT_RE.search(line)
                if m:
                    self.shift_count += 1
                    self.shift_events.append(
                        (now(), int(m.group(1)), int(m.group(2)))
                    )
                for s in self.ABORT_SIGS:
                    if s in line:
                        self.aborts.append(line.strip()[:200])
                        break
        return self.shift_count

    def has_abort(self):
        return bool(self.aborts)


class Sink:
    """Thread-safe turn recorder."""

    def __init__(self):
        self.lock = threading.Lock()
        self.turns = []

    def add(self, rec):
        with self.lock:
            self.turns.append(rec)


def run_session(cid, port, watcher, barrier, sink, filler_cap, tag):
    cast = CASTS[cid]
    other = CASTS["s1" if cid == "s2" else "s2"]
    msgs = [
        {
            "role": "system",
            "content": "You are a helpful, plain-writing assistant.",
        }
    ]
    m1 = cast["m1"]
    m2 = cast["m2"]

    def fire(user_content, temp=0.8, max_tokens=320):
        msgs.append({"role": "user", "content": user_content})
        if barrier is not None:
            # phases align so both sessions submit each turn together
            barrier.wait()
        r = chat(port, msgs, temp, max_tokens)
        count = 0
        if r.get("ok") and r.get("content"):
            msgs.append({"role": "assistant", "content": r["content"]})
            count = r["completion_tokens"]
        rec = {
            "cid": cid,
            "t": now(),
            "user": user_content[:80],
            "ok": r.get("ok", False),
            "wall_s": r.get("wall_s"),
            "err": r.get("error"),
            "content": r.get("content"),
            "prompt_tokens": r.get("prompt_tokens"),
            "completion_tokens": count,
        }
        sink.add(rec)
        watcher.refresh()
        return r

    # turn accounting
    turnlog = []   # (kind, prompt_tokens)

    def log(kind, r):
        turnlog.append({"kind": kind, "ptok": r.get("prompt_tokens")})

    def fire_and_log(kind, content, temp=0.8, max_tokens=320):
        r = fire(content, temp, max_tokens)
        log(kind, r)
        return r

    # ---------- phase 1: M1 plant (very start), filler until shift 1 ----------
    r = fire(m1["plant"], 0)
    log("M1-plant", r)
    base = watcher.shift_count
    first_shift_at = None
    for i in range(filler_cap):
        r = fire(GROWTH_FILLER[i % len(GROWTH_FILLER)] + GROWTH_SUFFIX, 0.8, 900)
        log(f"growth-{i}", r)
        if not r.get("ok"):
            return {"cid": cid, "fatal": f"chat error phase1: {r.get('error')}"}
        n = watcher.refresh()
        if n > base:
            first_shift_at = i
            break

    if first_shift_at is None:
        return {
            "cid": cid,
            "fatal": "no shift fired within filler cap "
            f"({filler_cap} turns); raise filler turns or n_ctx ratio",
        }

    # ---------- phase 2: M2 plant immediately after shift 1 ----------
    r = fire(m2["plant"], 0)
    log("M2-plant", r)
    m2_plant_p = r.get("prompt_tokens")

    # ---------- phase 3: filler until shift 2 ----------
    base = watcher.shift_count
    second_shift_at = None
    for i in range(filler_cap):
        r = fire(GROWTH_FILLER[(i + 7) % len(GROWTH_FILLER)] + GROWTH_SUFFIX, 0.8, 900)
        log(f"growth2-{i}", r)
        if not r.get("ok"):
            return {"cid": cid, "fatal": f"chat error phase3: {r.get('error')}"}
        n = watcher.refresh()
        if n > base:
            second_shift_at = i
            break

    # ---------- phase 4: probes ----------
    m1_out = []
    for i, fmt in enumerate(M1_PROBE_FMT):
        r = fire(fmt.format(animal=cast["animal"]), 0, 256)
        c = (r.get("content") or "")
        m1_out.append({"i": i, "hits": [f in c for f in m1["facts"]], "out": c})
        log(f"M1-probe-{i}", r)

    m2_out = []
    m2_p4_pair = []
    for i, fmt in enumerate(M2_PROBE_FMT):
        r = fire(fmt.format(animal2=cast["second_animal"]), 0, 256)
        c = (r.get("content") or "")
        m2_out.append({"i": i, "hits": [f in c for f in m2["facts"]], "out": c})
        log(f"M2-probe-{i}", r)
        if i == 2:  # P4: this exact probe (verbatim, temp0) re-sent once more
            r2 = fire(fmt.format(animal2=cast["second_animal"]), 0, 256)
            m2_p4_pair = [(r.get("content") or ""), (r2.get("content") or "")]
            log("P4-repeat", r2)

    p3_out = []
    for i, q in enumerate(P3_OPEN_Q):
        r = fire(q, 0.8, 512)
        p3_out.append({"i": i, "out": r.get("content"), "ok": r.get("ok")})
        log(f"P3-{i}", r)

    # leakage probes: any cross-cast (the OTHER session's markers) in replies
    leak_out = []
    leak_qs = [
        "Aside from the stories we've discussed so far, were there any "
        "other animals or characters mentioned in our conversation at "
        "all? List them.",
        "From anything earlier in our chat, name three colors and what "
        "each one On the Mark describes.",
    ]
    for i, q in enumerate(leak_qs):
        r = fire(q, 0, 256)
        c = (r.get("content") or "")
        cross_hits = [f in c for f in other["m1"]["facts"] + other["m2"]["facts"]]
        leak_out.append({"i": i, "cross_hits": cross_hits, "out": c})
        log(f"LEAK-probe-{i}", r)

    return {
        "cid": cid,
        "label_m1": m1["plant"][:60],
        "first_shift_at_turn": first_shift_at,
        "second_shift_at_turn": second_shift_at,
        "m2_planted_at_prompt_tokens": m2_plant_p,
        "m1_probes": m1_out,
        "m2_probes": m2_out,
        "p4_pair": m2_p4_pair,
        "p3": p3_out,
        "leak": leak_out,
        "turnlog": turnlog,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="A1")
    ap.add_argument("--concurrency", type=int, default=1, choices=(1, 2))
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument(
        "--server-log", required=True, help="llama-server stdout log to watch"
    )
    ap.add_argument(
        "--out",
        default=None,
        help="result JSON path (default wr-probes/<cells>-<ts>.json)",
    )
    ap.add_argument("--filler-cap", type=int, default=60)
    args = ap.parse_args()

    watcher = ServerLogWatcher(args.server_log)
    watcher.refresh()
    n_sessions = args.concurrency
    sink = Sink()
    barrier = (
        threading.Barrier(n_sessions, timeout=600) if n_sessions == 2 else None
    )
    ts = datetime.datetime.now().strftime("%H%M%S")
    out_path = (
        args.out
        or os.path.join(
            os.path.dirname(args.server_log)
            or ".",
            f"probe-{args.cells}-{ts}.json",
        )
    )

    sessions = ["s1"] if n_sessions == 1 else ["s1", "s2"]
    results = []
    t_start = time.time()
    starts = {}

    def worker(cid):
        starts[cid] = time.time()
        try:
            r = run_session(
                cid, args.port, watcher, barrier, sink, args.filler_cap,
                args.cells,
            )
        except Exception as e:
            r = {"cid": cid, "fatal": f"harness exception: {e!r}"}
        results.append(r)

    threads = [
        threading.Thread(target=worker, args=(c,), daemon=False)
        for c in sessions
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    wall = round(time.time() - t_start, 1)

    # ---- per-session scoring (per doc scoring table) ----
    scored = {"cells": args.cells, "wall_s": wall, "n_sessions": n_sessions}
    ab = watcher.refresh()
    scored["total_shift_events"] = ab
    scored["aborts"] = watcher.aborts

    for r in results:
        s = {"cid": r.get("cid"), "fatal": r.get("fatal")}
        if not r.get("fatal"):
            m1_any_hit = any(all(p["hits"]) for p in r["m1_probes"])
            m1_confident_hits = sum(all(p["hits"]) for p in r["m1_probes"])
            m2_confident_hits = sum(all(p["hits"]) for p in r["m2_probes"])
            p4_identical = len(set(r["p4_pair"])) == 1
            leak_hits = sum(sum(p["cross_hits"]) for p in r["leak"]) > 0
            p3_status = all(
                p["ok"] and p["out"] and len(p["out"]) > 40 for p in r["p3"]
            )
            s.update(
                {
                    "first_shift_at_turn": r["first_shift_at_turn"],
                    "second_shift_at_turn": r["second_shift_at_turn"],
                    "m1_confident_hits_all3": m1_confident_hits,
                    "m2_confident_hits_all3": m2_confident_hits,
                    "m1_any_full_hit": m1_any_hit,
                    "p4_identical": p4_identical,
                    "p4_pair": r["p4_pair"],
                    "leak_cross_hits_total": sum(
                        sum(p["cross_hits"]) for p in r["leak"]
                    ),
                    "leak_leaky": leak_hits,
                    "p3_all_substantial": p3_status,
                    "m2_probe_hits": [p["hits"] for p in r["m2_probes"]],
                    "m1_probe_hits": [p["hits"] for p in r["m1_probes"]],
                }
            )
        scored.setdefault("sessions", []).append(
            s if r.get("fatal") else {**r, **s}
        )

    with open(out_path, "w") as f:
        json.dump({"summary": scored, "raw_turns": sink.turns}, f, indent=1)
    print("WROTE", out_path)
    print(json.dumps(scored, indent=2)[:3000])


if __name__ == "__main__":
    main()
