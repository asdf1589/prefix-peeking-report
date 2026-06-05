#!/usr/bin/env python3
"""Generate Figure 3: Prefix cache hit rate over time (all 4 policies).

Data source: vLLM server logs from Gemma-3-4B-IT closed-loop arXiv QA run.
All four conditions use stock vLLM (--disable-hybrid-kv-cache-manager),
matching the LRU baseline definition used in Table 2.  Explicit PPE env var.

  LRU  : hitrate_gemma3_4b.lru.server.log   (PPE=0, PPS=off)
  PPE  : hitrate_gemma3_4b.ppe.server.log   (PPE=1, PPS=off)
  PPS  : hitrate_gemma3_4b.pps.server.log   (PPE=0, PPS=on)
  Both : hitrate_gemma3_4b.both.server.log  (PPE=1, PPS=on)

Each data point is the 10-second rolling average reported by vLLM's logger.
"""

import re
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

LOG_DIR = Path(__file__).parent.parent / "Jenga-SOSP25-AE" / "logs" / "e2e"

# Ordered for legend clarity; colors follow a colourblind-friendly palette.
LOG_FILES = {
    "LRU":        ("hitrate_gemma3_4b.lru.server.log",  "#d62728", "--",  "o"),
    "PPE (Ours)": ("hitrate_gemma3_4b.ppe.server.log",  "#ff7f0e", "-.",  "^"),
    "PPS (Ours)": ("hitrate_gemma3_4b.pps.server.log",  "#1f77b4", ":",   "s"),
    "Both (Ours)":("hitrate_gemma3_4b.both.server.log", "#2ca02c", "-",   "D"),
}

PATTERN = re.compile(
    r"INFO (\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*Prefix cache hit rate: ([0-9.]+)%"
)


def parse_log(path: Path):
    timestamps, rates = [], []
    if not path.exists():
        return [], []
    with open(path) as fh:
        for line in fh:
            m = PATTERN.search(line)
            if m:
                ts = datetime.strptime("2026-" + m.group(1), "%Y-%m-%d %H:%M:%S")
                timestamps.append(ts)
                rates.append(float(m.group(2)))
    if not timestamps:
        return [], []
    t0 = timestamps[0]
    elapsed = [(t - t0).total_seconds() for t in timestamps]
    return elapsed, rates


def main():
    fig, ax_t = plt.subplots(1, 1, figsize=(3.5, 2.4))

    for label, (fname, color, ls, marker) in LOG_FILES.items():
        path = LOG_DIR / fname
        elapsed, rates = parse_log(path)
        if not elapsed:
            print(f"WARNING: no data found in {path}")
            continue
        ax_t.plot(elapsed, rates,
                  label=label, color=color, linestyle=ls,
                  linewidth=1.4, marker=marker, ms=3.5, markevery=3)

    ax_t.set_ylabel("Prefix cache hit rate (%)", fontsize=9)
    ax_t.set_ylim(bottom=0)
    ax_t.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
    ax_t.tick_params(labelsize=8)
    ax_t.grid(axis="y", linestyle=":", alpha=0.5)

    ax_t.set_xlabel("Elapsed time (s)", fontsize=9)
    ax_t.legend(fontsize=7.5, loc="upper right", framealpha=0.85)

    fig.tight_layout(pad=0.5)
    out = Path(__file__).parent / "figure3_hit_rate.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")

    # Print summary statistics for paper text
    print("\n--- Summary (mean of steady-state values, ignoring first 30 s) ---")
    for label, (fname, *_) in LOG_FILES.items():
        elapsed, rates = parse_log(LOG_DIR / fname)
        if not elapsed:
            continue
        steady = [r for t, r in zip(elapsed, rates) if t > 30]
        if steady:
            print(f"  {label:15s}: mean={sum(steady)/len(steady):.1f}%  "
                  f"min={min(steady):.1f}%  max={max(steady):.1f}%")


if __name__ == "__main__":
    main()
