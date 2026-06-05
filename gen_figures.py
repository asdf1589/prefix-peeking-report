"""Generate Figure 1 (bar chart) and Figure 2 (line chart) for the report."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

COLORS = {
    'LRU':  '#4878CF',
    'PPE':  '#D65F5F',
    'PPS':  '#6ACC65',
    'Both': '#B47CC7',
}

# ─────────────────────────────────────────────
# Figure 1: Grouped bar chart — closed-loop throughput
# ─────────────────────────────────────────────

workloads = [
    ('Gemma-3-4B\narXiv QA',  {'LRU': 0.1203, 'PPE': 0.2253, 'PPS': 0.1747, 'Both': 0.2692}),
    ('Ministral-8B\narXiv QA', {'LRU': 0.1725, 'PPE': 0.1766, 'PPS': 0.2044, 'Both': 0.2193}),
    ('Llama-3.1-8B\narXiv QA', {'LRU': 0.0574, 'PPE': 0.0542, 'PPS': 0.0607, 'Both': 0.0613}),
    ('Llama-3.1-8B\nMMLL-pro\n(4k ctx)',  {'LRU': 12.47,  'PPE': 12.46,  'PPS': 12.50,  'Both': 12.43}),
]

policies = ['LRU', 'PPE', 'PPS', 'Both']
x = np.arange(len(workloads))
bar_w = 0.22
offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * bar_w

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8),
                         gridspec_kw={'width_ratios': [3, 1], 'wspace': 0.35})

# Left panel: first 3 workloads (low throughput, same scale)
ax = axes[0]
left_idxs = [0, 1, 2]
lru_vals = [workloads[i][1]['LRU'] for i in left_idxs]
for pi, pol in enumerate(policies):
    vals = [workloads[i][1][pol] for i in left_idxs]
    bars = ax.bar(np.arange(len(left_idxs)) + offsets[pi], vals, bar_w,
                  color=COLORS[pol], label=pol, zorder=3)
    for bar, v, lru_v in zip(bars, vals, lru_vals):
        if pol == 'LRU':
            continue  # baseline — no label
        pct = (v - lru_v) / lru_v * 100
        label = f'{pct:+.0f}%'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                label, ha='center', va='bottom', fontsize=5.5, color='#333',
                fontweight='bold')

ax.set_xticks(np.arange(len(left_idxs)))
ax.set_xticklabels([workloads[i][0] for i in left_idxs], fontsize=7.5)
ax.set_ylabel('Throughput (req/s)')
ax.set_ylim(0, 0.38)
ax.set_title('(a) Shared-prefix workloads', fontsize=8.5)
ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
ax.set_axisbelow(True)
ax.legend(loc='upper right', ncol=2, framealpha=0.85)

# Right panel: MMLU control (different scale)
ax2 = axes[1]
vals_mmlu = [workloads[3][1][pol] for pol in policies]
lru_mmlu = workloads[3][1]['LRU']
mmlu_positions = np.arange(len(policies)) * bar_w * 1.3
bars2 = ax2.bar(mmlu_positions, vals_mmlu, bar_w,
                color=[COLORS[p] for p in policies], zorder=3)
for bar, pol, v in zip(bars2, policies, vals_mmlu):
    if pol == 'LRU':
        continue
    pct = (v - lru_mmlu) / lru_mmlu * 100
    label = f'{pct:+.1f}%'
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             label, ha='center', va='bottom', fontsize=5.5, color='#333',
             fontweight='bold')
ax2.set_xticks(mmlu_positions)
ax2.set_xticklabels(policies, fontsize=7.5)
ax2.set_ylabel('Throughput (req/s)')
ax2.set_ylim(12.3, 12.65)
ax2.set_title('(b) No-prefix control\n(MMLU-pro)', fontsize=8.5)
ax2.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
ax2.set_axisbelow(True)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.2f}'))

fig.savefig('figure1_throughput.pdf')
fig.savefig('figure1_throughput.png')
print("Figure 1 saved.")

# ─────────────────────────────────────────────
# Figure 2: Line charts — rate sweep (LRU vs Both)
# Gemma-3-4B-IT, 3 subplots: req/s, TTFT, TPOT
# ─────────────────────────────────────────────

rates = [0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0,3.2,3.4,3.6,3.8,4.0]

lru_e2el  = [88.9,142.7,162.0,172.7,178.3,182.3,185.4,187.2,188.8,190.1,
             191.1,192.0,193.0,193.1,193.9,194.8,194.7,195.2,196.2,196.5]
both_e2el = [60.8,88.6,96.3,110.4,127.9,131.7,137.4,106.1,145.8,135.6,
             146.9,132.9,137.3,134.8,132.4,132.8,133.6,133.2,137.2,135.3]

lru_ttft  = [55.7,105.9,125.3,135.9,141.5,145.6,148.6,150.5,152.0,153.4,
             154.4,155.2,156.3,156.4,157.2,158.1,158.1,158.6,159.4,159.7]
both_ttft = [30.4,55.6,65.1,77.7,96.1,99.7,108.0,79.2,110.1,101.5,
             111.4,98.0,101.6,99.8,99.8,100.2,101.0,100.8,104.0,105.8]

lru_tpot  = [437.6,480.6,479.6,481.4,480.9,481.1,481.1,480.8,480.7,480.0,
             480.5,480.5,480.0,480.0,480.0,480.2,479.2,479.3,480.9,480.9]
both_tpot = [394.1,416.9,426.7,409.1,404.7,436.0,370.0,309.2,477.6,440.9,
             476.1,461.2,470.2,462.7,444.3,443.9,444.4,443.4,450.7,365.9]

fig2, axes2 = plt.subplots(1, 3, figsize=(7.0, 2.4), sharey=False)
fig2.subplots_adjust(wspace=0.38)

ls_lru  = dict(color='#4878CF', marker='o', markersize=3.5, linewidth=1.4, label='LRU')
ls_both = dict(color='#B47CC7', marker='s', markersize=3.5, linewidth=1.4, label='Both (PPE+PPS)')

# (a) E2E Latency
ax = axes2[0]
ax.plot(rates, lru_e2el,  **ls_lru)
ax.plot(rates, both_e2el, **ls_both)
ax.set_xlabel('Request rate (req/s)')
ax.set_ylabel('Mean E2E Latency (s)')
ax.set_title('(a) End-to-End Latency', fontsize=8.5)
ax.set_xlim(0, 4.2)
ax.yaxis.grid(True, linestyle='--', alpha=0.5)
ax.set_axisbelow(True)
ax.legend(fontsize=7, loc='lower right')

# (b) TTFT
ax = axes2[1]
ax.plot(rates, lru_ttft,  **ls_lru)
ax.plot(rates, both_ttft, **ls_both)
ax.set_xlabel('Request rate (req/s)')
ax.set_ylabel('Mean TTFT (s)')
ax.set_title('(b) Time to First Token', fontsize=8.5)
ax.set_xlim(0, 4.2)
ax.yaxis.grid(True, linestyle='--', alpha=0.5)
ax.set_axisbelow(True)

# (c) TPOT
ax = axes2[2]
ax.plot(rates, lru_tpot,  **ls_lru)
ax.plot(rates, both_tpot, **ls_both)
ax.set_xlabel('Request rate (req/s)')
ax.set_ylabel('Mean TPOT (ms)')
ax.set_title('(c) Time per Output Token', fontsize=8.5)
ax.set_xlim(0, 4.2)
ax.yaxis.grid(True, linestyle='--', alpha=0.5)
ax.set_axisbelow(True)

fig2.savefig('figure2_rate_sweep.pdf')
fig2.savefig('figure2_rate_sweep.png')
print("Figure 2 saved.")
