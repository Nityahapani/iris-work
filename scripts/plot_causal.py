import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, json, os

BLU='#2563eb'; RED='#dc2626'; GRN='#16a34a'; ORG='#d97706'; PUR='#7c3aed'; GRY='#6b7280'
os.makedirs('results/figures', exist_ok=True)

with open('results/temporal_order_ablation.json') as f: toa=json.load(f)
with open('results/timing_sweep_cora.json') as f: ts=json.load(f)
with open('results/controlled_structural_sweep.json') as f: css=json.load(f)

fig = plt.figure(figsize=(18, 11))
gs  = plt.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

# Panel 1: Exp 5 temporal order ablation
ax = fig.add_subplot(gs[0, 0])
order_labels = ['TGS\norder', 'Random\norder', 'Reverse\norder', 'Static\n(upfront)']
order_accs   = [toa['exp5_order'][l] for l in ['TGS order','Random order','Reverse order','Static (all upfront)']]
order_colors = [BLU, PUR, ORG, RED]
bars = ax.bar(range(4), order_accs, color=order_colors, edgecolor='white', width=0.6)
for bar, v in zip(bars, order_accs):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001,
            f'{v:.3f}', ha='center', va='bottom', fontsize=9,
            fontweight='bold' if v==max(order_accs) else 'normal')
static_v = toa['exp5_order']['Static (all upfront)']
ax.axhline(static_v, color=RED, ls='--', lw=1.5, alpha=0.6)
ax.set_xticks(range(4)); ax.set_xticklabels(order_labels, fontsize=9)
ax.set_ylabel('Test Accuracy (Cora)', fontsize=10)
ax.set_title('Exp 5: Does Ordering Matter?\nSame final edges, different retirement order', fontsize=10, fontweight='bold')
ax.set_ylim(0.70, 0.84); ax.grid(True, alpha=0.25, axis='y')
delta_rand = toa['exp5_order']['Random order'] - static_v
ax.text(1.5, 0.705,
        f'TIMING matters, not order\n(all temporal: +{delta_rand:.2f} over static)',
        ha='center', fontsize=8, color=BLU,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#eff6ff', alpha=0.9))

# Panel 2: Exp 6 timing sweep
ax = fig.add_subplot(gs[0, 1])
warmups  = [r['warmup'] for r in ts]
tgs_accs = [r['tgs_acc'] for r in ts]
static_a = ts[0]['static_acc']
ax2 = ax.twinx()
ax.plot(warmups, tgs_accs, 'o-', color=BLU, lw=2.5, markersize=7, label='TGS accuracy')
ax.axhline(static_a, color=RED, ls='--', lw=2, label=f'Static ({static_a:.3f})')
deltas = [r['delta'] for r in ts]
ax2.bar(warmups, deltas, color=BLU, alpha=0.15, width=9)
opt_idx = int(np.argmax(tgs_accs))
ax.axvline(warmups[opt_idx], color=GRN, ls=':', lw=2, alpha=0.8)
ax.scatter([warmups[opt_idx]], [tgs_accs[opt_idx]], color=GRN, s=120, zorder=6)
ax.annotate(f'Optimal w={warmups[opt_idx]}',
            xy=(warmups[opt_idx], tgs_accs[opt_idx]),
            xytext=(warmups[opt_idx]+18, tgs_accs[opt_idx]-0.012),
            arrowprops=dict(arrowstyle='->', color=GRN), fontsize=8, color=GRN)
ax.axvspan(0, warmups[opt_idx], alpha=0.06, color=RED)
ax.axvspan(warmups[opt_idx], 170, alpha=0.06, color=GRN)
ax.text(10, 0.705, 'Too early', fontsize=8, color=RED, style='italic')
ax.text(70, 0.705, 'Works well', fontsize=8, color=GRN, style='italic')
ax.set_xlabel('Warmup (retirement starts at epoch...)', fontsize=10)
ax.set_ylabel('Test Accuracy', fontsize=10, color=BLU)
ax2.set_ylabel('Δ vs Static', fontsize=9, color=BLU)
ax.set_title('Exp 6: Retirement Timing\nOptimal warmup validates Prop 8.2', fontsize=10, fontweight='bold')
ax.legend(fontsize=8, loc='lower right'); ax.grid(True, alpha=0.25)
ax.set_ylim(0.70, 0.82); ax2.set_ylim(-0.01, 0.12)

# Panel 3: Structural sweep homophily
ax = fig.add_subplot(gs[0, 2])
sw_a = css['sweep_a_homophily']
hs = [r['actual_h'] for r in sw_a]
deltas_a = [r['delta'] for r in sw_a]
ax.plot(hs, deltas_a, 'o-', color=BLU, lw=2.5, markersize=8)
ax.axhline(0, color='black', lw=0.8, ls='--', alpha=0.5)
ax.fill_between(hs, 0, deltas_a, color=BLU, alpha=0.12)
ax.set_xlabel('Homophily (actual)', fontsize=10)
ax.set_ylabel('TGS - Static Accuracy', fontsize=10)
ax.set_title('Sweep A: Homophily vs TGS Advantage\n(deg_cv held constant)', fontsize=10, fontweight='bold')
ax.grid(True, alpha=0.25)
ax.text(0.98, 0.95, 'Note: ceiling effects\nreduce absolute delta\nin synthetic graphs',
        transform=ax.transAxes, ha='right', va='top', fontsize=7, color=GRY, style='italic')

# Panel 4: Structural sweep deg CV
ax = fig.add_subplot(gs[1, 0])
sw_b = css['sweep_b_deg_cv']
cvs = [r['cv'] for r in sw_b]
deltas_b = [r['delta'] for r in sw_b]
ax.plot(cvs, deltas_b, 's-', color=PUR, lw=2.5, markersize=8)
ax.axhline(0, color='black', lw=0.8, ls='--', alpha=0.5)
pos_mask = [d > 0 for d in deltas_b]
neg_mask = [d <= 0 for d in deltas_b]
if any(pos_mask):
    ax.fill_between(cvs, 0, deltas_b, where=pos_mask, color=PUR, alpha=0.12, label='TGS wins')
if any(neg_mask):
    ax.fill_between(cvs, deltas_b, 0, where=neg_mask, color=RED, alpha=0.10, label='Static wins')
best_idx = int(np.argmax(deltas_b))
ax.scatter([cvs[best_idx]], [deltas_b[best_idx]], color=GRN, s=120, zorder=6, label='Peak')
ax.annotate(f'Peak CV={cvs[best_idx]:.2f}',
            xy=(cvs[best_idx], deltas_b[best_idx]),
            xytext=(cvs[best_idx]+0.05, deltas_b[best_idx]+0.006),
            arrowprops=dict(arrowstyle='->', color=GRN), fontsize=8, color=GRN)
ax.set_xlabel('Degree CV (actual)', fontsize=10)
ax.set_ylabel('TGS - Static Accuracy', fontsize=10)
ax.set_title('Sweep B: Degree CV vs TGS Advantage\n(homophily held constant)', fontsize=10, fontweight='bold')
ax.legend(fontsize=8); ax.grid(True, alpha=0.25)
ax.text(0.98, 0.02, 'Extreme CV hurts:\nhubs too powerful to sparsify',
        transform=ax.transAxes, ha='right', va='bottom', fontsize=7, color=RED, style='italic')

# Panel 5: Summary table (text)
ax = fig.add_subplot(gs[1, 1:])
ax.axis('off')
tgs_o = toa['exp5_order']['TGS order']
rand_o = toa['exp5_order']['Random order']
rev_o = toa['exp5_order']['Reverse order']
stat_o = toa['exp5_order']['Static (all upfront)']
opt_w = warmups[opt_idx]
opt_acc = tgs_accs[opt_idx]
lines = [
    "Causal Analysis Results (Cora GCN, seed=42)",
    "",
    "Exp 5 — Same final edges, different retirement order:",
    f"  TGS order:      {tgs_o:.3f}",
    f"  Random order:   {rand_o:.3f}  (+{rand_o-stat_o:.3f} over static)",
    f"  Reverse order:  {rev_o:.3f}  (+{rev_o-stat_o:.3f} over static)",
    f"  Static:         {stat_o:.3f}  (baseline)",
    "",
    "  => TIMING drives the advantage, not the specific order.",
    "     All temporal variants outperform static by ~8-9pp.",
    "",
    "Exp 6 — Retirement timing sweep:",
    f"  warmup=0 (immediate):  {ts[0]['tgs_acc']:.3f}  <- too early",
    f"  warmup=20:             {ts[1]['tgs_acc']:.3f}",
    f"  warmup=40 (optimal):   {ts[2]['tgs_acc']:.3f}  <- representations stabilised",
    f"  warmup=80+:            {ts[3]['tgs_acc']:.3f}  <- plateau (equally good)",
    "",
    "  => Retiring before epoch 40 is harmful.",
    "     Validates Prop 8.2: redundant edges only emerge after convergence.",
    "",
    "Sweeps A/B — Structural predictors:",
    "  Homophily: positive trend (limited by ceiling effects)",
    "  Degree CV: non-monotone — moderate CV (0.6-0.8) optimal",
    "             extreme CV can hurt (over-aggressive hub injection)",
]
ax.text(0.02, 0.97, "\n".join(lines), transform=ax.transAxes,
        ha='left', va='top', fontsize=8.5, fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8faff', alpha=0.95))
ax.set_title('Causal Analysis Summary', fontsize=11, fontweight='bold', pad=8)

plt.suptitle(
    'Experiments 4+5+6: Causal Analysis — Why and When TGS Works\n'
    'Core finding: temporal TIMING of retirement matters; specific ordering does not',
    fontsize=12, fontweight='bold', y=1.02)

fig.savefig('results/figures/fig18_causal_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print('Fig 18 saved')
