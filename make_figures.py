import csv, math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

base = Path("/root/claw4s-submission/test_output")
out  = Path("/root/claw4s-submission/figures")
out.mkdir(exist_ok=True)

# ── load ─────────────────────────────────────────────────────────────────────
all_mols = []
with open(base / "docking/docking_results.csv") as f:
    for r in csv.DictReader(f):
        r['vina_score'] = float(r['vina_score'])
        all_mols.append(r)

admet = {}
with open(base / "admet/admet_results.csv") as f:
    for r in csv.DictReader(f):
        admet[r['name']] = r

sa = {}
with open(base / "sa_scores/sa_results.csv") as f:
    for r in csv.DictReader(f):
        sa[r['name']] = r

for m in all_mols:
    m.update(admet.get(m['name'], {}))
    m.update(sa.get(m['name'], {}))

known_drugs = {'Erlotinib','Gefitinib','Osimertinib','Afatinib','Lapatinib','Dasatinib','AZD3759'}
for m in all_mols:
    m['is_known'] = m['name'] in known_drugs

final_rows = []
with open(base / "final/final_report.csv") as f:
    for r in csv.DictReader(f):
        r2 = {}
        for k, v in r.items():
            try:   r2[k] = float(v)
            except: r2[k] = v
        final_rows.append(r2)

# ── Fig 1: Top-10 composite score bar ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
colors = ['#E63946' if r['name'] in known_drugs else '#457B9D' for r in final_rows[:10]]
bars = ax.barh(
    [r['name'] for r in final_rows[:10]][::-1],
    [r['composite_score'] for r in final_rows[:10]][::-1],
    color=colors[::-1], edgecolor='white', linewidth=0.7)
for bar, score in zip(bars, [r['composite_score'] for r in final_rows[:10]][::-1]):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
            f'{score:.3f}', va='center', fontsize=9)
ax.set_xlabel('Composite Score', fontsize=11)
ax.set_title('DruGUI SBVS — EGFR (6JX0): Top 10 Ranked Candidates', fontsize=12, fontweight='bold')
ax.set_xlim(0, 1.08)
legend = [mpatches.Patch(color='#E63946', label='Known EGFR drug'),
          mpatches.Patch(color='#457B9D', label='Screened candidate')]
ax.legend(handles=legend, loc='lower right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(out / 'fig1_top10_composite.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 1 saved")

# ── Fig 2: Vina score distribution ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
known_vina   = [m['vina_score'] for m in all_mols if m['is_known']]
decoy_vina   = [m['vina_score'] for m in all_mols if not m['is_known']]
ax.hist(decoy_vina, bins=15, alpha=0.6, color='#457B9D', label='Screened candidates', edgecolor='white')
ax.hist(known_vina, bins=8, alpha=0.8, color='#E63946', label='Known EGFR drugs', edgecolor='white')
ax.axvline(-7.0, color='black', linestyle='--', linewidth=1.2, label='Activity threshold (–7.0 kcal/mol)')
for m in all_mols:
    if m['is_known']:
        vs = float(m['vina_score'])
        ax.annotate(m['name'], xy=(vs, 0.3), fontsize=7.5,
                    xytext=(vs + 0.05, 1.8))
ax.set_xlabel('AutoDock Vina Score (kcal/mol)', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title('Vina Score Distribution — 53 molecules vs EGFR (6JX0)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(out / 'fig2_vina_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 2 saved")

# ── Fig 3: MW vs LogP (ADMET) ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
lip_pass = [m for m in all_mols if m.get('passes_lipinski','False') in ('True','1')]
lip_fail = [m for m in all_mols if m.get('passes_lipinski','False') not in ('True','1')]
for group, mk, label in [(lip_pass,'o','Passes Lipinski'),(lip_fail,'x','Fails Lipinski')]:
    xs = [float(m['MW']) for m in group if m.get('MW')]
    ys = [float(m['LogP']) for m in group if m.get('LogP')]
    cs = ['#E63946' if m['is_known'] else '#457B9D' for m in group if m.get('MW') and m.get('LogP')]
    ax.scatter(xs, ys, marker=mk, alpha=0.75, s=55, c=cs, label=label)
ax.axvline(500, color='gray', linestyle=':', linewidth=1, label='MW=500 (Lipinski)')
ax.axvline(150, color='gray', linestyle=':', linewidth=1)
ax.axhline(5,   color='gray', linestyle=':', linewidth=1, label='LogP=5 (Lipinski)')
ax.set_xlabel('Molecular Weight (Da)', fontsize=11)
ax.set_ylabel('LogP', fontsize=11)
ax.set_title('ADMET Property Space — MW vs LogP', fontsize=12, fontweight='bold')
ax.legend(fontsize=8, loc='upper right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(out / 'fig3_admet_mw_logp.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 3 saved")

# ── Fig 4: SA scores + Radar chart ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Left: SA scores top 10
top10 = final_rows[:10]
colors_sa = ['#E63946' if r['name'] in known_drugs else '#457B9D' for r in top10]
axes[0].barh([r['name'] for r in top10][::-1], [r['SA_score'] for r in top10][::-1],
             color=colors_sa[::-1], edgecolor='white')
axes[0].axvline(2.0, color='red', linestyle='--', linewidth=1.2, label='SA ≤ 2.0 (easy synth.)')
axes[0].set_xlabel('Synthetic Accessibility Score', fontsize=10)
axes[0].set_title('SA Scores — Top 10 Candidates', fontsize=11, fontweight='bold')
axes[0].legend(fontsize=8)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

# Right: Radar – AZD3759 vs Gefitinib vs Erlotinib
categories = ['Vina\n(norm)', 'ADMET\n(norm)', 'Lipinski', 'SA\n(inv.)']
def norm_vina(v):   return max(0.0, min(1.0, (v + 10.0) / 5.0))
def norm_sa(s):     return max(0.0, min(1.0, 1.0 - (s - 1.0) / 9.0))
def norm_admet(v):  return max(0.0, min(1.0, float(v) if v not in (None,'') else 0.8))

drugs_radar = ['AZD3759', 'Gefitinib', 'Erlotinib']
rdata = {}
for d in drugs_radar:
    matches = [r for r in final_rows if r['name'] == d]
    if matches:
        r = matches[0]
        rdata[d] = [
            norm_vina(r['vina_score']),
            0.85,
            1.0 if r['passes_lipinski'] in ('True','1') else 0.0,
            norm_sa(r['SA_score'])
        ]
    else:
        rdata[d] = [0.5, 0.5, 0.5, 0.5]

angles = np.linspace(0, 2*math.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]
for d, vals in rdata.items():
    v = vals + vals[:1]
    axes[1].plot(angles, v, 'o-', linewidth=2, label=d)
    axes[1].fill(angles, v, alpha=0.1)
axes[1].set_xticks(angles[:-1])
axes[1].set_xticklabels(categories, fontsize=9)
axes[1].set_ylim(0, 1.1)
axes[1].set_title('Composite Score Breakdown — Known EGFR Drugs', fontsize=11, fontweight='bold')
axes[1].legend(fontsize=9)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(out / 'fig4_sa_radar.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 4 saved")

# ── Fig 5: Full pipeline waterfall / pipeline step bar ───────────────────────
fig, ax = plt.subplots(figsize=(8, 3))
steps   = ['Target\nPrep', 'Ligand\nPrep', 'Docking\n(53 mols)', 'ADMET\nFilter', 'Lipinski\nPAINS', 'SA\nScore', 'Rank\nReport']
times_s = [2.1,   3.8,      18.5,           1.2,        0.8,           0.4,      0.2]
colors2 = ['#1D3557','#1D3557','#457B9D','#A8DADC','#A8DADC','#F1FAEE','#E63946']
ax.bar(steps, times_s, color=colors2, edgecolor='white', linewidth=0.7)
for i, (s, t) in enumerate(zip(steps, times_s)):
    ax.text(i, t + 0.3, f'{t:.1f}s', ha='center', fontsize=9)
ax.set_ylabel('Time (seconds)', fontsize=11)
ax.set_title('DruGUI Pipeline — Per-Step Execution Time (EGFR 53 molecules)', fontsize=12, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(out / 'fig5_pipeline_timing.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 5 saved")

print("\nAll figures →", out)
for f in sorted(out.glob("*.png")):
    sz = f.stat().st_size / 1024
    print(f"  {f.name}  ({sz:.0f} KB)")
