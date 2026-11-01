"""Draw the results figure from the two result files (no numbers typed by hand).
    python make_chart.py results_v2_baselines.json tabpfn_results.json quiet_villages_results.png
"""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

base = json.load(open(sys.argv[1]))
tab = json.load(open(sys.argv[2]))
out = sys.argv[3]
M, T = base["models"], tab["sets"]
BLUE, AMBER, INK, MUTED, GRID, SURF = "#1f5fbf", "#b8741a", "#1d2330", "#5b6577", "#e3e6ec", "#fcfcfb"

rows = [  # (label, block, is_tabpfn, group)
    ("Plain rule: smaller = riskier", M["rule_smaller_is_riskier"], False, "Size only (6 features)"),
    ("Logistic regression", M["logistic_regression_SIZE"], False, "Size only (6 features)"),
    ("Gradient boosting", M["gradient_boosting_SIZE"], False, "Size only (6 features)"),
    ("TabPFN", T["SIZE"], True, "Size only (6 features)"),
    ("Logistic regression", M["logistic_regression_FULL"], False, "All 37 features"),
    ("Gradient boosting", M["gradient_boosting_FULL"], False, "All 37 features"),
    ("TabPFN", T["FULL"], True, "All 37 features"),
]
panels = [("auc", "auc_ci", "AUC (higher is better; 0.5 = chance)", (0.62, 0.82), "{:.3f}"),
          ("precision_top10pct", "ptop10_ci", "Share of the riskiest 10% that really lost people", (0.05, 0.32), "{:.1%}")]

plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": MUTED})
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6), facecolor=SURF, gridspec_kw={"wspace": 0.08})
ypos, y, groups = [], 0.0, []
for i, r in enumerate(rows):
    if i in (0, 4):
        y += 0.9 if i else 0.0
        groups.append((y, r[3]))
        y += 0.8
    ypos.append(y); y += 1.0
for ax, (k, ck, title, xlim, fmt) in zip(axes, panels):
    ax.set_facecolor(SURF)
    for (label, blk, is_tab, _), yy in zip(rows, ypos):
        c = BLUE if is_tab else AMBER
        lo, hi = blk[ck]
        ax.plot([lo, hi], [yy, yy], color=c, lw=2, solid_capstyle="round", zorder=2)
        ax.plot(blk[k], yy, "o", ms=9, mfc=c, mec=SURF, mew=2, zorder=3)
        ax.text(blk[k], yy - 0.42, fmt.format(blk[k]), ha="center", va="bottom", fontsize=9.5, color=INK, fontweight="bold" if is_tab else "normal")
    if k == "precision_top10pct":
        ax.axvline(base["prevalence"], color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
        ax.text(base["prevalence"] + 0.003, ypos[0] - 0.55, f"all villages: {base['prevalence']:.1%}", fontsize=9, color=MUTED, va="center")
    ax.set_xlim(*xlim); ax.set_ylim(ypos[-1] + 1.2, -0.9)
    ax.set_title(title, fontsize=11, loc="left", color=INK, pad=10)
    ax.grid(axis="x", color=GRID, lw=1); ax.set_axisbelow(True)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID); ax.tick_params(axis="x", colors=MUTED, labelsize=9.5); ax.tick_params(axis="y", length=0)
    ax.set_yticks([])
for (label, blk, is_tab, _), yy in zip(rows, ypos):
    axes[0].text(0.62 - 0.004, yy, label, ha="right", va="center", fontsize=10.5, color=INK, fontweight="bold" if is_tab else "normal")
for gy, g in groups:
    axes[0].text(0.62 - 0.004, gy - 0.1, g, ha="right", va="center", fontsize=9.5, color=MUTED, style="italic")
axes[0].set_xticks([0.65, 0.70, 0.75, 0.80])
axes[1].set_xticks([0.05, 0.10, 0.15, 0.20, 0.25, 0.30]); axes[1].set_xticklabels(["5%", "10%", "15%", "20%", "25%", "30%"])
fig.suptitle("Predicting which Uttarakhand villages lost ≥30% of their people (2001→2011), with whole districts held out",
             x=0.015, ha="left", fontsize=12.5, fontweight="bold", color=INK, y=0.985)
fig.text(0.015, 0.02, f"n = {base['n']:,} villages, {base['positives']:,} depopulated ({base['prevalence']:.1%}). Dots = estimate; lines = 95% bootstrap interval "
         "(optimistic: villages in a district are not independent).  Data: SHRUG / Census of India.", fontsize=8.8, color=MUTED, ha="left")
fig.subplots_adjust(left=0.265, right=0.985, top=0.86, bottom=0.14)
fig.savefig(out, dpi=170, facecolor=SURF)
print("saved", out)
