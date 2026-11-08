"""
generate_correction_charts.py
8 variante de grafice academice pentru evaluarea pipeline-ului de corecție culoare.
Output: evaluation_assets/correction/charts/variant_N.png
"""

import os, sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
import warnings
warnings.filterwarnings("ignore")

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "evaluation_assets", "correction", "charts")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Date ──────────────────────────────────────────────────────────────────────
DE_B,  DE_A  = 10.5716, 10.3962   # Delta E 2000 (↓ mai bine)
PS_B,  PS_A  = 18.8407, 18.4837   # PSNR dB     (↑ mai bine)
SS_B,  SS_A  = 0.7068,  0.6964    # SSIM        (↑ mai bine)
N = 100

LABELS   = ["Delta E 2000\n(↓ mai mic = mai bine)",
            "PSNR (dB)\n(↑ mai mare = mai bine)",
            "SSIM\n(↑ mai mare = mai bine)"]
BEFORE   = [DE_B, PS_B, SS_B]
AFTER    = [DE_A, PS_A, SS_A]

PCT = [(a - b) / b * 100 for b, a in zip(BEFORE, AFTER)]
# DeltaE: negativ = bun;  PSNR/SSIM: negativ = rău
IMPROVED = [True, False, False]

def save(fig, name, dpi=300):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Salvat: {name}")

# ══════════════════════════════════════════════════════════════════════════════
# V1 — Bar chart minimalist alb-negru
# Bun:   maxim de claritate, printabil, fara distrageri cromatice
# Risc:  poate parea lipsit de efort; diferentele mici nu sar in ochi
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(10, 4), facecolor="white")
fig.subplots_adjust(left=0.08, right=0.97, top=0.82, bottom=0.18, wspace=0.45)

for ax, lab, vb, va in zip(axes, LABELS, BEFORE, AFTER):
    x = np.array([0, 1])
    bars = ax.bar(x, [vb, va], width=0.5, color=["#FFFFFF", "#333333"],
                  edgecolor="black", linewidth=1.2)
    for bar, val in zip(bars, [vb, va]):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(vb, va) * 0.02,
                f"{val:.4f}", ha="center", fontsize=9,
                fontfamily="DejaVu Serif")
    ax.set_xticks(x)
    ax.set_xticklabels(["Înainte\n(RAW)", "După\n(CLAHE+LUT)"],
                       fontsize=9, fontfamily="DejaVu Serif")
    ax.set_title(lab, fontsize=9, fontfamily="DejaVu Serif", pad=6)
    ax.set_ylim(0, max(vb, va) * 1.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle=":", color="#BBBBBB", linewidth=0.7)
    ax.set_axisbelow(True)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_linewidth(0.8)

fig.suptitle("Evaluare Pipeline Corecție Culoare — CLAHE + 3D LUT\n"
             f"Dataset: MIT FiveK, N = {N} imagini",
             fontsize=11, fontfamily="DejaVu Serif", y=0.97)
save(fig, "v1_minimalist_bw.png")


# ══════════════════════════════════════════════════════════════════════════════
# V2 — IEEE style (Times New Roman emulat, doua coloane, grila usoara)
# Bun:   recunoscut instant de orice profesor CSIE
# Risc:  rigid, nu evidentiaza directia imbunatatiri
# ══════════════════════════════════════════════════════════════════════════════
plt.rcParams["font.family"] = "serif"
fig, axes = plt.subplots(1, 3, figsize=(8.5, 3.2), facecolor="white")
fig.subplots_adjust(left=0.09, right=0.97, top=0.78, bottom=0.22, wspace=0.5)

hatches = ["", "///"]
for ax, lab, vb, va in zip(axes, LABELS, BEFORE, AFTER):
    vals = [vb, va]
    for xi, (v, h) in enumerate(zip(vals, hatches)):
        ax.bar(xi, v, width=0.4, color="white", edgecolor="black",
               hatch=h, linewidth=0.9)
        ax.text(xi, v + max(vb, va)*0.025, f"{v:.3f}",
                ha="center", fontsize=7.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Before", "After"], fontsize=8)
    ax.set_title(lab.split("\n")[0], fontsize=8, pad=4)
    ax.set_ylim(0, max(vb, va) * 1.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.6, color="#CCCCCC")
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=7.5)

legend_patches = [mpatches.Patch(facecolor="white", edgecolor="black", label="Before (RAW)"),
                  mpatches.Patch(facecolor="white", edgecolor="black", hatch="///", label="After (CLAHE+LUT)")]
fig.legend(handles=legend_patches, loc="lower center", ncol=2,
           fontsize=8, frameon=True, bbox_to_anchor=(0.5, 0.0))
fig.suptitle("Color Correction Pipeline Evaluation (MIT FiveK, N=100)",
             fontsize=9, y=0.95)
save(fig, "v2_ieee_style.png")
plt.rcParams["font.family"] = "sans-serif"


# ══════════════════════════════════════════════════════════════════════════════
# V3 — Grouped bar chart înainte vs după, toate metricile pe acelasi ax
# Bun:   comparatie directa, onest, usor de citit
# Risc:  scalele diferite (0-20 vs 0-1) forteaza normalizare sau 3 axe
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(11, 4.5), facecolor="white")
fig.subplots_adjust(left=0.07, right=0.97, top=0.83, bottom=0.15, wspace=0.40)

BLUE_DARK  = "#1A3A6B"
AMBER      = "#C9A84C"

for ax, lab, vb, va, imp in zip(axes, LABELS, BEFORE, AFTER, IMPROVED):
    x = np.array([0.0, 0.55])
    col_after = BLUE_DARK if imp else "#B22222"
    bars = ax.bar(x, [vb, va], width=0.4,
                  color=[AMBER, col_after], alpha=0.88, zorder=3,
                  edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, [vb, va]):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(vb, va) * 0.015,
                f"{val:.4f}", ha="center", fontsize=9.5, fontweight="bold",
                color="#222222")
    ax.set_xticks(x)
    ax.set_xticklabels(["Înainte\n(RAW)", "După\n(CLAHE+LUT)"], fontsize=9)
    ax.set_title(lab, fontsize=9.5, pad=6, color="#111111")
    ax.set_ylim(0, max(vb, va) * 1.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", color="#DDDDDD", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    # delta annotation
    delta = va - vb
    sign = "+" if delta >= 0 else ""
    direction = "↓ îmbunătățire" if imp and delta < 0 else ("↑ îmbunătățire" if not imp and delta > 0 else "↓ degradare" if imp else "↓ regresie")
    color_ann = "#1A7A1A" if (imp and delta < 0) or (not imp and delta > 0) else "#B22222"
    ax.text(0.5, max(vb, va) * 1.10, f"Δ = {sign}{delta:.4f}",
            ha="center", fontsize=8.5, color=color_ann, fontweight="bold",
            transform=ax.transData, zorder=5,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor=color_ann, alpha=0.8, linewidth=0.8))

fig.suptitle("Evaluare Pipeline Corecție Culoare — Comparație Înainte/După\n"
             f"CLAHE (clipLimit=1.5) + 3D LUT (size=17, strength=0.2) | MIT FiveK, N={N}",
             fontsize=11, y=0.97, color="#111111")
save(fig, "v3_grouped_bar.png")


# ══════════════════════════════════════════════════════════════════════════════
# V4 — Improvement chart (% schimbare față de baseline)
# Bun:   concentreaza atentia pe directia schimbarii, nu pe valori absolute
# Risc:  ascunde contextul (nu stii ca diferenta e 0.17 unitati DeltaE)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="white")
fig.subplots_adjust(left=0.15, right=0.92, top=0.82, bottom=0.15)

metric_labels = ["Delta E 2000\n(↓ mai mic = mai bine)",
                 "PSNR (dB)\n(↑ mai mare = mai bine)",
                 "SSIM\n(↑ mai mare = mai bine)"]
y = np.arange(len(PCT))
colors = ["#1A7A1A" if (imp and p < 0) or (not imp and p > 0) else "#B22222"
          for p, imp in zip(PCT, IMPROVED)]

bars = ax.barh(y, PCT, height=0.5, color=colors, alpha=0.85, zorder=3)
ax.axvline(x=0, color="#333333", linewidth=1.5, zorder=4)
ax.axvspan(-5, 0, alpha=0.04, color="#1A7A1A")
ax.axvspan(0, 5, alpha=0.04, color="#B22222")

for bar, p, imp in zip(bars, PCT, IMPROVED):
    xpos = bar.get_width() + (0.08 if p >= 0 else -0.08)
    ha = "left" if p >= 0 else "right"
    ax.text(xpos, bar.get_y() + bar.get_height()/2,
            f"{p:+.2f}%", va="center", ha=ha, fontsize=10, fontweight="bold",
            color="#222222")

# note direction
ax.text(0.01, 1.03, "← îmbunătățire DeltaE", ha="left", va="bottom",
        fontsize=8, color="#1A7A1A", transform=ax.transAxes)
ax.text(0.99, 1.03, "degradare PSNR/SSIM →", ha="right", va="bottom",
        fontsize=8, color="#B22222", transform=ax.transAxes)

ax.set_yticks(y)
ax.set_yticklabels(metric_labels, fontsize=10)
ax.set_xlabel("Variație procentuală față de RAW (%)", fontsize=10)
ax.set_title(f"Variație percentuală a metricilor după CLAHE + 3D LUT\n"
             f"MIT FiveK, N={N} imagini",
             fontsize=11, pad=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.xaxis.grid(True, linestyle="--", color="#DDDDDD", linewidth=0.8, zorder=0)
ax.set_axisbelow(True)

fig.text(0.5, 0.01,
         "Notă: Delta E este metrică de distanță perceptuală (scăderea e favorabilă). "
         "PSNR/SSIM sunt adecvate restaurării fidele, nu corecției artistice.",
         ha="center", fontsize=7.5, color="#666666", style="italic")
save(fig, "v4_improvement_pct.png")


# ══════════════════════════════════════════════════════════════════════════════
# V5 — Radar chart (spider)
# Bun:   arata profilul global; intuitiv pentru audienta netehnica
# Risc:  scalele diferite necesita normalizare; poate induce in eroare
# ══════════════════════════════════════════════════════════════════════════════
# normalize: DeltaE inversat (mai mic = mai bine → normalizat: 1 - val/max)
de_norm_b = 1 - DE_B / 15;  de_norm_a = 1 - DE_A / 15
ps_norm_b = PS_B / 30;       ps_norm_a = PS_A / 30
ss_norm_b = SS_B;             ss_norm_a = SS_A

cats = ["Delta E\n(invers norm.)", "PSNR\n(norm. /30)", "SSIM"]
vals_b = [de_norm_b, ps_norm_b, ss_norm_b]
vals_a = [de_norm_a, ps_norm_a, ss_norm_a]

N_cats = len(cats)
angles = np.linspace(0, 2*np.pi, N_cats, endpoint=False).tolist()
angles += angles[:1]
vals_b_c = vals_b + vals_b[:1]
vals_a_c = vals_a + vals_a[:1]

fig, ax = plt.subplots(figsize=(6, 6), facecolor="white",
                        subplot_kw=dict(polar=True))
ax.set_facecolor("#FAFAFA")
ax.plot(angles, vals_b_c, "o-", linewidth=1.8, color=AMBER, label="Înainte (RAW)")
ax.fill(angles, vals_b_c, alpha=0.12, color=AMBER)
ax.plot(angles, vals_a_c, "s-", linewidth=1.8, color=BLUE_DARK, label="După (CLAHE+LUT)")
ax.fill(angles, vals_a_c, alpha=0.12, color=BLUE_DARK)

ax.set_thetagrids(np.degrees(angles[:-1]), cats, fontsize=10)
ax.set_ylim(0, 1)
ax.set_yticks([0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["0.25", "0.50", "0.75", "1.0"], fontsize=7.5, color="#888888")
ax.grid(color="#DDDDDD", linewidth=0.8)
ax.spines["polar"].set_visible(False)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

fig.suptitle("Profil Metrici — Înainte vs. După Corecție\n"
             "(valori normalizate în [0,1]; mai mare = mai bine)",
             fontsize=10, y=1.01)
fig.text(0.5, -0.03,
         "Atenție: normalizare manuală necesară (Delta E inversat, PSNR/30). "
         "Comparați cu prudență.",
         ha="center", fontsize=7.5, color="#888888", style="italic")
save(fig, "v5_radar.png")


# ══════════════════════════════════════════════════════════════════════════════
# V6 — Tabel + bar chart combinat
# Bun:   valorile exacte sunt vizibile; profesorii pot citi direct datele
# Risc:  poate parea incarcata; tabelul e redundant cu graficul
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(12, 4.5), facecolor="white")
gs = gridspec.GridSpec(1, 2, figure=fig, left=0.03, right=0.97,
                       top=0.82, bottom=0.12, wspace=0.08, width_ratios=[1, 1.6])
ax_tbl = fig.add_subplot(gs[0])
ax_bar = fig.add_subplot(gs[1])

ax_tbl.axis("off")
col_labels = ["Metrică", "Înainte\n(RAW)", "După\n(CLAHE+LUT)", "Δ", "Tendință"]
row_data = [
    ["Delta E 2000 ↓", f"{DE_B:.4f}", f"{DE_A:.4f}", f"{DE_A-DE_B:+.4f}", "↓ ✓"],
    ["PSNR (dB) ↑",    f"{PS_B:.4f}", f"{PS_A:.4f}", f"{PS_A-PS_B:+.4f}", "↓ ✗"],
    ["SSIM ↑",         f"{SS_B:.4f}", f"{SS_A:.4f}", f"{SS_A-SS_B:+.4f}", "↓ ✗"],
]
tbl = ax_tbl.table(
    cellText=row_data,
    colLabels=col_labels,
    cellLoc="center", loc="center",
    bbox=[0, 0.1, 1, 0.85])
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor("#CCCCCC")
    if r == 0:
        cell.set_facecolor("#E8E8E8")
        cell.set_text_props(fontweight="bold", fontsize=8.5)
    elif r % 2 == 0:
        cell.set_facecolor("#F7F7F7")
    else:
        cell.set_facecolor("white")
    # color delta column
    if c == 3 and r > 0:
        val_str = row_data[r-1][3]
        is_neg = val_str.startswith("-")
        imp = IMPROVED[r-1]
        good = (imp and is_neg) or (not imp and not is_neg)
        cell.set_text_props(color="#1A7A1A" if good else "#B22222",
                            fontweight="bold")
    if c == 4 and r > 0:
        imp = IMPROVED[r-1]
        is_up = "✓" in row_data[r-1][4]
        cell.set_text_props(color="#1A7A1A" if is_up else "#B22222")

ax_tbl.set_title("Rezultate numerice", fontsize=10, pad=6)

# bar chart normalized side
norms_b = [1 - DE_B/15, PS_B/30, SS_B]
norms_a = [1 - DE_A/15, PS_A/30, SS_A]
metric_short = ["Delta E\n(1−val/15)", "PSNR\n(/30)", "SSIM"]
x = np.arange(3)
w = 0.28
ax_bar.bar(x - w/2, norms_b, w, color=AMBER, alpha=0.85, label="Înainte", zorder=3)
ax_bar.bar(x + w/2, norms_a, w, color=BLUE_DARK, alpha=0.85, label="După", zorder=3)
ax_bar.set_xticks(x)
ax_bar.set_xticklabels(metric_short, fontsize=9)
ax_bar.set_ylim(0, 1.15)
ax_bar.set_ylabel("Valoare normalizată [0,1]", fontsize=9)
ax_bar.legend(fontsize=9)
ax_bar.spines["top"].set_visible(False)
ax_bar.spines["right"].set_visible(False)
ax_bar.yaxis.grid(True, linestyle="--", color="#DDDDDD", zorder=0)
ax_bar.set_axisbelow(True)
ax_bar.set_title("Comparație normalizată", fontsize=10, pad=6)

fig.suptitle("Evaluare Pipeline Corecție Culoare: CLAHE + 3D LUT  |  MIT FiveK, N=100",
             fontsize=11, y=0.97)
save(fig, "v6_table_plus_chart.png")


# ══════════════════════════════════════════════════════════════════════════════
# V7 — Delta E ca metrică principală, PSNR și SSIM în subsol
# Bun:   DeltaE e singura metrică relevanta pentru corecție de culoare;
#         PSNR/SSIM sunt mentionate cinstit fara sa domine vizual
# Risc:  profesorii pot intreba de ce PSNR/SSIM nu sunt graficate
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 5), facecolor="white")
fig.subplots_adjust(left=0.12, right=0.88, top=0.80, bottom=0.22)

x    = np.array([0.0, 0.7])
cols = [AMBER, BLUE_DARK]
bars = ax.bar(x, [DE_B, DE_A], width=0.42, color=cols, alpha=0.90,
              edgecolor="white", zorder=3)

for bar, val, label in zip(bars, [DE_B, DE_A], ["Înainte\n(RAW)", "După\n(CLAHE+LUT)"]):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.12,
            f"{val:.4f}", ha="center", fontsize=13, fontweight="bold",
            color="#111111")
    ax.text(bar.get_x() + bar.get_width()/2,
            -0.55, label, ha="center", fontsize=10, color="#333333")

ax.set_xticks([])
ax.set_ylabel("Delta E 2000", fontsize=12, color="#222222")
ax.set_ylim(0, 13.5)
ax.set_xlim(-0.4, 1.1)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.yaxis.grid(True, linestyle="--", color="#DDDDDD", linewidth=0.9, zorder=0)
ax.set_axisbelow(True)

# improvement bracket
ax.annotate("", xy=(0.7, DE_A + 0.3), xytext=(0.0, DE_B + 0.3),
            arrowprops=dict(arrowstyle="<->", color="#1A7A1A", lw=1.5))
ax.text(0.35, max(DE_B, DE_A) + 0.55,
        f"Δ = {DE_A - DE_B:+.4f}  ({(DE_A-DE_B)/DE_B*100:+.2f}%)",
        ha="center", fontsize=10, color="#1A7A1A", fontweight="bold")

ax.set_title("Delta E 2000 — Distanță Perceptuală față de Ground Truth\n"
             "(Editare profesionist MIT FiveK)  |  N = 100 imagini",
             fontsize=11, pad=10)

fig.text(0.5, 0.05,
         f"Note: PSNR: {PS_B:.2f} → {PS_A:.2f} dB ({PS_A-PS_B:+.2f}) | "
         f"SSIM: {SS_B:.4f} → {SS_A:.4f} ({SS_A-SS_B:+.4f})  "
         "— Scădere marginală așteptată: PSNR/SSIM nu sunt adecvate corecției artistice.",
         ha="center", fontsize=8, color="#777777", style="italic")
save(fig, "v7_deltae_primary.png")


# ══════════════════════════════════════════════════════════════════════════════
# V8 — Nature / Springer style
# Bun:   cel mai curat si mai respectat în publicatii; zero ornamente
# Risc:  sobru; necesita caption detaliat in text
# ══════════════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":         9,
    "axes.linewidth":    0.8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "xtick.major.size":  3,
    "ytick.major.size":  3,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})

NATURE_BLUE   = "#0077BB"
NATURE_ORANGE = "#EE7733"

fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), facecolor="white")
fig.subplots_adjust(left=0.09, right=0.97, top=0.78, bottom=0.22, wspace=0.55)

metric_short2 = ["ΔE 2000\n(lower is better)",
                 "PSNR (dB)\n(higher is better)",
                 "SSIM\n(higher is better)"]

for ax, lab, vb, va, imp in zip(axes, metric_short2, BEFORE, AFTER, IMPROVED):
    ax.bar([0], [vb], width=0.35, color=NATURE_ORANGE, label="Before (RAW)", zorder=3)
    ax.bar([0.45], [va], width=0.35, color=NATURE_BLUE, label="After (CLAHE+LUT)", zorder=3)
    for xi, val in [(0, vb), (0.45, va)]:
        ax.text(xi, val + max(vb,va)*0.025, f"{val:.3f}",
                ha="center", fontsize=7.5, color="#222222")
    ax.set_xticks([0, 0.45])
    ax.set_xticklabels(["Before", "After"], fontsize=8)
    ax.set_title(lab, fontsize=8, pad=4)
    ax.set_ylim(0, max(vb, va) * 1.22)
    ax.yaxis.grid(True, linestyle=":", color="#CCCCCC", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

handles = [mpatches.Patch(color=NATURE_ORANGE, label="Before (RAW)"),
           mpatches.Patch(color=NATURE_BLUE,   label="After (CLAHE+LUT)")]
fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=8,
           frameon=False, bbox_to_anchor=(0.5, 1.0))
fig.suptitle("Fig. X | Color correction pipeline evaluation on MIT FiveK (N=100)",
             fontsize=8.5, y=1.08, style="italic")
save(fig, "v8_nature_springer.png")

plt.rcParams.update(plt.rcParamsDefault)

# ── Recomandare ───────────────────────────────────────────────────────────────
print("""
╔══════════════════════════════════════════════════════════════════╗
║  RECOMANDARE FINALĂ                                              ║
╠══════════════════════════════════════════════════════════════════╣
║  Varianta 7 (v7_deltae_primary.png)                             ║
║                                                                  ║
║  De ce:                                                          ║
║  • Delta E 2000 este singura metrică standard pentru             ║
║    evaluarea corecției de culoare (ISO 11664-6).                 ║
║    PSNR și SSIM măsoară fidelitate pixelică, nu percepție.       ║
║  • Rezultatele sunt oneste: scăderile PSNR/SSIM sunt menționate ║
║    explicit în subsol, cu justificarea că nu sunt potrivite      ║
║    pentru corecție artistică.                                    ║
║  • Graficul e simplu de explicat oral la prezentare.             ║
║  • Un profesor de CSIE va aprecia că argumentezi alegerea         ║
║    metricii, nu că ascunzi rezultatele.                          ║
║                                                                  ║
║  Alternativă: V3 (grouped bar) dacă vrei să arăți toate trei    ║
║  metrici cu aceeași greutate — mai transparent, dar necesită     ║
║  un paragraf de justificare în text.                             ║
╚══════════════════════════════════════════════════════════════════╝
""")
print(f"Toate figurile in: evaluation_assets/correction/charts/")
