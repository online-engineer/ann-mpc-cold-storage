"""Framework diagram (Figure 1) for the v2 adaptive ANN-MPC-PdM architecture.
No em dashes anywhere in the rendered image."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D


from pathlib import Path
# Resolve output path relative to this script so it works from anywhere.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
OUT_PNG = _REPO_ROOT / 'figures' / 'fig01_framework.png'
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(11, 9.5))
ax.set_xlim(0, 11)
ax.set_ylim(0, 11)
ax.axis('off')

# Colour palette (brand-neutral, print-friendly)
C_PLANT   = '#c4e0d0'
C_SENSOR  = '#cddff0'
C_ANN     = '#d8d0e8'
C_MPC     = '#ffd8b8'
C_PDM     = '#ffc8c8'
C_ACT     = '#e0e0e0'
BORDER    = '#334155'

def box(x, y, w, h, label, sub, fill):
    """Rounded box with title and one-line subtitle."""
    b = FancyBboxPatch((x, y), w, h,
                       boxstyle='round,pad=0.02,rounding_size=0.14',
                       linewidth=1.3, edgecolor=BORDER,
                       facecolor=fill)
    ax.add_patch(b)
    ax.text(x + w/2, y + h*0.62, label, ha='center', va='center',
            fontsize=11, fontweight='bold', color=BORDER)
    ax.text(x + w/2, y + h*0.28, sub, ha='center', va='center',
            fontsize=8.5, color=BORDER, style='italic')

def arrow(x1, y1, x2, y2, label='', style='solid', color=BORDER, offset=0.08, fs=8):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle='->', mutation_scale=14,
                        linewidth=1.2, color=color,
                        linestyle=style)
    ax.add_patch(a)
    if label:
        mx, my = (x1 + x2)/2, (y1 + y2)/2
        ax.text(mx + offset, my + offset, label, fontsize=fs,
                color=color, style='italic')


# ---- top row: plant & sensors ----
box(3.4, 8.2, 4.2, 1.0, 'Walk-in cold storage plant',
    'R-404A primary + glycol secondary, setpoint at negative 20 C', C_PLANT)

box(3.4, 6.8, 4.2, 0.9, 'Sensor layer',
    'T, secondary-loop mass flow, Cp, cooling load Q_L', C_SENSOR)
arrow(5.5, 8.2, 5.5, 7.7)

# ---- ANN prediction block ----
box(3.4, 5.3, 4.2, 1.0, 'ANN prediction (4 to 4 to 1)',
    '25 parameters, Bayesian regularisation', C_ANN)
arrow(5.5, 6.8, 5.5, 6.3)

# ---- adaptive MPC (left) ----
box(0.4, 3.3, 3.8, 1.4,
    'Adaptive MPC controller',
    'Beta_k updates from ANN COP every hour',
    C_MPC)
arrow(4.4, 5.5, 4.2, 4.5, label='predicted COP')

# ---- Health monitor (right) ----
box(6.8, 3.3, 3.8, 1.4,
    'Health monitor',
    'H = COP_pred / COP_ref; warn, critical, RUL',
    C_PDM)
arrow(6.6, 5.5, 6.8, 4.5, label='COP trend')

# ---- Actuator ----
box(0.4, 1.4, 3.8, 1.0, 'Compressor actuator',
    'Power in 3 to 12 kW, rate limited', C_ACT)
arrow(2.3, 3.3, 2.3, 2.4)

# ---- Mode switch (from PdM to MPC) ----
box(6.8, 1.4, 3.8, 1.0, 'Mode switch',
    'Normal or Protective (8 kW cap, 0.5 kW/h rate)',
    C_ACT)
arrow(8.7, 3.3, 8.7, 2.4)

# ---- Feedback loops ----
# Actuator drives plant (long left path)
arrow(2.3, 1.4, 2.3, 0.5)
arrow(2.3, 0.5, 5.5, 0.5)
arrow(5.5, 0.5, 5.5, 1.4, label='W_comp')
# Mode switch feeds MPC (adaptive constraint update)
arrow(6.8, 1.9, 4.2, 1.9, style='dashed', label='constraint update')

# ---- Legend row along the top ----
legend_items = [
    ('Plant / physical process', C_PLANT),
    ('Sensor data', C_SENSOR),
    ('ANN prediction layer', C_ANN),
    ('MPC control layer', C_MPC),
    ('Predictive maintenance', C_PDM),
    ('Actuation', C_ACT),
]
# Two rows of three swatches at very top
for i, (lbl, col) in enumerate(legend_items):
    xx = 0.3 + (i % 3) * 3.4
    yy = 10.55 - (i // 3) * 0.35
    ax.add_patch(FancyBboxPatch((xx, yy - 0.11), 0.32, 0.22,
                                boxstyle='round,pad=0.01,rounding_size=0.05',
                                linewidth=0.8, edgecolor=BORDER,
                                facecolor=col))
    ax.text(xx + 0.42, yy, lbl, fontsize=9, va='center', color=BORDER)

# Separator line beneath legend
ax.plot([0.3, 10.7], [9.95, 9.95], color='#888', lw=0.5, linestyle='-')

# solid vs dashed line legend (bottom right area, well below diagram)
ax.plot([0.4, 0.8], [9.60, 9.60], color=BORDER, lw=1.2)
ax.text(0.9, 9.60, 'Data flow (solid arrow)', fontsize=9, va='center', color=BORDER)
ax.plot([5.0, 5.4], [9.60, 9.60], color=BORDER, lw=1.2, linestyle='--')
ax.text(5.5, 9.60, 'Adaptive update (dashed arrow)', fontsize=9, va='center', color=BORDER)

plt.tight_layout()
plt.savefig(str(OUT_PNG), dpi=300, bbox_inches='tight')
print(f'Saved {OUT_PNG}')
