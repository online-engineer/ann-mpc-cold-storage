"""Figure 4: 4-4-1 ANN architecture used in the paper.

Compact feedforward network:
  - Input layer:  4 neurons (T, m_dot, Cp, Q_L)
  - Hidden layer: 4 neurons, tansig activation
  - Output layer: 1 neuron, purelin activation (COP)

Trainable parameters: 4*4 + 4 + 4*1 + 1 = 25.

Runs on plain matplotlib. Save as PNG for the paper, or upload/share as-is.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch


from pathlib import Path
# Resolve output path relative to this script so it works from anywhere.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
OUT_PNG = _REPO_ROOT / 'figures' / 'fig04_ann_architecture.png'
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)

# ---- layout ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis('off')

BORDER = '#334155'
IN_FILL  = '#cddff0'   # input   (matches sensor layer in Fig. 1)
HID_FILL = '#d8d0e8'   # hidden  (matches ANN layer in Fig. 1)
OUT_FILL = '#ffd8b8'   # output  (matches MPC-facing colour)

# x-coordinates for each layer
x_in, x_hid, x_out = 1.6, 5.0, 8.4

# y-coordinates for neurons (centred vertically around y = 3)
def y_column(n, y_center=3.0, gap=0.9):
    total_h = (n - 1) * gap
    top = y_center + total_h / 2
    return [top - i * gap for i in range(n)]

y_in  = y_column(4)
y_hid = y_column(4)
y_out = y_column(1)

# ---- input labels (feature names) --------------------------------------
input_labels = [
    r'$T$  (°C)',
    r'$\dot{m}$  (kg/s)',
    r'$C_p$  (kJ/kg°C)',
    r'$Q_L$  (kW)',
]

for y, lbl in zip(y_in, input_labels):
    ax.text(x_in - 0.9, y, lbl, ha='right', va='center',
            fontsize=10.5, color=BORDER)

# ---- weight lines: input -> hidden -------------------------------------
for yi in y_in:
    for yh in y_hid:
        ax.plot([x_in + 0.28, x_hid - 0.28], [yi, yh],
                color='#8892a6', linewidth=0.9, alpha=0.7, zorder=1)

# ---- weight lines: hidden -> output ------------------------------------
for yh in y_hid:
    for yo in y_out:
        ax.plot([x_hid + 0.28, x_out - 0.28], [yh, yo],
                color='#8892a6', linewidth=1.0, alpha=0.75, zorder=1)

# ---- neuron circles ----------------------------------------------------
def draw_neuron(x, y, fill, label=None, label_below=False):
    c = Circle((x, y), 0.28, facecolor=fill, edgecolor=BORDER,
               linewidth=1.4, zorder=3)
    ax.add_patch(c)
    if label:
        ly = y - 0.55 if label_below else y
        ax.text(x, ly, label, ha='center', va='center',
                fontsize=9.5, color=BORDER, zorder=4)

for y in y_in:
    draw_neuron(x_in, y, IN_FILL)

for y in y_hid:
    draw_neuron(x_hid, y, HID_FILL)

# Output neuron with COP label right of it
draw_neuron(x_out, y_out[0], OUT_FILL)
ax.text(x_out + 0.65, y_out[0], 'COP', ha='left', va='center',
        fontsize=11, fontweight='bold', color=BORDER)

# ---- layer titles + activation notes -----------------------------------
title_y = 5.35
sub_y   = 4.95

ax.text(x_in,  title_y, 'Input layer',
        ha='center', fontsize=11, fontweight='bold', color=BORDER)
ax.text(x_in,  sub_y,   '4 neurons  (identity)',
        ha='center', fontsize=9, style='italic', color=BORDER)

ax.text(x_hid, title_y, 'Hidden layer',
        ha='center', fontsize=11, fontweight='bold', color=BORDER)
ax.text(x_hid, sub_y,   '4 neurons  (tansig)',
        ha='center', fontsize=9, style='italic', color=BORDER)

ax.text(x_out, title_y, 'Output layer',
        ha='center', fontsize=11, fontweight='bold', color=BORDER)
ax.text(x_out, sub_y,   '1 neuron  (purelin)',
        ha='center', fontsize=9, style='italic', color=BORDER)

# ---- parameter-count callout at bottom ---------------------------------
callout_x, callout_y, callout_w, callout_h = 2.6, 0.25, 4.8, 0.75
box = FancyBboxPatch((callout_x, callout_y), callout_w, callout_h,
                     boxstyle='round,pad=0.02,rounding_size=0.12',
                     linewidth=1.0, edgecolor=BORDER, facecolor='#f4f6fb')
ax.add_patch(box)
ax.text(callout_x + callout_w / 2, callout_y + callout_h * 0.62,
        r'Trainable parameters:  $4 \times 4 + 4 + 4 \times 1 + 1 = 25$',
        ha='center', va='center', fontsize=10.5, color=BORDER)
ax.text(callout_x + callout_w / 2, callout_y + callout_h * 0.22,
        r'Samples-to-parameters ratio: $168 / 25 \approx 6.7 : 1$',
        ha='center', va='center', fontsize=9.5,
        style='italic', color=BORDER)

plt.tight_layout()
out = str(OUT_PNG)
plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
print(f'Saved {out}')
