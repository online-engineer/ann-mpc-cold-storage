"""Figure 5: internal processing of a single artificial neuron.

Shows: inputs x_i with weights w_i -> weighted sum + bias -> activation -> output.
Formula rendered above the diagram: y = sigma( sum_i w_i x_i + b ).

Runs on plain matplotlib. Same colour palette as Figure 4 for visual consistency.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch


from pathlib import Path
# Resolve output path relative to this script so it works from anywhere.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
OUT_PNG = _REPO_ROOT / 'figures' / 'fig05_neuron.png'
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)

# ---- layout ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis('off')

BORDER  = '#334155'
IN_FILL = '#cddff0'
SUM_FILL = '#d8d0e8'
ACT_FILL = '#ffd8b8'
OUT_FILL = '#c4e0d0'

# x-coordinates for each stage
x_in   = 1.2   # input nodes
x_sum  = 4.8   # summation node
x_act  = 6.9   # activation node
x_out  = 8.7   # output node

# input y-positions
y_in = [4.6, 3.7, 2.8, 1.4]        # x_1, x_2, x_3, ... x_n
y_center = 3.0                      # summation, activation, output all sit here

# ---- input labels (x_1, x_2, x_3, ellipsis, x_n) -----------------------
input_texts = [r'$x_1$', r'$x_2$', r'$x_3$', r'$x_n$']
weight_texts = [r'$w_1$', r'$w_2$', r'$w_3$', r'$w_n$']

# vertical dots between x_3 and x_n
ax.text(x_in - 0.05, 2.15, r'$\vdots$', ha='center', va='center',
        fontsize=16, color=BORDER)

# input neurons + labels
for y, xl in zip(y_in, input_texts):
    c = Circle((x_in, y), 0.24, facecolor=IN_FILL, edgecolor=BORDER,
               linewidth=1.3, zorder=3)
    ax.add_patch(c)
    ax.text(x_in - 0.7, y, xl, ha='right', va='center',
            fontsize=12, color=BORDER)

# ---- weight arrows: input -> summation ---------------------------------
for y, wl in zip(y_in, weight_texts):
    a = FancyArrowPatch((x_in + 0.24, y), (x_sum - 0.38, y_center),
                        arrowstyle='->', mutation_scale=13,
                        linewidth=1.1, color=BORDER, zorder=2)
    ax.add_patch(a)
    # weight label on the mid-segment, slightly above the arrow
    mx = 0.55 * x_in + 0.45 * x_sum
    my = 0.55 * y + 0.45 * y_center
    ax.text(mx, my + 0.18, wl, ha='center', va='center',
            fontsize=11, color=BORDER,
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                      edgecolor='none', alpha=0.85))

# ---- bias input from below ---------------------------------------------
bias_x, bias_y = x_sum, 0.55
b_box = FancyBboxPatch((bias_x - 0.22, bias_y - 0.22), 0.44, 0.44,
                       boxstyle='round,pad=0.02,rounding_size=0.08',
                       facecolor='#f4f6fb', edgecolor=BORDER, linewidth=1.1)
ax.add_patch(b_box)
ax.text(bias_x, bias_y, r'$b$', ha='center', va='center',
        fontsize=12, color=BORDER)
ax.text(bias_x + 0.55, bias_y, 'bias', ha='left', va='center',
        fontsize=10, style='italic', color=BORDER)

a_bias = FancyArrowPatch((bias_x, bias_y + 0.22), (x_sum, y_center - 0.38),
                         arrowstyle='->', mutation_scale=13,
                         linewidth=1.1, color=BORDER)
ax.add_patch(a_bias)

# ---- summation node ----------------------------------------------------
sum_c = Circle((x_sum, y_center), 0.38, facecolor=SUM_FILL,
               edgecolor=BORDER, linewidth=1.4, zorder=3)
ax.add_patch(sum_c)
ax.text(x_sum, y_center, r'$\Sigma$', ha='center', va='center',
        fontsize=17, fontweight='bold', color=BORDER, zorder=4)
ax.text(x_sum, y_center - 0.85, 'weighted sum',
        ha='center', va='center', fontsize=9.5,
        style='italic', color=BORDER)

# arrow: summation -> activation (with intermediate "net input" label z)
a_sum_act = FancyArrowPatch((x_sum + 0.38, y_center), (x_act - 0.38, y_center),
                            arrowstyle='->', mutation_scale=14,
                            linewidth=1.2, color=BORDER)
ax.add_patch(a_sum_act)
ax.text((x_sum + x_act) / 2, y_center + 0.28, r'$z$',
        ha='center', va='center', fontsize=11, color=BORDER)

# ---- activation node ---------------------------------------------------
act_c = Circle((x_act, y_center), 0.38, facecolor=ACT_FILL,
               edgecolor=BORDER, linewidth=1.4, zorder=3)
ax.add_patch(act_c)
ax.text(x_act, y_center, r'$\sigma$', ha='center', va='center',
        fontsize=17, fontweight='bold', color=BORDER, zorder=4)
ax.text(x_act, y_center - 0.85, 'activation',
        ha='center', va='center', fontsize=9.5,
        style='italic', color=BORDER)

# arrow: activation -> output
a_act_out = FancyArrowPatch((x_act + 0.38, y_center), (x_out - 0.28, y_center),
                            arrowstyle='->', mutation_scale=14,
                            linewidth=1.2, color=BORDER)
ax.add_patch(a_act_out)

# ---- output node -------------------------------------------------------
out_c = Circle((x_out, y_center), 0.28, facecolor=OUT_FILL,
               edgecolor=BORDER, linewidth=1.4, zorder=3)
ax.add_patch(out_c)
ax.text(x_out + 0.55, y_center, r'$y$', ha='left', va='center',
        fontsize=13, fontweight='bold', color=BORDER)
ax.text(x_out, y_center - 0.85, 'output',
        ha='center', va='center', fontsize=9.5,
        style='italic', color=BORDER)

# ---- top formula banner ------------------------------------------------
banner = FancyBboxPatch((1.6, 5.05), 6.8, 0.75,
                        boxstyle='round,pad=0.02,rounding_size=0.14',
                        facecolor='#f4f6fb', edgecolor=BORDER, linewidth=1.0)
ax.add_patch(banner)
ax.text(5.0, 5.42,
        r'$y = \sigma\!\left(\sum_{i=1}^{n} w_i\, x_i + b\right)$',
        ha='center', va='center', fontsize=15, color=BORDER)

# ---- caption note under diagram ----------------------------------------
ax.text(5.0, 0.18,
        r'Hidden-layer neurons use $\sigma = \mathrm{tansig}$;  the '
        r'output neuron uses $\sigma = \mathrm{purelin}$ (linear).',
        ha='center', va='center', fontsize=10, style='italic', color=BORDER)

plt.tight_layout()
out_path = str(OUT_PNG)
plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f'Saved {out_path}')
