%% Figure 1: Adaptive ANN, MPC, and predictive maintenance framework
%
%  Recreates the framework diagram from the paper in MATLAB.
%  No em dashes, no external image dependencies.
%  Saves fig01_framework.png at 300 dpi in the current folder.

clear; clc; close all;

% ---- Portable path setup (script runnable from any folder) --------------
[script_dir, ~, ~] = fileparts(mfilename('fullpath'));
repo_root = fullfile(script_dir, '..', '..');
data_dir  = fullfile(repo_root, 'data');
fig_dir   = fullfile(repo_root, 'figures');
if ~exist(fig_dir, 'dir'); mkdir(fig_dir); end
cd(script_dir);   % .mat intermediates stay next to the scripts
% -------------------------------------------------------------------------



fig = figure('Position', [100 100 1100 900], 'Color', 'w');
ax  = axes('Position', [0.03 0.03 0.94 0.94]);
axis(ax, [0 11 0 11]); axis off; hold on;

% Colour palette
C_PLANT   = [0.77 0.88 0.82];
C_SENSOR  = [0.80 0.87 0.94];
C_ANN     = [0.85 0.82 0.91];
C_MPC     = [1.00 0.85 0.72];
C_PDM     = [1.00 0.78 0.78];
C_ACT     = [0.88 0.88 0.88];
BORDER    = [0.20 0.24 0.33];

% =============== LEGEND (top of figure) ===============
legend_items = {
    'Plant / physical process', C_PLANT;
    'Sensor data',              C_SENSOR;
    'ANN prediction layer',     C_ANN;
    'MPC control layer',        C_MPC;
    'Predictive maintenance',   C_PDM;
    'Actuation',                C_ACT
};

for i = 1:size(legend_items,1)
    xx = 0.3 + mod(i-1, 3) * 3.4;
    yy = 10.55 - floor((i-1)/3) * 0.35;
    rectangle('Position', [xx yy-0.11 0.32 0.22], 'Curvature', 0.3, ...
              'FaceColor', legend_items{i,2}, 'EdgeColor', BORDER, ...
              'LineWidth', 0.8);
    text(xx + 0.42, yy, legend_items{i,1}, ...
         'FontSize', 9, 'VerticalAlignment', 'middle', 'Color', BORDER);
end

plot([0.3 10.7], [9.95 9.95], 'Color', [0.6 0.6 0.6], 'LineWidth', 0.5);

% Solid / dashed arrow legend
plot([0.4 0.8], [9.60 9.60], '-', 'Color', BORDER, 'LineWidth', 1.2);
text(0.9, 9.60, 'Data flow (solid arrow)', 'FontSize', 9, ...
     'VerticalAlignment', 'middle', 'Color', BORDER);
plot([5.0 5.4], [9.60 9.60], '--', 'Color', BORDER, 'LineWidth', 1.2);
text(5.5, 9.60, 'Adaptive update (dashed arrow)', 'FontSize', 9, ...
     'VerticalAlignment', 'middle', 'Color', BORDER);

% =============== BOXES ===============
draw_box(3.4, 8.2, 4.2, 1.0, 'Walk-in cold storage plant', ...
         'R-404A primary + glycol secondary loop', C_PLANT, BORDER);

draw_box(3.4, 6.8, 4.2, 0.9, 'Sensor layer', ...
         'T, secondary-loop mass flow, Cp, cooling load', C_SENSOR, BORDER);
draw_arrow(5.5, 8.2, 5.5, 7.7, '', '-', BORDER);

draw_box(3.4, 5.3, 4.2, 1.0, 'ANN prediction (4 to 4 to 1)', ...
         '25 parameters, Bayesian regularisation', C_ANN, BORDER);
draw_arrow(5.5, 6.8, 5.5, 6.3, '', '-', BORDER);

draw_box(0.4, 3.3, 3.8, 1.4, 'Adaptive MPC controller', ...
         'Beta_k updates from ANN COP every hour', C_MPC, BORDER);
draw_arrow(4.4, 5.5, 4.2, 4.5, 'predicted COP', '-', BORDER);

draw_box(6.8, 3.3, 3.8, 1.4, 'Health monitor', ...
         'H = COP_pred / COP_ref;  warn, critical, RUL', C_PDM, BORDER);
draw_arrow(6.6, 5.5, 6.8, 4.5, 'COP trend', '-', BORDER);

draw_box(0.4, 1.4, 3.8, 1.0, 'Compressor actuator', ...
         'Power in 3 to 12 kW, rate limited', C_ACT, BORDER);
draw_arrow(2.3, 3.3, 2.3, 2.4, '', '-', BORDER);

draw_box(6.8, 1.4, 3.8, 1.0, 'Mode switch', ...
         'Normal or Protective (8 kW cap, 0.5 kW/h rate)', C_ACT, BORDER);
draw_arrow(8.7, 3.3, 8.7, 2.4, '', '-', BORDER);

% Actuator feedback path along the bottom
draw_arrow(2.3, 1.4, 2.3, 0.5, '', '-', BORDER);
draw_arrow(2.3, 0.5, 5.5, 0.5, '', '-', BORDER);
draw_arrow(5.5, 0.5, 5.5, 1.4, 'W\_comp', '-', BORDER);

% Adaptive constraint update from Mode switch to MPC (dashed)
draw_arrow(6.8, 1.9, 4.2, 1.9, 'constraint update', '--', BORDER);

% =============== SAVE ===============
set(gcf, 'PaperPositionMode', 'auto');
exportgraphics(gcf, fullfile(fig_dir, 'fig01_framework.png'), 'Resolution', 300);
fprintf('Saved: fig01_framework.png\n');


%% =============== LOCAL FUNCTIONS (must appear at end of script) ==========

function draw_box(x, y, w, h, ttl, sub, fill, border)
    rectangle('Position', [x y w h], 'Curvature', 0.18, ...
              'FaceColor', fill, 'EdgeColor', border, 'LineWidth', 1.2);
    text(x + w/2, y + 0.62*h, ttl, ...
         'HorizontalAlignment','center','VerticalAlignment','middle', ...
         'FontSize', 11, 'FontWeight', 'bold', 'Color', border);
    text(x + w/2, y + 0.28*h, sub, ...
         'HorizontalAlignment','center','VerticalAlignment','middle', ...
         'FontSize', 9, 'FontAngle', 'italic', 'Color', border);
end


function draw_arrow(x1, y1, x2, y2, lbl, style, border)
    plot([x1 x2], [y1 y2], style, 'Color', border, 'LineWidth', 1.1);
    dx = x2 - x1; dy = y2 - y1;
    L  = sqrt(dx^2 + dy^2);
    if L == 0, return; end
    ux = dx / L; uy = dy / L;
    head = 0.13;
    base = 0.055;
    px = [x2, ...
          x2 - head*ux - base*(-uy), ...
          x2 - head*ux + base*(-uy)];
    py = [y2, ...
          y2 - head*uy - base*( ux), ...
          y2 - head*uy + base*( ux)];
    patch(px, py, border, 'EdgeColor', border);
    if ~isempty(lbl)
        text(0.5*(x1+x2) + 0.15, 0.5*(y1+y2) + 0.05, lbl, ...
             'FontSize', 8.5, 'FontAngle', 'italic', 'Color', border);
    end
end
