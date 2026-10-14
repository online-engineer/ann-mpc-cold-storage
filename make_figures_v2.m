%% Day 4 - Publication-quality figures and metric tables for the paper
%
%  Reads: cold_storage_v2.csv, net_v2.mat, results_v2.mat
%  Writes: fig02_...fig18_*.png (300 dpi PNGs), plus a text summary of
%          every number needed for Tables 6, 7 and the manuscript prose.
%
%  Naming convention keeps figure numbers aligned with the paper's
%  section 4 order (Results and Discussion).

clear; clc; close all;

% ---- Portable path setup (script runnable from any folder) --------------
[script_dir, ~, ~] = fileparts(mfilename('fullpath'));
repo_root = fullfile(script_dir, '..', '..');
data_dir  = fullfile(repo_root, 'data');
fig_dir   = fullfile(repo_root, 'figures');
if ~exist(fig_dir, 'dir'); mkdir(fig_dir); end
cd(script_dir);   % .mat intermediates stay next to the scripts
% -------------------------------------------------------------------------



%% --- Load everything ---
T = readtable(fullfile(data_dir, 'cold_storage_v2.csv'));
load('results_v2.mat');
load('net_v2.mat', 'net', 'fold_R2', 'fold_RMSE', 'fold_MAE', ...
     'R_train', 'R_test', 'R_all', 'MSE_train', 'MSE_test', 'Yhat', 'Y', 'tr');

% Consistent styling
set(groot, 'defaultAxesFontSize', 11);
set(groot, 'defaultAxesFontName', 'Arial');
set(groot, 'defaultLineLineWidth', 1.3);
set(groot, 'defaultFigureColor', 'w');

DPI = 300;
save_fig = @(name) exportgraphics(gcf, fullfile(fig_dir, name), 'Resolution', DPI);

%% =====================================================================
%% Figure 2: Dataset temperature profile
%% =====================================================================
figure('Position', [100 100 700 350]); hold on; grid on;
plot(T.Hour, T.Temperature_C, 'Color', [0.15 0.35 0.6], 'LineWidth', 1.3);
yline(-20, '--k', 'Setpoint (-20 C)', 'LabelHorizontalAlignment', 'left');
xlabel('Hour'); ylabel('Room temperature (\circC)');
title('Temperature profile across 168-hour dataset');
xlim([0 168]); ylim([-22 -17]);
save_fig('fig02_dataset_temperature.png');

%% =====================================================================
%% Figure 3: Dataset COP profile
%% =====================================================================
figure('Position', [100 100 700 350]); hold on; grid on;
plot(T.Hour, T.COP, 'Color', [0.13 0.48 0.32], 'LineWidth', 1.3);
yline(mean(T.COP), '--k', ...
      sprintf('Mean = %.2f', mean(T.COP)), 'LabelHorizontalAlignment', 'left');
xlabel('Hour'); ylabel('COP');
title('Coefficient of performance across 168-hour dataset');
xlim([0 168]);
save_fig('fig03_dataset_cop.png');

%% =====================================================================
%% Figure 6: ANN training state plot
%% =====================================================================
figure('Position', [100 100 700 500]);
plotperform(tr);
save_fig('fig06_training_performance.png');

%% =====================================================================
%% Figure 7: ANN regression (4-panel: train, val, test, all)
%% =====================================================================
figure('Position', [100 100 800 700]);
train_idx = tr.trainInd; test_idx = tr.testInd;
subplot(2,2,1); plotregression(Y(train_idx), Yhat(train_idx), 'Training');
subplot(2,2,2); plotregression(Y(test_idx),  Yhat(test_idx),  'Test');
subplot(2,2,3); bar(fold_R2); grid on;
   xlabel('Fold'); ylabel('Out-of-sample R^2');
   title('5-fold cross-validation'); ylim([0.5 1]);
subplot(2,2,4); plotregression(Y, Yhat, 'All data');
save_fig('fig07_regression.png');

%% =====================================================================
%% Figure 8: Error histogram
%% =====================================================================
figure('Position', [100 100 700 400]);
err = Y - Yhat;
histogram(err, 20, 'FaceColor', [0.65 0.2 0.3], 'EdgeColor', 'w'); grid on;
xlabel('Prediction error (COP units)'); ylabel('Count');
title(sprintf('ANN error histogram (mean = %.4f, std = %.4f)', mean(err), std(err)));
xline(0, '--k', 'LineWidth', 1);
save_fig('fig08_error_histogram.png');

%% =====================================================================
%% Figure 10: Temperature trajectories across all four scenarios
%% =====================================================================
figure('Position', [100 100 900 400]); hold on; grid on;
h1 = plot(1:N_HOURS, results.Baseline.T,      'LineWidth', 1.1);
h2 = plot(1:N_HOURS, results.HealthyMPC.T,    'LineWidth', 1.3);
h3 = plot(1:N_HOURS, results.DegradedMPC.T,   'LineWidth', 1.3);
h4 = plot(1:N_HOURS, results.MaintAwareMPC.T, 'LineWidth', 1.3);
yline(Tset, '--k', 'Setpoint');
xlabel('Hour'); ylabel('Room temperature (\circC)');
legend([h1 h2 h3 h4], {'Baseline thermostat', 'Healthy MPC', ...
        'Degraded MPC', 'Maintenance-aware MPC'}, 'Location', 'best');
title('Temperature trajectories across control scenarios');
xlim([0 168]);
save_fig('fig10_temperature_comparison.png');

%% =====================================================================
%% Figure 11: Compressor power trajectories
%% =====================================================================
figure('Position', [100 100 900 400]); hold on; grid on;
plot(1:N_HOURS, results.Baseline.U,      'LineWidth', 1.1);
plot(1:N_HOURS, results.HealthyMPC.U,    'LineWidth', 1.3);
plot(1:N_HOURS, results.DegradedMPC.U,   'LineWidth', 1.3);
plot(1:N_HOURS, results.MaintAwareMPC.U, 'LineWidth', 1.3);
xlabel('Hour'); ylabel('Compressor power (kW)');
legend({'Baseline thermostat', 'Healthy MPC', 'Degraded MPC', ...
        'Maintenance-aware MPC'}, 'Location', 'best');
title('Compressor power trajectories across control scenarios');
xlim([0 168]);
save_fig('fig11_power_comparison.png');

%% =====================================================================
%% Figure 12: ANN-estimated COP trajectories
%% =====================================================================
figure('Position', [100 100 900 400]); hold on; grid on;
plot(1:N_HOURS, results.Baseline.cop_ann,      'LineWidth', 1.1);
plot(1:N_HOURS, results.HealthyMPC.cop_ann,    'LineWidth', 1.3);
plot(1:N_HOURS, results.DegradedMPC.cop_ann,   'LineWidth', 1.3);
plot(1:N_HOURS, results.MaintAwareMPC.cop_ann, 'LineWidth', 1.3);
yline(COP_REF, '--k', 'Healthy reference');
xlabel('Hour'); ylabel('ANN-predicted COP');
legend({'Baseline thermostat', 'Healthy MPC', 'Degraded MPC', ...
        'Maintenance-aware MPC'}, 'Location', 'best');
title('ANN-estimated COP across scenarios');
xlim([0 168]);
save_fig('fig12_cop_comparison.png');

%% =====================================================================
%% Figure 13: Health index
%% =====================================================================
figure('Position', [100 100 900 400]); hold on; grid on;
plot(1:N_HOURS, results.DegradedMPC.health,   'LineWidth', 1.3, ...
     'DisplayName', 'Degraded MPC');
plot(1:N_HOURS, results.MaintAwareMPC.health, 'LineWidth', 1.3, ...
     'DisplayName', 'Maintenance-aware MPC');
yline(WARN_THRESHOLD, '--', sprintf('Warning (%.2f)', WARN_THRESHOLD), ...
      'Color', [0.85 0.55 0], 'LineWidth', 1.2);
yline(CRIT_THRESHOLD, '--', sprintf('Critical (%.2f)', CRIT_THRESHOLD), ...
      'Color', [0.8 0.1 0.1], 'LineWidth', 1.2);
xlabel('Hour'); ylabel('Health index (H)');
legend('Location', 'best');
title('ANN-based health index under simulated degradation');
xlim([0 168]); ylim([0.5 1.2]);
save_fig('fig13_health_index.png');

%% =====================================================================
%% Figure 14: COP-based performance degradation
%% =====================================================================
figure('Position', [100 100 900 400]); hold on; grid on;
deg_pct_d = (COP_REF - results.DegradedMPC.cop_ann)   / COP_REF * 100;
deg_pct_m = (COP_REF - results.MaintAwareMPC.cop_ann) / COP_REF * 100;
plot(1:N_HOURS, deg_pct_d, 'LineWidth', 1.3, 'DisplayName', 'Degraded MPC');
plot(1:N_HOURS, deg_pct_m, 'LineWidth', 1.3, 'DisplayName', 'Maintenance-aware MPC');
yline(0, '--k');
xlabel('Hour'); ylabel('Performance degradation (%)');
legend('Location', 'best');
title(sprintf('COP-based performance degradation (mean deg = %.1f%%)', ...
              mean(deg_pct_d)));
xlim([0 168]);
save_fig('fig14_degradation.png');

%% =====================================================================
%% Figure 15: Warning and critical alarm flags (stacked)
%% =====================================================================
figure('Position', [100 100 900 500]);
subplot(2,1,1);
bar(1:N_HOURS, results.DegradedMPC.warn, 'FaceColor', [0.9 0.6 0.2], 'EdgeColor', 'none');
hold on;
bar(1:N_HOURS, results.DegradedMPC.crit * -1, 'FaceColor', [0.85 0.15 0.15], 'EdgeColor', 'none');
ylim([-1.3 1.3]);
yticks([-1 0 1]); yticklabels({'Critical', 'Off', 'Warning'});
xlabel('Hour'); title('Degraded MPC: alarm state timeline');
grid on; xlim([0 168]);

subplot(2,1,2);
bar(1:N_HOURS, results.MaintAwareMPC.warn, 'FaceColor', [0.9 0.6 0.2], 'EdgeColor', 'none');
hold on;
bar(1:N_HOURS, results.MaintAwareMPC.crit * -1, 'FaceColor', [0.85 0.15 0.15], 'EdgeColor', 'none');
ylim([-1.3 1.3]);
yticks([-1 0 1]); yticklabels({'Critical', 'Off', 'Warning'});
xlabel('Hour'); title('Maintenance-aware MPC: alarm state timeline');
grid on; xlim([0 168]);
save_fig('fig15_alarm_flags.png');

%% =====================================================================
%% Figure 16: Mode activation timeline (maintenance-aware)
%% =====================================================================
figure('Position', [100 100 900 300]); hold on; grid on;
stairs(1:N_HOURS, results.MaintAwareMPC.mode, 'LineWidth', 1.6, ...
       'Color', [0.5 0.15 0.55]);
ylim([-0.3 1.3]); yticks([0 1]); yticklabels({'Normal', 'Protective'});
xlabel('Hour'); ylabel('Operating mode');
title('Maintenance-aware MPC operating mode timeline');
xlim([0 168]);
save_fig('fig16_mode_activation.png');

%% =====================================================================
%% Figure 17: Cooling load disturbance profile
%% =====================================================================
figure('Position', [100 100 900 350]); hold on; grid on;
plot(1:N_HOURS, Qload, 'LineWidth', 1.2, 'Color', [0.3 0.4 0.5]);
yline(mean(Qload), '--k', sprintf('Mean = %.2f kW', mean(Qload)));
xlabel('Hour'); ylabel('Cooling load Q_L (kW)');
title('External cooling load disturbance profile');
xlim([0 168]);
save_fig('fig17_cooling_load.png');

%% =====================================================================
%% Figure 18: Bar chart comparison across all four scenarios
%% =====================================================================
figure('Position', [100 100 900 450]);
metrics = [results.Baseline.mean_power,       results.HealthyMPC.mean_power, ...
           results.DegradedMPC.mean_power,    results.MaintAwareMPC.mean_power;
           results.Baseline.mean_cop,         results.HealthyMPC.mean_cop, ...
           results.DegradedMPC.mean_cop,      results.MaintAwareMPC.mean_cop;
           results.Baseline.mean_T_err,       results.HealthyMPC.mean_T_err, ...
           results.DegradedMPC.mean_T_err,    results.MaintAwareMPC.mean_T_err];

b = bar(metrics, 'grouped');
b(1).FaceColor = [0.55 0.55 0.55];   % Baseline: grey
b(2).FaceColor = [0.20 0.55 0.30];   % Healthy: green
b(3).FaceColor = [0.85 0.55 0.15];   % Degraded: orange
b(4).FaceColor = [0.55 0.25 0.55];   % MaintAware: purple

set(gca, 'XTickLabel', {'Avg Power (kW)', 'Avg COP', 'Mean |T-Tset| (\circC)'});
legend({'Baseline', 'Healthy MPC', 'Degraded MPC', 'Maintenance-aware MPC'}, ...
       'Location', 'northwest');
ylabel('Value'); grid on;
title('Integrated performance comparison across control strategies');
save_fig('fig18_summary_bars.png');

%% =====================================================================
%% METRICS SUMMARY (for direct paste into the paper)
%% =====================================================================
diary('metrics_for_paper.txt'); diary on;

fprintf('\n\n============ METRICS FOR PAPER (v2) ============\n\n');

fprintf('---- Table 1: dataset descriptive statistics ----\n');
fprintf('%-32s %10s %10s %10s %10s\n', 'Parameter', 'Min', 'Max', 'Mean', 'Std');
cols = {'Temperature_C', 'MassFlowRate_kg_s', 'SpecificHeatCapacity_kJ_kgC', ...
        'CoolingLoad_kW', 'CompressorPower_kW', 'COP'};
lbls = {'Temperature (C)', 'Mass flow (kg/s)', 'Specific heat (kJ/kg.C)', ...
        'Cooling load (kW)', 'Compressor power (kW)', 'COP'};
for i = 1:numel(cols)
    v = T.(cols{i});
    fprintf('%-32s %10.3f %10.3f %10.3f %10.3f\n', lbls{i}, ...
            min(v), max(v), mean(v), std(v));
end

fprintf('\n---- Table 6: ANN training results summary (v2) ----\n');
fprintf('%-14s %-14s %-10s %-10s\n', 'Dataset', 'Observations', 'MSE', 'R');
fprintf('%-14s %-14d %-10.5f %-10.4f\n', 'Training',  numel(train_idx), MSE_train, R_train);
fprintf('%-14s %-14d %-10.5f %-10.4f\n', 'Test',      numel(test_idx),  MSE_test,  R_test);
fprintf('%-14s %-14d %-10s %-10.4f\n',   'All data',  numel(Y),         '-',       R_all);
fprintf('\n5-fold cross-validation summary (out-of-sample):\n');
fprintf('  Mean R^2  = %.4f  (std %.4f)\n', mean(fold_R2),   std(fold_R2));
fprintf('  Mean RMSE = %.4f  (std %.4f)\n', mean(fold_RMSE), std(fold_RMSE));
fprintf('  Mean MAE  = %.4f  (std %.4f)\n', mean(fold_MAE),  std(fold_MAE));

fprintf('\n---- Table 7: Integrated performance summary (v2) ----\n');
sc = {'Baseline', 'HealthyMPC', 'DegradedMPC', 'MaintAwareMPC'};
sc_lbl = {'Baseline', 'Healthy MPC', 'Degraded MPC', 'Maint-Aware MPC'};

fprintf('%-30s', 'Metric');
for i = 1:4, fprintf('%-16s', sc_lbl{i}); end
fprintf('\n');

fprintf('%-30s', 'Mean |T-err| (C)');
for i = 1:4, fprintf('%-16.4f', results.(sc{i}).mean_T_err); end
fprintf('\n');

fprintf('%-30s', 'Avg Compressor Power (kW)');
for i = 1:4, fprintf('%-16.4f', results.(sc{i}).mean_power); end
fprintf('\n');

fprintf('%-30s', 'Avg ANN COP');
for i = 1:4, fprintf('%-16.4f', results.(sc{i}).mean_cop); end
fprintf('\n');

fprintf('%-30s', 'Warning Duration (hrs)');
for i = 1:4, fprintf('%-16d', results.(sc{i}).warn_hours); end
fprintf('\n');

fprintf('%-30s', 'Critical Duration (hrs)');
for i = 1:4, fprintf('%-16d', results.(sc{i}).crit_hours); end
fprintf('\n');

fprintf('%-30s', 'Protective Duration (hrs)');
for i = 1:4
    if isfield(results.(sc{i}), 'protect_hrs')
        fprintf('%-16d', results.(sc{i}).protect_hrs);
    else
        fprintf('%-16s', '-');
    end
end
fprintf('\n');

fprintf('\n---- Derived improvement percentages ----\n');
base_T = results.Baseline.mean_T_err;
base_W = results.Baseline.mean_power;
base_C = results.Baseline.mean_cop;
for i = 2:4
    r = results.(sc{i});
    fprintf('%s vs baseline:\n', sc_lbl{i});
    fprintf('  Temp regulation improvement: %.2f%%\n', ...
            (base_T - r.mean_T_err) / base_T * 100);
    fprintf('  Power change:                %+.2f%%\n', ...
            (r.mean_power - base_W) / base_W * 100);
    fprintf('  COP improvement:             %.2f%%\n', ...
            (r.mean_cop - base_C) / base_C * 100);
end

fprintf('\n---- Time-to-first-protective-mode (MaintAware) ----\n');
first_prot = find(results.MaintAwareMPC.mode == 1, 1);
if isempty(first_prot)
    fprintf('  Protective mode never activated\n');
else
    fprintf('  First activation at hour %d\n', first_prot);
    fprintf('  Total protective hours:   %d of %d\n', ...
            sum(results.MaintAwareMPC.mode), N_HOURS);
end

fprintf('\n---- Mean COP-based degradation across the operating window ----\n');
fprintf('  Mean degradation (Degraded MPC):   %.2f%%\n', mean(deg_pct_d));
fprintf('  Mean degradation (Maint-Aware):    %.2f%%\n', mean(deg_pct_m));

diary off;

fprintf('\n>>> All figures saved to current folder (fig02*..fig18*.png).\n');
fprintf('>>> Numeric summary written to metrics_for_paper.txt\n');
