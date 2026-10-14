%% Day 3 - MPC that actually uses the ANN COP prediction
%
%  This replaces the paper's static-linearisation MPC with an adaptive MPC
%  scheme in which the trained ANN COP prediction updates the plant model
%  coefficient beta at every timestep. Four scenarios are simulated over
%  168 hours: baseline thermostat, healthy MPC, degraded MPC, and
%  maintenance-aware MPC.
%
%  Fixes addressed:
%    - ANN output is now a genuine input to the MPC control law (adaptive
%      beta). This addresses Gemini's "MPC does not use ANN" criticism.
%    - The plant used to simulate the room is a slightly perturbed version
%      of the MPC's internal model, avoiding "inverse crime".
%    - Multi-parameter degradation: beta and gamma both change under fault
%      (was single-parameter in the original paper).
%
%  Requirements: cold_storage_v2.csv, net_v2.mat in the same folder.
%                MATLAB Model Predictive Control Toolbox.
%
%  Output: results_v2.mat with all trajectories and performance metrics.

clear; clc; close all;

% ---- Portable path setup (script runnable from any folder) --------------
[script_dir, ~, ~] = fileparts(mfilename('fullpath'));
repo_root = fullfile(script_dir, '..', '..');
data_dir  = fullfile(repo_root, 'data');
fig_dir   = fullfile(repo_root, 'figures');
if ~exist(fig_dir, 'dir'); mkdir(fig_dir); end
cd(script_dir);   % .mat intermediates stay next to the scripts
% -------------------------------------------------------------------------


rng(2026);

%% 1. Load ANN and dataset
load('net_v2.mat', 'net');                       % ANN from Day 2
T_data = readtable(fullfile(data_dir, 'cold_storage_v2.csv'));
Qload  = T_data.CoolingLoad_kW;                  % 168x1 cooling-load disturbance
massFlow = T_data.MassFlowRate_kg_s;
cp       = T_data.SpecificHeatCapacity_kJ_kgC;

N_HOURS  = height(T_data);
Ts       = 1;                                    % 1-hour timestep
Tset     = -20;                                  % C setpoint

%% 2. Nominal plant model (identified at correct operating point)
%   T_{k+1} = alpha*T_k + beta*U_k + gamma*Q_k + b
%   Linearised at design setpoint T0 = -20, U0 = 5.0, Q0 = 5.0
T0       = -20.0;
U0       = 5.0;
Q0       = 5.0;
alpha    = 0.85;                                  % thermal inertia
beta0    = -0.60;                                 % compressor cooling coeff (healthy)
gamma0   = 0.25;                                  % heat load warming coeff
b        = T0*(1-alpha) - beta0*U0 - gamma0*Q0;   % analytic offset

fprintf('Nominal plant: alpha=%.3f, beta=%.3f, gamma=%.3f, b=%.3f\n', ...
        alpha, beta0, gamma0, b);
fprintf('Nominal point: T0=%.1f C, U0=%.2f kW, Q0=%.2f kW\n\n', T0, U0, Q0);

%% 3. Build linear MPC controller (deviation form around nominal point)
%   The A,B,C matrices are updated on each timestep with the current beta
A = alpha;
Bmpc = [beta0 gamma0];        % [manipulated, disturbance]
C = 1; D = [0 0];
plant_dev = ss(A, Bmpc, C, D, Ts);
plant_dev = setmpcsignals(plant_dev, 'MV', 1, 'MD', 2);

mpc_ctrl = mpc(plant_dev, Ts, 12, 3);            % pred horizon 12, ctrl horizon 3
mpc_ctrl.MV.Min      = 3 - U0;                    % deviation from nominal
mpc_ctrl.MV.Max      = 12 - U0;
mpc_ctrl.MV.RateMin  = -1.5;
mpc_ctrl.MV.RateMax  =  1.5;
mpc_ctrl.Weights.OV  = 10;
mpc_ctrl.Weights.MV  = 0.01;
mpc_ctrl.Weights.MVRate = 0.10;

mpc_state = mpcstate(mpc_ctrl);

%% 4. Health-monitoring configuration
COP_REF          = 1.35;                          % from Day 2 dataset mean
WARN_THRESHOLD   = 0.90;
CRIT_THRESHOLD   = 0.80;
PROTECT_TRIGGER_HRS = 3;                          % consecutive warning hrs
PROTECT_W_MAX    = 8.0;                           % kW cap in protective mode
PROTECT_RATE     = 0.5;                           % kW/hr rate cap in protective mode

%% 5. Run four scenarios
scenarios = {'Baseline', 'HealthyMPC', 'DegradedMPC', 'MaintAwareMPC'};

results = struct();

for s = 1:numel(scenarios)
    scen = scenarios{s};
    fprintf('=== Scenario: %s ===\n', scen);

    % ---------- degradation flags ----------
    is_degraded  = ismember(scen, {'DegradedMPC','MaintAwareMPC'});
    is_maint     = strcmp(scen, 'MaintAwareMPC');
    is_baseline  = strcmp(scen, 'Baseline');

    % ---------- plant coefficients (multi-parameter degradation) ----------
    beta_plant  = beta0;      % healthy plant compressor coefficient
    gamma_plant = gamma0;
    if is_degraded
        beta_plant  = beta0  * 0.75;     % 25% loss of cooling effectiveness
        gamma_plant = gamma0 * 1.10;     % 10% increase in disturbance sensitivity (fouling)
    end
    b_plant = T0*(1-alpha) - beta_plant*U0 - gamma_plant*Q0;

    % Small model-plant mismatch to avoid inverse crime
    alpha_plant = alpha * 1.02;   % plant thermal inertia slightly different
    b_plant     = b_plant * 1.01;

    % ---------- storage ----------
    Tk = T0;
    Uk = U0;
    warn_flags   = zeros(N_HOURS,1);
    crit_flags   = zeros(N_HOURS,1);
    mode_flag    = zeros(N_HOURS,1);   % 0 = normal, 1 = protective
    cop_ann      = zeros(N_HOURS,1);
    health_idx   = zeros(N_HOURS,1);
    T_traj       = zeros(N_HOURS,1);
    U_traj       = zeros(N_HOURS,1);
    warn_streak  = 0;

    % Reset MPC state
    mpc_state = mpcstate(mpc_ctrl);

    for k = 1:N_HOURS
        % ---------- ANN COP prediction from current operating state ----------
        ann_input = [Tk; massFlow(k); cp(k); Qload(k)];
        cop_k = net(ann_input);
        cop_ann(k) = cop_k;
        health_idx(k) = cop_k / COP_REF;

        % ---------- alarm flags ----------
        if health_idx(k) < WARN_THRESHOLD
            warn_flags(k) = 1;
            warn_streak = warn_streak + 1;
        else
            warn_streak = 0;
        end
        if health_idx(k) < CRIT_THRESHOLD
            crit_flags(k) = 1;
        end

        % ---------- controller ----------
        if is_baseline
            % on/off thermostat
            if Tk > -19.5
                Uk = 10.0;
            elseif Tk < -20.5
                Uk = 4.0;
            end
        else
            % ANN-adaptive MPC:
            % update beta in MPC using ANN COP prediction
            % expected relation: |beta| scales with expected cooling per unit power
            % (larger predicted COP => larger cooling delivered per kW => larger |beta|)
            beta_ann = -0.60 * (cop_k / COP_REF);
            beta_ann = max(min(beta_ann, -0.35), -0.75);   % clamp

            % rebuild MPC plant with new beta each timestep
            A_k = alpha;
            B_k = [beta_ann gamma0];
            plant_k = ss(A_k, B_k, C, D, Ts);
            plant_k = setmpcsignals(plant_k, 'MV', 1, 'MD', 2);
            mpc_ctrl.Model.Plant = plant_k;

            % maintenance-aware: switch to protective mode if warning persists
            if is_maint && warn_streak >= PROTECT_TRIGGER_HRS
                mode_flag(k) = 1;
                mpc_ctrl.MV.Max     = PROTECT_W_MAX - U0;
                mpc_ctrl.MV.RateMin = -PROTECT_RATE;
                mpc_ctrl.MV.RateMax =  PROTECT_RATE;
            else
                mpc_ctrl.MV.Max     = 12 - U0;
                mpc_ctrl.MV.RateMin = -1.5;
                mpc_ctrl.MV.RateMax =  1.5;
            end

            % solve MPC (deviation form)
            dT   = Tk - T0;
            dQ   = Qload(k) - Q0;
            try
                du = mpcmove(mpc_ctrl, mpc_state, dT, 0, dQ);
                Uk = U0 + du;
            catch
                % keep previous Uk if MPC fails
            end
        end

        Uk = max(min(Uk, 12), 3);

        % ---------- advance plant (with mismatch) ----------
        T_next = alpha_plant*Tk + beta_plant*Uk + gamma_plant*Qload(k) + b_plant;
        % small measurement/process noise so residuals aren't zero
        T_next = T_next + 0.02 * randn();

        T_traj(k) = Tk;
        U_traj(k) = Uk;
        Tk = T_next;
    end

    % ---------- metrics ----------
    mean_T_err   = mean(abs(T_traj - Tset));
    mean_power   = mean(U_traj);
    mean_cop     = mean(cop_ann);
    warn_hours   = sum(warn_flags);
    crit_hours   = sum(crit_flags);
    protect_hrs  = sum(mode_flag);

    results.(scen).T          = T_traj;
    results.(scen).U          = U_traj;
    results.(scen).cop_ann    = cop_ann;
    results.(scen).health     = health_idx;
    results.(scen).warn       = warn_flags;
    results.(scen).crit       = crit_flags;
    results.(scen).mode       = mode_flag;
    results.(scen).mean_T_err = mean_T_err;
    results.(scen).mean_power = mean_power;
    results.(scen).mean_cop   = mean_cop;
    results.(scen).warn_hours = warn_hours;
    results.(scen).crit_hours = crit_hours;
    results.(scen).protect_hrs = protect_hrs;

    fprintf('  Mean |T-Tset|  = %.4f C\n', mean_T_err);
    fprintf('  Mean power     = %.4f kW\n', mean_power);
    fprintf('  Mean ANN COP   = %.4f\n', mean_cop);
    fprintf('  Warning hours  = %d\n', warn_hours);
    fprintf('  Critical hours = %d\n', crit_hours);
    if is_maint
        fprintf('  Protective hrs = %d\n', protect_hrs);
    end
    fprintf('\n');
end

%% 6. Save
save('results_v2.mat', 'results', 'Qload', 'massFlow', 'cp', ...
     'N_HOURS', 'Ts', 'Tset', 'T0', 'U0', 'Q0', ...
     'alpha', 'beta0', 'gamma0', 'b', ...
     'COP_REF', 'WARN_THRESHOLD', 'CRIT_THRESHOLD');
fprintf('Saved to results_v2.mat\n');

%% 7. Quick comparison summary
fprintf('\n=================== Comparison summary ===================\n');
fprintf('%-18s | %-12s | %-14s | %-12s\n', 'Scenario', 'Mean |T-err|', 'Mean W (kW)', 'Mean COP');
fprintf('%s\n', repmat('-', 1, 70));
for s = 1:numel(scenarios)
    scen = scenarios{s};
    r = results.(scen);
    fprintf('%-18s | %-12.4f | %-14.4f | %-12.4f\n', ...
            scen, r.mean_T_err, r.mean_power, r.mean_cop);
end
fprintf('%s\n', repmat('=', 1, 70));

%% 8. Plots
figure('Position', [100 100 900 600]);
subplot(2,1,1); hold on; grid on;
plot(results.Baseline.T,      'DisplayName','Baseline',      'LineWidth', 1);
plot(results.HealthyMPC.T,    'DisplayName','Healthy MPC',   'LineWidth', 1);
plot(results.DegradedMPC.T,   'DisplayName','Degraded MPC',  'LineWidth', 1);
plot(results.MaintAwareMPC.T, 'DisplayName','Maint-Aware',   'LineWidth', 1);
yline(Tset, '--k', 'Setpoint');
xlabel('Hour'); ylabel('Room temperature (C)'); legend('Location','best');
title('Temperature trajectories across scenarios');

subplot(2,1,2); hold on; grid on;
plot(results.Baseline.U,      'DisplayName','Baseline',      'LineWidth', 1);
plot(results.HealthyMPC.U,    'DisplayName','Healthy MPC',   'LineWidth', 1);
plot(results.DegradedMPC.U,   'DisplayName','Degraded MPC',  'LineWidth', 1);
plot(results.MaintAwareMPC.U, 'DisplayName','Maint-Aware',   'LineWidth', 1);
xlabel('Hour'); ylabel('Compressor power (kW)'); legend('Location','best');
title('Compressor power trajectories');
saveas(gcf, fullfile(fig_dir, 'mpc_scenarios_v2.png'));

figure('Position', [100 100 900 400]);
plot(results.DegradedMPC.health, 'DisplayName','Degraded', 'LineWidth', 1); hold on;
plot(results.MaintAwareMPC.health, 'DisplayName','Maint-Aware', 'LineWidth', 1);
yline(WARN_THRESHOLD, '--', 'Warning (0.90)');
yline(CRIT_THRESHOLD, '--r', 'Critical (0.80)');
xlabel('Hour'); ylabel('Health index'); legend('Location','best');
title('ANN-based health index under degradation');
grid on; saveas(gcf, fullfile(fig_dir, 'health_index_v2.png'));
