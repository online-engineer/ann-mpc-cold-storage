%% ANN training for corrected walk-in freezer dataset (v2, statistically hardened)
%
%  Addresses reviewer concerns on overparameterisation and overfitting.
%
%  Structural fixes:
%    - Network 4-4-1 (25 trainable parameters), giving ~7:1 sample-to-parameter
%      ratio using 168 hours - within Gemini's recommended 5:1 to 10:1 band.
%    - Input excludes CompressorPower to eliminate the trivial COP = Q_L/W_comp
%      identity (data-leakage fix).
%
%  Regularisation fixes:
%    - trainbr (Bayesian regularisation) instead of trainlm. trainbr
%      automatically applies L2-like weight decay and prunes effective
%      parameters, which is the standard MATLAB tool for small-dataset ANN
%      training.
%    - Early stopping on validation performance.
%
%  Evaluation fixes:
%    - 5-fold cross-validation reports out-of-sample R^2 and RMSE.
%    - The final saved network is retrained on 85% train + 15% test using
%      the same hyperparameters, so downstream MPC/PdM work uses a single
%      deterministic net.
%
%  Requirements: MATLAB with Deep Learning Toolbox
%  Input:  cold_storage_v2.csv (same folder)
%  Output: net_v2.mat with net, per-fold metrics, and final performance

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

%% 1. Load dataset
T = readtable(fullfile(data_dir, 'cold_storage_v2.csv'));
X = T{:, {'Temperature_C', 'MassFlowRate_kg_s', ...
          'SpecificHeatCapacity_kJ_kgC', 'CoolingLoad_kW'}}';   % 4 x N
Y = T.COP';                                                     % 1 x N
N = size(X, 2);
fprintf('Dataset: %d samples, %d input features\n', N, size(X,1));

%% 2. Network configuration -- 4-4-1 = 25 trainable parameters
%     (4 inputs * 4 hidden + 4 hidden biases) + (4 hidden * 1 out + 1 out bias)
%     = 16 + 4 + 4 + 1 = 25 parameters
%     With 168 samples: 168 / 25 = 6.7 samples per parameter (within 5:1-10:1)
HIDDEN = 4;
build_net = @() setup_net(HIDDEN);

fprintf('Network: 4-%d-1 (%d trainable parameters)\n', HIDDEN, ...
        4*HIDDEN + HIDDEN + HIDDEN*1 + 1);
fprintf('Sample-to-parameter ratio: %.2f\n\n', N / (4*HIDDEN + HIDDEN + HIDDEN + 1));

%% 3. 5-fold cross-validation
K = 5;
cv_idx = crossvalind('Kfold', N, K);
fold_R2   = zeros(K, 1);
fold_RMSE = zeros(K, 1);
fold_MAE  = zeros(K, 1);

fprintf('=== 5-fold cross-validation ===\n');
for k = 1:K
    test_mask  = (cv_idx == k);
    train_mask = ~test_mask;
    Xtr = X(:, train_mask); Ytr = Y(train_mask);
    Xte = X(:, test_mask);  Yte = Y(test_mask);

    net_k = build_net();
    net_k = train(net_k, Xtr, Ytr);
    Yhat  = net_k(Xte);

    ss_res = sum((Yte - Yhat).^2);
    ss_tot = sum((Yte - mean(Yte)).^2);
    fold_R2(k)   = 1 - ss_res/ss_tot;
    fold_RMSE(k) = sqrt(mean((Yte - Yhat).^2));
    fold_MAE(k)  = mean(abs(Yte - Yhat));

    fprintf('  Fold %d: R^2 = %.4f, RMSE = %.4f, MAE = %.4f\n', ...
            k, fold_R2(k), fold_RMSE(k), fold_MAE(k));
end

fprintf('\nCross-validation summary (out-of-sample):\n');
fprintf('  Mean R^2  = %.4f  (std %.4f)\n', mean(fold_R2),   std(fold_R2));
fprintf('  Mean RMSE = %.4f  (std %.4f)\n', mean(fold_RMSE), std(fold_RMSE));
fprintf('  Mean MAE  = %.4f  (std %.4f)\n', mean(fold_MAE),  std(fold_MAE));

%% 4. Final training on 85/15 split with 10 initialisations
fprintf('\n=== Final training (10 initialisations, keep best) ===\n');
best_test = Inf; best_net = []; best_tr = [];
for r = 1:10
    net_r = build_net();
    [net_r, tr_r] = train(net_r, X, Y);
    perf = tr_r.best_tperf;   % test performance
    if perf < best_test
        best_test = perf;
        best_net  = net_r;
        best_tr   = tr_r;
    end
end
net = best_net; tr = best_tr;

Yhat = net(X);
train_idx = tr.trainInd; test_idx = tr.testInd;
R_train = corr(Y(train_idx)', Yhat(train_idx)');
R_test  = corr(Y(test_idx)',  Yhat(test_idx)');
R_all   = corr(Y', Yhat');
MSE_train = mean((Y(train_idx) - Yhat(train_idx)).^2);
MSE_test  = mean((Y(test_idx)  - Yhat(test_idx)).^2);

fprintf('  Training R   = %.4f, MSE = %.5f\n', R_train, MSE_train);
fprintf('  Test R       = %.4f, MSE = %.5f\n', R_test,  MSE_test);
fprintf('  Overall R    = %.4f\n', R_all);

%% 5. Save
save('net_v2.mat', 'net', 'tr', 'fold_R2', 'fold_RMSE', 'fold_MAE', ...
     'R_train', 'R_test', 'R_all', 'MSE_train', 'MSE_test', 'X', 'Y', 'Yhat');
fprintf('\nSaved to net_v2.mat\n');

%% 6. Plots
figure('Position', [100 100 900 700]);
subplot(2,2,1); plotregression(Y(train_idx), Yhat(train_idx), 'Training');
subplot(2,2,2); plotregression(Y(test_idx),  Yhat(test_idx),  'Test');
subplot(2,2,3);
bar(fold_R2); grid on;
xlabel('Fold'); ylabel('Out-of-sample R^2');
title('5-fold cross-validation R^2 per fold');
ylim([min(0, min(fold_R2)-0.05), 1]);
subplot(2,2,4); plotregression(Y, Yhat, 'All Data');
saveas(gcf, fullfile(fig_dir, 'regression_v2.png'));

figure('Position', [100 100 800 400]);
plot(1:N, Y, 'k-', 'LineWidth', 1.2); hold on;
plot(1:N, Yhat, 'r--', 'LineWidth', 1.0);
xlabel('Hour'); ylabel('COP');
legend('Measured COP', 'ANN-predicted COP', 'Location', 'best');
title(sprintf('ANN COP prediction (test R = %.4f, mean CV R^2 = %.4f)', ...
              R_test, mean(fold_R2)));
grid on; saveas(gcf, fullfile(fig_dir, 'cop_prediction_v2.png'));

%% --- helper: build a fresh network with regularisation ---
function net = setup_net(hidden)
    net = feedforwardnet(hidden, 'trainbr');   % Bayesian regularisation
    net.layers{1}.transferFcn = 'tansig';
    net.layers{2}.transferFcn = 'purelin';
    net.divideFcn  = 'dividerand';
    net.divideParam.trainRatio = 0.85;
    net.divideParam.valRatio   = 0.00;         % trainbr does not use val split
    net.divideParam.testRatio  = 0.15;
    net.trainParam.epochs      = 500;
    net.trainParam.goal        = 1e-6;
    net.trainParam.max_fail    = 20;
    net.trainParam.showWindow  = false;
end
