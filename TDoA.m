%% TDoA EKF Localization

clear; clc; close all;
load('LNSM_Project_Data.mat');

i = 1; % scenario 1, 2, or 3

% Extract Data
tdoa_raw = TDoA{i};       % [10 x T]
truth = ground_truth{i};  % [3 x T]
APs = AP;                 % [3 x 10]
c = 3e8;                  % speed of light [m/s]

T = size(tdoa_raw, 2);
tdoa_master_idx = round(tdoa_raw(end, :)); % 1 x T vector of master indices (AP numbers)
tdoa_data = tdoa_raw(1:end-1, :);          % 9 x T (TDoA measurements)

% Use only x-y from ground truth
x_gt = truth(1, :);
y_gt = truth(2, :);

%% EKF initialization
dt = 0.1;  % 10 Hz
x_hat = zeros(4, T);       % state: [x, y, vx, vy]
P = zeros(4, 4, T);        % covariance matrices

% Initial state estimate from ground truth
x_hat(:,1) = [x_gt(1); y_gt(1); 0; 0];
P(:,:,1) = diag([10, 10, 5, 5]);

% === Motion Model ===
F = [1 , 0 , dt , 0; 
     0 , 1 , 0 , dt;
     0 , 0 , 1 , 0;
     0 , 0 , 0 , 1];

% Process noise model
sigma_tdoa = 3; 
L = [0.5*dt^2,      0;
         0, 0.5*dt^2;
        dt,      0;
         0,     dt];

Q = sigma_tdoa^2 * (L * L');

R_base = sigma_tdoa^2;

all_aps = 1:10; % total AP indices




%% EKF Loop
for k = 2:T
    x_pred = F * x_hat(:,k-1);
    P_pred = F * P(:,:,k-1) * F' + Q;

    tdoa_k = tdoa_data(:, k);
    master_idx = tdoa_master_idx(k);
    other_aps = setdiff(all_aps, master_idx);

    z = [];
    H = [];

    for row_idx = 1:length(other_aps)
        ap_idx = other_aps(row_idx);
        tdoa_meas = tdoa_k(row_idx);
        if isnan(tdoa_meas), continue; end

        ap_pos = APs(1:2, ap_idx);
        master_pos = APs(1:2, master_idx);

        dx_ap = x_pred(1) - ap_pos(1);
        dy_ap = x_pred(2) - ap_pos(2);
        dx_master = x_pred(1) - master_pos(1);
        dy_master = x_pred(2) - master_pos(2);

        d_ap = sqrt(dx_ap^2 + dy_ap^2);
        d_master = sqrt(dx_master^2 + dy_master^2);
        h_i = d_ap - d_master;

        H_i = [(dx_ap/d_ap - dx_master/d_master), ...
               (dy_ap/d_ap - dy_master/d_master), 0, 0];

        residual = tdoa_meas - h_i;
        z = [z; residual];
        H = [H; H_i];
    end

    if isempty(z)
        x_hat(:,k) = x_pred;
        P(:,:,k) = P_pred;
        continue;
    end

    R = R_base * eye(length(z));
    S = H * P_pred * H' + R;
    K = P_pred * H' / S;

    x_hat(:,k) = x_pred + K * z;
    P(:,:,k) = (eye(4) - K * H) * P_pred;
end

%% Plot EKF vs Ground Truth
figure;
plot(x_gt, y_gt, 'b-', 'LineWidth', 1.5); hold on;
plot(x_hat(1,:), x_hat(2,:), 'r--', 'LineWidth', 1.5);
plot(APs(1,:), APs(2,:), 'k^', 'MarkerSize', 8, 'LineWidth', 1.5);
xlabel('X [m]'); ylabel('Y [m]');
title('TDoA EKF Localization');
legend('Ground Truth', 'EKF Estimate', 'APs');
axis equal; grid on;

%% Position Error Plot
pos_err = sqrt((x_hat(1,:) - x_gt).^2 + (x_hat(2,:) - y_gt).^2);
figure;
plot((1:T)*dt, pos_err, 'LineWidth', 1.5);
xlabel('Time [s]');
ylabel('Position Error [m]');
title('TDoA EKF Position Error Over Time');
grid on;

%% CDF of Position Error
figure;
cdfplot(pos_err);
grid on;
xlabel('Position Error [m]');
title('CDF of Position Error (TDoA EKF)');

%% RMSE
valid_idx = ~any(isnan(x_hat(1:2,:)), 1);
tdoa_rmse = sqrt(mean((x_hat(1,valid_idx) - x_gt(valid_idx)).^2 + ...
                      (x_hat(2,valid_idx) - y_gt(valid_idx)).^2));
fprintf('TDoA EKF RMSE: %.2f meters\n', tdoa_rmse);

%% PARAMETRIZATION STUDY: Impact of sigma_tdoa
sigma_tdoa_vals = [1, 2, 3, 5, 10];
rmse_vals = zeros(size(sigma_tdoa_vals));
estimates_sigma = cell(length(sigma_tdoa_vals), 1);

for i_param = 1:length(sigma_tdoa_vals)
    sigma_tdoa = sigma_tdoa_vals(i_param);
    R_base = sigma_tdoa^2;
    x_hat_test = zeros(4, T);
    P_test = zeros(4, 4, T);
    x_hat_test(:,1) = [x_gt(1); y_gt(1); 0; 0];
    P_test(:,:,1) = diag([10, 10, 5, 5]);

    for k = 2:T
        x_pred = F * x_hat_test(:,k-1);
        P_pred = F * P_test(:,:,k-1) * F' + Q;
        tdoa_k = tdoa_data(:, k);
        master_idx = tdoa_master_idx(k);
        z = []; H = [];
        other_aps = setdiff(all_aps, master_idx);
        for j = 1:length(other_aps)
            ap_idx = other_aps(j);
            tdoa_meas = tdoa_k(j);
            if isnan(tdoa_meas), continue; end
            ap_pos = APs(1:2, ap_idx);
            master_pos = APs(1:2, master_idx);
            dx_ap = x_pred(1) - ap_pos(1);
            dy_ap = x_pred(2) - ap_pos(2);
            dx_master = x_pred(1) - master_pos(1);
            dy_master = x_pred(2) - master_pos(2);
            d_ap = sqrt(dx_ap^2 + dy_ap^2);
            d_master = sqrt(dx_master^2 + dy_master^2);
            h_i = d_ap - d_master;
            H_i = [(dx_ap/d_ap - dx_master/d_master), ...
                   (dy_ap/d_ap - dy_master/d_master), 0, 0];
            residual = tdoa_meas - h_i;
            z = [z; residual];
            H = [H; H_i];
        end
        if isempty(z)
            x_hat_test(:,k) = x_pred;
            P_test(:,:,k) = P_pred;
            continue;
        end
        R = R_base * eye(length(z));
        S = H * P_pred * H' + R;
        K = P_pred * H' / S;
        x_hat_test(:,k) = x_pred + K * z;
        P_test(:,:,k) = (eye(4) - K * H) * P_pred;
    end

    estimates_sigma{i_param} = x_hat_test;
    valid_idx = ~any(isnan(x_hat_test(1:2,:)), 1);
    rmse_vals(i_param) = sqrt(mean((x_hat_test(1,valid_idx) - x_gt(valid_idx)).^2 + ...
                                   (x_hat_test(2,valid_idx) - y_gt(valid_idx)).^2));
end

% Plot RMSE vs sigma_tdoa
figure;
plot(sigma_tdoa_vals, rmse_vals, '-o', 'LineWidth', 2);
xlabel('TDoA Measurement Std Deviation [m]');
ylabel('RMSE [m]');
title('Effect of TDoA Noise on EKF Accuracy');
grid on;

% Plot trajectories for different sigma_tdoa
figure; hold on;
plot(x_gt, y_gt, 'k-', 'LineWidth', 2);
colors = lines(length(sigma_tdoa_vals));
for i = 1:length(sigma_tdoa_vals)
    traj = estimates_sigma{i};
    plot(traj(1,:), traj(2,:), '--', 'Color', colors(i,:), ...
         'DisplayName', sprintf('\\sigma = %d m', sigma_tdoa_vals(i)));
end
plot(APs(1,:), APs(2,:), 'ko', 'DisplayName', 'APs');
xlabel('X [m]'); ylabel('Y [m]');
legend; title('TDoA EKF Estimates for Different \\sigma_{TDoA}');
axis equal; grid on;

% CDF Comparison for sigma_tdoa
figure; hold on;
for i = 1:length(sigma_tdoa_vals)
    traj = estimates_sigma{i};
    pos_err = sqrt((traj(1,:) - x_gt).^2 + (traj(2,:) - y_gt).^2);
    cdfplot(pos_err);
end
legend(arrayfun(@(s) sprintf('\\sigma = %d m', s), sigma_tdoa_vals, 'UniformOutput', false));
xlabel('Position Error [m]'); title('CDF of Position Errors for Different \\sigma_{TDoA}');
grid on;

%% PARAMETRIZATION STUDY: Q scaling
q_scales = [0.01, 0.1, 0.5, 1.0, 2.0];
rmse_vals_Q = zeros(size(q_scales));
estimates_q = cell(length(q_scales), 1);

for i_q = 1:length(q_scales)
    scale = q_scales(i_q);
    Q_test = scale * diag([0.05, 0.05, 0.5, 0.5]);
    x_hat_q = zeros(4, T);
    P_q = zeros(4, 4, T);
    x_hat_q(:,1) = [x_gt(1); y_gt(1); 0; 0];
    P_q(:,:,1) = diag([10, 10, 5, 5]);

    for k = 2:T
        x_pred = F * x_hat_q(:,k-1);
        P_pred = F * P_q(:,:,k-1) * F' + Q_test;
        tdoa_k = tdoa_data(:, k);
        master_idx = tdoa_master_idx(k);
        z = []; H = [];
        other_aps = setdiff(all_aps, master_idx);
        for j = 1:length(other_aps)
            ap_idx = other_aps(j);
            tdoa_meas = tdoa_k(j);
            if isnan(tdoa_meas), continue; end
            ap_pos = APs(1:2, ap_idx);
            master_pos = APs(1:2, master_idx);
            dx_ap = x_pred(1) - ap_pos(1);
            dy_ap = x_pred(2) - ap_pos(2);
            dx_master = x_pred(1) - master_pos(1);
            dy_master = x_pred(2) - master_pos(2);
            d_ap = sqrt(dx_ap^2 + dy_ap^2);
            d_master = sqrt(dx_master^2 + dy_master^2);
            h_i = d_ap - d_master;
            H_i = [(dx_ap/d_ap - dx_master/d_master), ...
                   (dy_ap/d_ap - dy_master/d_master), 0, 0];
            residual = tdoa_meas - h_i;
            z = [z; residual];
            H = [H; H_i];
        end
        if isempty(z)
            x_hat_q(:,k) = x_pred;
            P_q(:,:,k) = P_pred;
            continue;
        end
        R = sigma_tdoa^2 * eye(length(z));
        S = H * P_pred * H' + R;
        K = P_pred * H' / S;
        x_hat_q(:,k) = x_pred + K * z;
        P_q(:,:,k) = (eye(4) - K * H) * P_pred;
    end

    estimates_q{i_q} = x_hat_q;
    valid_idx = ~any(isnan(x_hat_q(1:2,:)), 1);
    rmse_vals_Q(i_q) = sqrt(mean((x_hat_q(1,valid_idx) - x_gt(valid_idx)).^2 + ...
                                 (x_hat_q(2,valid_idx) - y_gt(valid_idx)).^2));
end

% Plot RMSE vs Q scaling
figure;
plot(q_scales, rmse_vals_Q, '-o', 'LineWidth', 2);
xlabel('Process Noise Scaling Factor');
ylabel('RMSE [m]');
title('Effect of Process Noise Q on TDoA EKF Accuracy');
grid on;

% Trajectory comparison for Q
figure; hold on;
plot(x_gt, y_gt, 'k-', 'LineWidth', 2);
colors = lines(length(q_scales));
for i = 1:length(q_scales)
    traj = estimates_q{i};
    plot(traj(1,:), traj(2,:), '--', 'Color', colors(i,:), ...
         'DisplayName', sprintf('Q scale = %.2f', q_scales(i)));
end
plot(APs(1,:), APs(2,:), 'ko', 'DisplayName', 'APs');
xlabel('X [m]'); ylabel('Y [m]');
legend; title('TDoA EKF Estimates for Different Q Scaling');
axis equal; grid on;

%% Summary Table
fprintf('\n--- RMSE vs sigma_tdoa ---\n');
for i = 1:length(sigma_tdoa_vals)
    fprintf('  sigma = %d m: RMSE = %.2f m\n', sigma_tdoa_vals(i), rmse_vals(i));
end

fprintf('\n--- RMSE vs Q scaling ---\n');
for i = 1:length(q_scales)
    fprintf('  Q scale = %.2f: RMSE = %.2f m\n', q_scales(i), rmse_vals_Q(i));
end
