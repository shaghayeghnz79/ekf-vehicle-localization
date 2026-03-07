%% TDoA_AoA EKF Localization
clear; clc; close all;
load('LNSM_Project_Data.mat');

i = 1; % scenario index

% Extract data
T = size(TDoA{i}, 2);
dt = 0.1; % 10 Hz

% Ground truth
gt = ground_truth{i};
x_gt = gt(1,:);
y_gt = gt(2,:);

% Measurements
tdoa_raw = TDoA{i};
tdoa_master_idx = round(tdoa_raw(end, :));
tdoa_data = tdoa_raw(1:end-1, :);

azimuth = AoA{i}(1:10, :); % [10 x T]

% APs
APs = AP;
APyaw = APyaw(:);

% EKF Initialization
x_hat = zeros(4, T);
P = zeros(4, 4, T);
x_hat(:,1) = [x_gt(1); y_gt(1); 0; 0];
P(:,:,1) = diag([10, 10, 5, 5]);

% Motion model
F = [1 0 dt 0;
     0 1 0 dt;
     0 0 1 0;
     0 0 0 1];

% Process noise model
sigma_acc = 0.2;
L = [0.5*dt^2, 0;
     0,        0.5*dt^2;
     dt,       0;
     0,        dt];
Q = sigma_acc^2 * (L * L');

% Measurement noise
sigma_tdoa = 3;              % meters
sigma_aoa_deg = 5;           
sigma_aoa = deg2rad(sigma_aoa_deg);  % radians

% Measurement noise covariances
R_aoa = sigma_aoa^2;
R_tdoa = sigma_tdoa^2;



% EKF Loop
all_aps = 1:10;
for k = 2:T
    % Prediction
    x_pred = F * x_hat(:,k-1);
    P_pred = F * P(:,:,k-1) * F' + Q;

    H = [];
    z = [];
    R_blocks = {};

    % TDOA measurements
    tdoa_k = tdoa_data(:,k);
    master_idx = tdoa_master_idx(k);
    if ismember(master_idx, 1:10)
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
            H_i = [(dx_ap/d_ap - dx_master/d_master), (dy_ap/d_ap - dy_master/d_master), 0, 0];

            z = [z; tdoa_meas - h_i];
            H = [H; H_i];
            R_blocks{end+1} = sigma_tdoa^2;
        end
    end

    % AOA measurements
    for ap_idx = 1:10
        meas = azimuth(ap_idx, k);
        if isnan(meas), continue; end

        dx = x_pred(1) - APs(1, ap_idx);
        dy = x_pred(2) - APs(2, ap_idx);

        global_angle = atan2(dy, dx);
        measured_global = wrapToPi(meas + APyaw(ap_idx));
        y_diff = wrapToPi(measured_global - global_angle);

        if abs(y_diff) < deg2rad(30)
            H_i = [-dy / (dx^2 + dy^2), dx / (dx^2 + dy^2), 0, 0];
            z = [z; y_diff];
            H = [H; H_i];
            R_blocks{end+1} = sigma_aoa^2;
        end
    end

    if isempty(z)
        x_hat(:,k) = x_pred;
        P(:,:,k) = P_pred;
        continue;
    end

    R = blkdiag(R_blocks{:});
    S = H * P_pred * H' + R;
    K = P_pred * H' / S;

    x_hat(:,k) = x_pred + K * z;
    P(:,:,k) = (eye(4) - K * H) * P_pred;
end

%% Plot results
figure;
plot(x_gt, y_gt, 'b-', 'LineWidth', 1.5); hold on;
plot(x_hat(1,:), x_hat(2,:), 'r--', 'LineWidth', 1.5);
plot(APs(1,:), APs(2,:), 'k^', 'MarkerSize', 8, 'LineWidth', 1.5);
xlabel('X [m]'); ylabel('Y [m]');
title('TDoA + AoA EKF Localization');
legend('Ground Truth', 'EKF Estimate', 'APs');
axis equal; grid on;

%% Position Error
pos_err = sqrt((x_hat(1,:) - x_gt).^2 + (x_hat(2,:) - y_gt).^2);

figure;
plot((1:T)*dt, pos_err, 'LineWidth', 1.5);
xlabel('Time [s]');
ylabel('Position Error [m]');
title('Fusion EKF Position Error Over Time');
grid on;

figure;
cdfplot(pos_err);
xlabel('Position Error [m]');
title('CDF of Position Error (Fusion EKF)');
grid on;

%% RMSE
valid_idx = ~any(isnan(x_hat(1:2,:)), 1);
fusion_rmse = sqrt(mean((x_hat(1,valid_idx) - x_gt(valid_idx)).^2 + (x_hat(2,valid_idx) - y_gt(valid_idx)).^2));
fprintf('Fusion EKF RMSE: %.2f meters\n', fusion_rmse);


%% PARAMETRIZATION STUDY: Impact of AoA noise on Fusion EKF (TDoA + AoA)
aoa_sigmas_deg = [1, 3, 5, 10, 15];  % test values
rmse_vals_fusion = zeros(size(aoa_sigmas_deg));

for i_s = 1:length(aoa_sigmas_deg)
    sigma_aoa = deg2rad(aoa_sigmas_deg(i_s));  % convert to radians

    % Reinitialize state
    x_hat_s = zeros(4, T);
    P_s = zeros(4, 4, T);
    x_hat_s(:,1) = [x_gt(1); y_gt(1); 0; 0];
    P_s(:,:,1) = diag([10, 10, 5, 5]);

    for k = 2:T
        % Prediction
        x_pred = F * x_hat_s(:,k-1);
        P_pred = F * P_s(:,:,k-1) * F' + Q;

        H = []; z = []; R_blocks = {};

        % TDoA update
        tdoa_k = tdoa_data(:, k);
        master_idx = tdoa_master_idx(k);
        if ismember(master_idx, 1:10)
            other_aps = setdiff(1:10, master_idx);
            for j = 1:length(other_aps)
                ap_idx = other_aps(j);
                tdoa_meas = tdoa_k(j);
                if isnan(tdoa_meas), continue; end

                ap_pos = APs(1:2, ap_idx);
                master_pos = APs(1:2, master_idx);

                dx_ap = x_pred(1) - ap_pos(1);
                dy_ap = x_pred(2) - ap_pos(2);
                dx_m = x_pred(1) - master_pos(1);
                dy_m = x_pred(2) - master_pos(2);

                d_ap = sqrt(dx_ap^2 + dy_ap^2);
                d_m  = sqrt(dx_m^2 + dy_m^2);

                h_i = d_ap - d_m;
                H_i = [(dx_ap/d_ap - dx_m/d_m), (dy_ap/d_ap - dy_m/d_m), 0, 0];
                z = [z; tdoa_meas - h_i];
                H = [H; H_i];
                R_blocks{end+1} = sigma_tdoa^2;
            end
        end

        %% AoA update
        for ap_idx = 1:10
            meas = azimuth(ap_idx, k);
            if isnan(meas), continue; end

            dx = x_pred(1) - APs(1,ap_idx);
            dy = x_pred(2) - APs(2,ap_idx);

            global_angle = atan2(dy, dx);
            measured_global = wrapToPi(meas + APyaw(ap_idx));
            y_diff = wrapToPi(measured_global - global_angle);

            if abs(y_diff) < deg2rad(30)
                H_i = [-dy / (dx^2 + dy^2), dx / (dx^2 + dy^2), 0, 0];
                z = [z; y_diff];
                H = [H; H_i];
                R_blocks{end+1} = sigma_aoa^2;
            end
        end

        % Skip update if no measurements
        if isempty(z)
            x_hat_s(:,k) = x_pred;
            P_s(:,:,k) = P_pred;
            continue;
        end

        R = blkdiag(R_blocks{:});
        S = H * P_pred * H' + R;
        K = P_pred * H' / S;

        x_hat_s(:,k) = x_pred + K * z;
        P_s(:,:,k) = (eye(4) - K * H) * P_pred;
    end

    % Compute RMSE
    valid_idx = ~any(isnan(x_hat_s(1:2,:)), 1);
    rmse_vals_fusion(i_s) = sqrt(mean((x_hat_s(1,valid_idx) - x_gt(valid_idx)).^2 + ...
                                      (x_hat_s(2,valid_idx) - y_gt(valid_idx)).^2));
end

% Plot RMSE vs AoA noise (in deg)
figure;
plot(aoa_sigmas_deg, rmse_vals_fusion, '-o', 'LineWidth', 2);
xlabel('AOA Measurement Std Dev [deg]');
ylabel('RMSE [m]');
title('Effect of AOA Noise on Fusion EKF Accuracy');
grid on;

%% PARAMETRIZATION STUDY: Impact of TDoA noise on Fusion EKF (AoA + TDoA)
tdoa_sigmas = [1, 2, 3, 5, 10];  % std dev in meters
rmse_vals_tdoa = zeros(size(tdoa_sigmas));

for i_s = 1:length(tdoa_sigmas)
    sigma_tdoa = tdoa_sigmas(i_s); 
    R_base = sigma_tdoa^2;

    % Reinitialize state
    x_hat_s = zeros(4, T);
    P_s = zeros(4, 4, T);
    x_hat_s(:,1) = [x_gt(1); y_gt(1); 0; 0];
    P_s(:,:,1) = diag([10, 10, 5, 5]);

    for k = 2:T
        x_pred = F * x_hat_s(:,k-1);
        P_pred = F * P_s(:,:,k-1) * F' + Q;

        H = []; z = []; R_blocks = {};

        %% TDoA update
        tdoa_k = tdoa_data(:, k);
        master_idx = tdoa_master_idx(k);
        if ismember(master_idx, 1:10)
            other_aps = setdiff(1:10, master_idx);
            for j = 1:length(other_aps)
                ap_idx = other_aps(j);
                tdoa_meas = tdoa_k(j);
                if isnan(tdoa_meas), continue; end

                ap_pos = APs(1:2, ap_idx);
                master_pos = APs(1:2, master_idx);

                dx_ap = x_pred(1) - ap_pos(1);
                dy_ap = x_pred(2) - ap_pos(2);
                dx_m  = x_pred(1) - master_pos(1);
                dy_m  = x_pred(2) - master_pos(2);

                d_ap = sqrt(dx_ap^2 + dy_ap^2);
                d_m  = sqrt(dx_m^2 + dy_m^2);

                h_i = d_ap - d_m;
                H_i = [(dx_ap/d_ap - dx_m/d_m), (dy_ap/d_ap - dy_m/d_m), 0, 0];

                z = [z; tdoa_meas - h_i];
                H = [H; H_i];
                R_blocks{end+1} = R_base;
            end
        end

        %% AoA update (fixed sigma_aoa, e.g. 5 deg)
        sigma_aoa = deg2rad(5);  
        for ap_idx = 1:10
            meas = azimuth(ap_idx, k);
            if isnan(meas), continue; end

            dx = x_pred(1) - APs(1,ap_idx);
            dy = x_pred(2) - APs(2,ap_idx);

            global_angle = atan2(dy, dx);
            measured_global = wrapToPi(meas + APyaw(ap_idx));
            y_diff = wrapToPi(measured_global - global_angle);

            if abs(y_diff) < deg2rad(30)
                H_i = [-dy / (dx^2 + dy^2), dx / (dx^2 + dy^2), 0, 0];
                z = [z; y_diff];
                H = [H; H_i];
                R_blocks{end+1} = sigma_aoa^2;
            end
        end

        % EKF Update
        if isempty(z)
            x_hat_s(:,k) = x_pred;
            P_s(:,:,k) = P_pred;
            continue;
        end

        R = blkdiag(R_blocks{:});
        S = H * P_pred * H' + R;
        K = P_pred * H' / S;

        x_hat_s(:,k) = x_pred + K * z;
        P_s(:,:,k) = (eye(4) - K * H) * P_pred;
    end

    % Compute RMSE
    valid_idx = ~any(isnan(x_hat_s(1:2,:)), 1);
    rmse_vals_tdoa(i_s) = sqrt(mean((x_hat_s(1,valid_idx) - x_gt(valid_idx)).^2 + ...
                                    (x_hat_s(2,valid_idx) - y_gt(valid_idx)).^2));
end

% Plot RMSE vs TDoA noise (in meters)
figure;
plot(tdoa_sigmas, rmse_vals_tdoa, '-o', 'LineWidth', 2);
xlabel('TDoA Measurement Std Dev [m]');
ylabel('RMSE [m]');
title('Effect of TDoA Noise on Fusion EKF Accuracy');
grid on;

%% PARAMETRIZATION STUDY: Impact of Q on Fusion EKF
q_scales = [0.01, 0.1, 0.5, 1.0, 2.0];  % scale factors
rmse_vals_q_fusion = zeros(size(q_scales));

for i_q = 1:length(q_scales)
    scale = q_scales(i_q);
    Q_test = scale * diag([0.05, 0.05, 0.5, 0.5]);

    % Reinitialize state
    x_hat_q = zeros(4, T);
    P_q = zeros(4, 4, T);
    x_hat_q(:,1) = [x_gt(1); y_gt(1); 0; 0];
    P_q(:,:,1) = diag([10, 10, 5, 5]);

    for k = 2:T
        x_pred = F * x_hat_q(:,k-1);
        P_pred = F * P_q(:,:,k-1) * F' + Q_test;

        H = []; z = []; R_blocks = {};

        %% TDoA update
        tdoa_k = tdoa_data(:, k);
        master_idx = tdoa_master_idx(k);
        if ismember(master_idx, 1:10)
            other_aps = setdiff(1:10, master_idx);
            for j = 1:length(other_aps)
                ap_idx = other_aps(j);
                tdoa_meas = tdoa_k(j);
                if isnan(tdoa_meas), continue; end

                ap_pos = APs(1:2, ap_idx);
                master_pos = APs(1:2, master_idx);

                dx_ap = x_pred(1) - ap_pos(1);
                dy_ap = x_pred(2) - ap_pos(2);
                dx_m  = x_pred(1) - master_pos(1);
                dy_m  = x_pred(2) - master_pos(2);

                d_ap = sqrt(dx_ap^2 + dy_ap^2);
                d_m  = sqrt(dx_m^2 + dy_m^2);

                h_i = d_ap - d_m;
                H_i = [(dx_ap/d_ap - dx_m/d_m), (dy_ap/d_ap - dy_m/d_m), 0, 0];

                z = [z; tdoa_meas - h_i];
                H = [H; H_i];
                R_blocks{end+1} = sigma_tdoa^2;
            end
        end

        %% AoA update (fixed sigma_aoa, e.g., 5 deg)
        sigma_aoa = deg2rad(5);
        for ap_idx = 1:10
            meas = azimuth(ap_idx, k);
            if isnan(meas), continue; end

            dx = x_pred(1) - APs(1,ap_idx);
            dy = x_pred(2) - APs(2,ap_idx);

            global_angle = atan2(dy, dx);
            measured_global = wrapToPi(meas + APyaw(ap_idx));
            y_diff = wrapToPi(measured_global - global_angle);

            if abs(y_diff) < deg2rad(30)
                H_i = [-dy / (dx^2 + dy^2), dx / (dx^2 + dy^2), 0, 0];
                z = [z; y_diff];
                H = [H; H_i];
                R_blocks{end+1} = sigma_aoa^2;
            end
        end

        % EKF Update
        if isempty(z)
            x_hat_q(:,k) = x_pred;
            P_q(:,:,k) = P_pred;
            continue;
        end

        R = blkdiag(R_blocks{:});
        S = H * P_pred * H' + R;
        K = P_pred * H' / S;

        x_hat_q(:,k) = x_pred + K * z;
        P_q(:,:,k) = (eye(4) - K * H) * P_pred;
    end

    % Compute RMSE
    valid_idx = ~any(isnan(x_hat_q(1:2,:)), 1);
    rmse_vals_q_fusion(i_q) = sqrt(mean((x_hat_q(1,valid_idx) - x_gt(valid_idx)).^2 + ...
                                        (x_hat_q(2,valid_idx) - y_gt(valid_idx)).^2));
end

% Plot RMSE vs Q scaling
figure;
plot(q_scales, rmse_vals_q_fusion, '-o', 'LineWidth', 2);
xlabel('Process Noise Scaling Factor');
ylabel('RMSE [m]');
title('Effect of Q on Fusion EKF Accuracy');
grid on;



%% Q-sweep analysis
q_scales = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0];
x_hat_q_all = cell(1, length(q_scales));
pos_err_q = cell(1, length(q_scales));

for i_q = 1:length(q_scales)
    Q_test = q_scales(i_q) * diag([0.05, 0.05, 0.5, 0.5]);
    x_hat_q = zeros(4, T);
    P_q = zeros(4, 4, T);
    x_hat_q(:,1) = [x_gt(1); y_gt(1); 0; 0];
    P_q(:,:,1) = diag([10, 10, 5, 5]);

    for k = 2:T
        x_pred = F * x_hat_q(:,k-1);
        P_pred = F * P_q(:,:,k-1) * F' + Q_test;

        H = []; z = []; R_blocks = {};

        tdoa_k = tdoa_data(:,k);
        master_idx = tdoa_master_idx(k);
        if ismember(master_idx, 1:10)
            other_aps = setdiff(all_aps, master_idx);
            for j = 1:length(other_aps)
                ap_idx = other_aps(j);
                tdoa_meas = tdoa_k(j);
                if isnan(tdoa_meas), continue; end
                ap_pos = APs(1:2, ap_idx);
                master_pos = APs(1:2, master_idx);
                dx_ap = x_pred(1) - ap_pos(1);
                dy_ap = x_pred(2) - ap_pos(2);
                dx_m = x_pred(1) - master_pos(1);
                dy_m = x_pred(2) - master_pos(2);
                d_ap = sqrt(dx_ap^2 + dy_ap^2);
                d_m = sqrt(dx_m^2 + dy_m^2);
                h_i = d_ap - d_m;
                H_i = [(dx_ap/d_ap - dx_m/d_m), (dy_ap/d_ap - dy_m/d_m), 0, 0];
                z = [z; tdoa_meas - h_i];
                H = [H; H_i];
                R_blocks{end+1} = sigma_tdoa^2;
            end
        end

        for ap_idx = 1:10
            meas = azimuth(ap_idx, k);
            if isnan(meas), continue; end
            dx = x_pred(1) - APs(1, ap_idx);
            dy = x_pred(2) - APs(2, ap_idx);
            global_angle = atan2(dy, dx);
            measured_global = wrapToPi(meas + APyaw(ap_idx));
            y_diff = wrapToPi(measured_global - global_angle);
            if abs(y_diff) < deg2rad(30)
                H_i = [-dy / (dx^2 + dy^2), dx / (dx^2 + dy^2), 0, 0];
                z = [z; y_diff];
                H = [H; H_i];
                R_blocks{end+1} = sigma_aoa^2;
            end
        end

        if isempty(z)
            x_hat_q(:,k) = x_pred;
            P_q(:,:,k) = P_pred;
            continue;
        end

        R = blkdiag(R_blocks{:});
        S = H * P_pred * H' + R;
        K = P_pred * H' / S;

        x_hat_q(:,k) = x_pred + K * z;
        P_q(:,:,k) = (eye(4) - K * H) * P_pred;
    end

    x_hat_q_all{i_q} = x_hat_q;
    valid_idx_q = ~any(isnan(x_hat_q(1:2,:)), 1);
    pos_err_q{i_q} = sqrt((x_hat_q(1,valid_idx_q) - x_gt(valid_idx_q)).^2 + (x_hat_q(2,valid_idx_q) - y_gt(valid_idx_q)).^2);
end

%% AoA noise sweep analysis
sigma_aoa_vals = deg2rad([1, 3, 5, 10, 15]);
x_hat_aoa_all = cell(1, length(sigma_aoa_vals));
pos_err_aoa_all = cell(1, length(sigma_aoa_vals));

for i_s = 1:length(sigma_aoa_vals)
    sigma_aoa = sigma_aoa_vals(i_s);
    x_hat_s = zeros(4, T);
    P_s = zeros(4, 4, T);
    x_hat_s(:,1) = [x_gt(1); y_gt(1); 0; 0];
    P_s(:,:,1) = diag([10, 10, 5, 5]);

    for k = 2:T
        x_pred = F * x_hat_s(:,k-1);
        P_pred = F * P_s(:,:,k-1) * F' + Q;

        H = []; z = []; R_blocks = {};

        tdoa_k = tdoa_data(:,k);
        master_idx = tdoa_master_idx(k);
        if ismember(master_idx, 1:10)
            other_aps = setdiff(all_aps, master_idx);
            for j = 1:length(other_aps)
                ap_idx = other_aps(j);
                tdoa_meas = tdoa_k(j);
                if isnan(tdoa_meas), continue; end
                ap_pos = APs(1:2, ap_idx);
                master_pos = APs(1:2, master_idx);
                dx_ap = x_pred(1) - ap_pos(1);
                dy_ap = x_pred(2) - ap_pos(2);
                dx_m = x_pred(1) - master_pos(1);
                dy_m = x_pred(2) - master_pos(2);
                d_ap = sqrt(dx_ap^2 + dy_ap^2);
                d_m  = sqrt(dx_m^2 + dy_m^2);
                h_i = d_ap - d_m;
                H_i = [(dx_ap/d_ap - dx_m/d_m), (dy_ap/d_ap - dy_m/d_m), 0, 0];
                z = [z; tdoa_meas - h_i];
                H = [H; H_i];
                R_blocks{end+1} = sigma_tdoa^2;
            end
        end

        for ap_idx = 1:10
            meas = azimuth(ap_idx, k);
            if isnan(meas), continue; end
            dx = x_pred(1) - APs(1, ap_idx);
            dy = x_pred(2) - APs(2, ap_idx);
            global_angle = atan2(dy, dx);
            measured_global = wrapToPi(meas + APyaw(ap_idx));
            y_diff = wrapToPi(measured_global - global_angle);
            if abs(y_diff) < deg2rad(30)
                H_i = [-dy / (dx^2 + dy^2), dx / (dx^2 + dy^2), 0, 0];
                z = [z; y_diff];
                H = [H; H_i];
                R_blocks{end+1} = sigma_aoa^2;
            end
        end

        if isempty(z)
            x_hat_s(:,k) = x_pred;
            P_s(:,:,k) = P_pred;
            continue;
        end

        R = blkdiag(R_blocks{:});
        S = H * P_pred * H' + R;
        K = P_pred * H' / S;

        x_hat_s(:,k) = x_pred + K * z;
        P_s(:,:,k) = (eye(4) - K * H) * P_pred;
    end

    x_hat_aoa_all{i_s} = x_hat_s;
    valid_idx_s = ~any(isnan(x_hat_s(1:2,:)), 1);
    pos_err_aoa_all{i_s} = sqrt((x_hat_s(1,valid_idx_s) - x_gt(valid_idx_s)).^2 + (x_hat_s(2,valid_idx_s) - y_gt(valid_idx_s)).^2);
end

%% AoA noise sweep trajectory visualization
colors = lines(length(sigma_aoa_vals));

figure;
plot(x_gt, y_gt, 'k-', 'LineWidth', 2, 'DisplayName', 'Ground Truth'); hold on;
for i = 1:length(sigma_aoa_vals)
    plot(x_hat_aoa_all{i}(1,:), x_hat_aoa_all{i}(2,:), '--', 'Color', colors(i,:), 'DisplayName', sprintf('AOA Noise = %d°', round(rad2deg(sigma_aoa_vals(i)))));
end
xlabel('X [m]'); ylabel('Y [m]');
title('Fusion EKF: Trajectories under Varying AoA Noise');
legend('Location', 'best');
grid on; axis equal;

figure;
hold on;
for i = 1:length(sigma_aoa_vals)
    plot((1:length(pos_err_aoa_all{i}))*dt, pos_err_aoa_all{i}, 'Color', colors(i,:), 'DisplayName', sprintf('AOA Noise = %d°', round(rad2deg(sigma_aoa_vals(i)))));
end
xlabel('Time [s]'); ylabel('Position Error [m]');
title('Fusion EKF: Position Error over Time under AoA Noise');
legend('Location', 'northeast');
grid on;

%% TDoA noise sweep analysis
sigma_tdoa_vals = [1, 2, 3, 5, 10];
x_hat_tdoa_all = cell(1, length(sigma_tdoa_vals));
pos_err_tdoa_all = cell(1, length(sigma_tdoa_vals));

for i_s = 1:length(sigma_tdoa_vals)
    sigma_tdoa = sigma_tdoa_vals(i_s);
    x_hat_s = zeros(4, T);
    P_s = zeros(4, 4, T);
    x_hat_s(:,1) = [x_gt(1); y_gt(1); 0; 0];
    P_s(:,:,1) = diag([10, 10, 5, 5]);

    for k = 2:T
        x_pred = F * x_hat_s(:,k-1);
        P_pred = F * P_s(:,:,k-1) * F' + Q;

        H = []; z = []; R_blocks = {};

        tdoa_k = tdoa_data(:,k);
        master_idx = tdoa_master_idx(k);
        if ismember(master_idx, 1:10)
            other_aps = setdiff(all_aps, master_idx);
            for j = 1:length(other_aps)
                ap_idx = other_aps(j);
                tdoa_meas = tdoa_k(j);
                if isnan(tdoa_meas), continue; end
                ap_pos = APs(1:2, ap_idx);
                master_pos = APs(1:2, master_idx);
                dx_ap = x_pred(1) - ap_pos(1);
                dy_ap = x_pred(2) - ap_pos(2);
                dx_m = x_pred(1) - master_pos(1);
                dy_m = x_pred(2) - master_pos(2);
                d_ap = sqrt(dx_ap^2 + dy_ap^2);
                d_m  = sqrt(dx_m^2 + dy_m^2);
                h_i = d_ap - d_m;
                H_i = [(dx_ap/d_ap - dx_m/d_m), (dy_ap/d_ap - dy_m/d_m), 0, 0];
                z = [z; tdoa_meas - h_i];
                H = [H; H_i];
                R_blocks{end+1} = sigma_tdoa^2;
            end
        end

        for ap_idx = 1:10
            meas = azimuth(ap_idx, k);
            if isnan(meas), continue; end
            dx = x_pred(1) - APs(1, ap_idx);
            dy = x_pred(2) - APs(2, ap_idx);
            global_angle = atan2(dy, dx);
            measured_global = wrapToPi(meas + APyaw(ap_idx));
            y_diff = wrapToPi(measured_global - global_angle);
            if abs(y_diff) < deg2rad(30)
                H_i = [-dy / (dx^2 + dy^2), dx / (dx^2 + dy^2), 0, 0];
                z = [z; y_diff];
                H = [H; H_i];
                R_blocks{end+1} = sigma_aoa^2;
            end
        end

        if isempty(z)
            x_hat_s(:,k) = x_pred;
            P_s(:,:,k) = P_pred;
            continue;
        end

        R = blkdiag(R_blocks{:});
        S = H * P_pred * H' + R;
        K = P_pred * H' / S;

        x_hat_s(:,k) = x_pred + K * z;
        P_s(:,:,k) = (eye(4) - K * H) * P_pred;
    end

    x_hat_tdoa_all{i_s} = x_hat_s;
    valid_idx_s = ~any(isnan(x_hat_s(1:2,:)), 1);
    pos_err_tdoa_all{i_s} = sqrt((x_hat_s(1,valid_idx_s) - x_gt(valid_idx_s)).^2 + (x_hat_s(2,valid_idx_s) - y_gt(valid_idx_s)).^2);
end

%% TDoA noise sweep trajectory visualization
colors = lines(length(sigma_tdoa_vals));

figure;
plot(x_gt, y_gt, 'k-', 'LineWidth', 2, 'DisplayName', 'Ground Truth'); hold on;
for i = 1:length(sigma_tdoa_vals)
    plot(x_hat_tdoa_all{i}(1,:), x_hat_tdoa_all{i}(2,:), '--', 'Color', colors(i,:), 'DisplayName', sprintf('TDoA Noise = %d m', sigma_tdoa_vals(i)));
end
xlabel('X [m]'); ylabel('Y [m]');
title('Fusion EKF: Trajectories under Varying TDoA Noise');
legend('Location', 'best');
grid on; axis equal;

figure;
hold on;
for i = 1:length(sigma_tdoa_vals)
    plot((1:length(pos_err_tdoa_all{i}))*dt, pos_err_tdoa_all{i}, 'Color', colors(i,:), 'DisplayName', sprintf('TDoA Noise = %d m', sigma_tdoa_vals(i)));
end
xlabel('Time [s]'); ylabel('Position Error [m]');
title('Fusion EKF: Position Error over Time under TDoA Noise');
legend('Location', 'northeast');
grid on;

%% Q-sweep trajectory visualization

colors = lines(length(q_scales));

figure;
plot(x_gt, y_gt, 'k-', 'LineWidth', 2, 'DisplayName', 'Ground Truth'); hold on;
for i = 1:length(q_scales)
    plot(x_hat_q_all{i}(1,:), x_hat_q_all{i}(2,:), '--', 'Color', colors(i,:), ...
        'DisplayName', sprintf('Q = %.2f', q_scales(i)));
end
xlabel('X [m]');
ylabel('Y [m]');
title('Fusion EKF: Trajectories under Different Q');
legend('Location', 'best');
grid on;
axis equal;

figure;
hold on;
for i = 1:length(q_scales)
    plot((1:length(pos_err_q{i}))*dt, pos_err_q{i}, 'Color', colors(i,:), ...
        'DisplayName', sprintf('Q = %.2f', q_scales(i)));
end
xlabel('Time [s]');
ylabel('Position Error [m]');
title('Fusion EKF: Position Error over Time under Different Q');
legend('Location', 'northeast');
grid on;
