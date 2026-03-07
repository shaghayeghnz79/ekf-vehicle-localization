%% AoA EKF Localization
clear; close all; clc;
load('LNSM_Project_Data.mat');

i = 1; % Choose scenario 1, 2, or 3

% Extract azimuth AoA
azimuth = AoA{i}(1:10, :); % 10 APs x T
T = size(azimuth, 2);
dt = 0.1; % sampling interval (10 Hz)
APyaw = APyaw(:); 

% Ground truth
gt = ground_truth{i};
x_gt = gt(1, :);
y_gt = gt(2, :);

% === Initialization ===
numStates = 4; % [x, y, vx, vy]
T = length(x_gt);  

% Define sampling time
parameters.samplingTime = dt;

% Estimate initial velocity
vx0 = (x_gt(3) - x_gt(1)) / (2*dt);
vy0 = (y_gt(3) - y_gt(1)) / (2*dt);

% Initial state and covariance
x_hat = zeros(numStates, T);
P = zeros(numStates, numStates, T);
x_hat(:,1) = [x_gt(1); y_gt(1); vx0; vy0];
P(:,:,1) = diag([10, 10, 5, 5]);

% === Motion Model ===
F = [1 , 0 , dt , 0; 
     0 , 1 , 0 , dt;
     0 , 0 , 1 , 0;
     0 , 0 , 0 , 1];

% Process noise model
sigma_aoa = 0.1; 
L = [0.5*dt^2,      0;
         0, 0.5*dt^2;
        dt,      0;
         0,     dt];

Q = sigma_aoa^2 * (L * L');


% === EKF Loop ===
for k = 2:T
    % Prediction step
    x_pred = F * x_hat(:, k-1);
    P_pred = F * P(:,:,k-1) * F' + Q;

    % Measurement update  
    z = []; H = [];
    for ap_idx = 1:10
        meas = azimuth(ap_idx, k);
        if ~isnan(meas)
            % Compute predicted AoA
            dx = x_pred(1) - AP(1,ap_idx);
            dy = x_pred(2) - AP(2,ap_idx);
            global_angle = atan2(dy, dx);

            % Convert AP-local AoA to global
            measured_global = wrapToPi(meas + APyaw(ap_idx));

            
            y_diff = wrapToPi(measured_global - global_angle);

            % Gate: reject outliers
            if abs(y_diff) < deg2rad(30)
                % Jacobian
                denom = dx^2 + dy^2;
                dh_dx = [-dy / denom, dx / denom, 0, 0];
                z = [z; y_diff];
                H = [H; dh_dx];
            end
        end
    end

    % Update step
    if ~isempty(z)
        R = sigma_aoa^2 * eye(length(z));
        S = H * P_pred * H' + R;
        K = P_pred * H' / S;
        x_hat(:,k) = x_pred + K * z;
        P(:,:,k) = (eye(numStates) - K * H) * P_pred;
    else

        x_hat(:,k) = x_pred;
        P(:,:,k) = P_pred;
    end
end


% === Plot EKF estimate ===
figure;
plot(x_gt, y_gt, 'b-', 'LineWidth', 2); hold on;
plot(x_hat(1,:), x_hat(2,:), 'r--', 'LineWidth', 1.5);
plot(AP(1,:), AP(2,:), 'k^', 'MarkerSize', 8, 'LineWidth', 1.5);
xlabel('X [m]'); ylabel('Y [m]');
title('AoA EKF Localization');
legend('Ground Truth', 'EKF Estimate', 'APs');
axis equal; grid on;

% === Error and RMSE ===
pos_error = sqrt((x_gt - x_hat(1,:)).^2 + (y_gt - x_hat(2,:)).^2);
valid_idx = ~any(isnan(x_hat(1:2,:)), 1);
aoa_rmse = sqrt(mean(pos_error(valid_idx).^2));
fprintf('AoA EKF RMSE: %.2f meters\n', aoa_rmse);

figure;
plot(dt*(1:T), pos_error, 'LineWidth', 1.5);
xlabel('Time [s]'); ylabel('Position Error [m]');
title('AoA EKF Position Error Over Time');
grid on;

figure;
cdfplot(pos_error); grid on;
xlabel('Position Error [m]'); title('CDF of AoA EKF Position Error');

%% === Param Study: AoA Noise Impact ===
aoa_sigmas_deg = [1, 3, 5, 10, 15];
rmse_vals_sigma = zeros(size(aoa_sigmas_deg));
estimates_sigma = cell(length(aoa_sigmas_deg), 1);

for i_s = 1:length(aoa_sigmas_deg)
    sigma_aoa = deg2rad(aoa_sigmas_deg(i_s));

    % Reinitialize state
    x_hat_s = zeros(numStates, T);
    P_s = zeros(numStates, numStates, T);
    x_hat_s(:,1) = [x_gt(1); y_gt(1); vx0; vy0];
    P_s(:,:,1) = diag([10, 10, 5, 5]);

    for k = 2:T
        x_pred = F * x_hat_s(:,k-1);
        P_pred = F * P_s(:,:,k-1) * F' + Q;

        z = []; H = [];
        for ap_idx = 1:10
            meas = azimuth(ap_idx, k);
            if isnan(meas), continue; end

            dx = x_pred(1) - AP(1,ap_idx);
            dy = x_pred(2) - AP(2,ap_idx);
            global_angle = atan2(dy, dx);
            measured_global = wrapToPi(meas + APyaw(ap_idx));
            y_diff = wrapToPi(measured_global - global_angle);

            if abs(y_diff) < deg2rad(30)
                dh_dx = [-dy / (dx^2 + dy^2), dx / (dx^2 + dy^2), 0, 0];
                z = [z; y_diff];
                H = [H; dh_dx];
            end
        end

        if isempty(z)
            x_hat_s(:,k) = x_pred;
            P_s(:,:,k) = P_pred;
        else
            R = sigma_aoa^2 * eye(length(z));
            K = P_pred * H' / (H * P_pred * H' + R);
            x_hat_s(:,k) = x_pred + K * z;
            P_s(:,:,k) = (eye(numStates) - K*H) * P_pred;
        end
    end

    estimates_sigma{i_s} = x_hat_s;
    valid_idx = ~any(isnan(x_hat_s(1:2,:)), 1);
    rmse_vals_sigma(i_s) = sqrt(mean((x_hat_s(1,valid_idx) - x_gt(valid_idx)).^2 + ...
                                     (x_hat_s(2,valid_idx) - y_gt(valid_idx)).^2));
end

% Plot RMSE vs sigma
figure;
plot(aoa_sigmas_deg, rmse_vals_sigma, '-o', 'LineWidth', 2);
xlabel('AoA Measurement Std Dev [deg]');
ylabel('RMSE [m]');
title('Effect of AoA Noise on EKF Accuracy');
grid on;

% Trajectory comparisons for sigma
figure; hold on;
plot(x_gt, y_gt, 'k-', 'LineWidth', 2);
colors = lines(length(aoa_sigmas_deg));
for i = 1:length(aoa_sigmas_deg)
    traj = estimates_sigma{i};
    plot(traj(1,:), traj(2,:), '--', 'Color', colors(i,:), ...
        'DisplayName', sprintf('\\sigma = %d°', aoa_sigmas_deg(i)));
end
plot(AP(1,:), AP(2,:), 'ko', 'DisplayName', 'APs');
xlabel('X [m]'); ylabel('Y [m]');
legend; title('AoA EKF Trajectories for Different AoA Noise Levels');
axis equal; grid on;

% CDF plots for sigma
figure; hold on;
for i = 1:length(aoa_sigmas_deg)
    traj = estimates_sigma{i};
    err = sqrt((traj(1,:) - x_gt).^2 + (traj(2,:) - y_gt).^2);
    cdfplot(err);
end
legend(arrayfun(@(s) sprintf('\\sigma = %d°', s), aoa_sigmas_deg, 'UniformOutput', false));
xlabel('Position Error [m]'); title('CDF of AoA Errors (varying AoA noise)');
grid on;

%% === Param Study: Q scaling ===
q_scales = [0.01, 0.1, 0.5, 1.0, 2.0];
rmse_vals_Q = zeros(size(q_scales));
estimates_q = cell(length(q_scales), 1);

for i_q = 1:length(q_scales)
    Q_test = q_scales(i_q) * diag([0.5^2, 0.5^2, 2^2, 2^2]);

    x_hat_q = zeros(numStates, T);
    P_q = zeros(numStates, numStates, T);
    x_hat_q(:,1) = [x_gt(1); y_gt(1); vx0; vy0];
    P_q(:,:,1) = diag([10, 10, 5, 5]);

    for k = 2:T
        x_pred = F * x_hat_q(:,k-1);
        P_pred = F * P_q(:,:,k-1) * F' + Q_test;

        z = []; H = [];
        for ap_idx = 1:10
            meas = azimuth(ap_idx, k);
            if isnan(meas), continue; end

            dx = x_pred(1) - AP(1,ap_idx);
            dy = x_pred(2) - AP(2,ap_idx);
            global_angle = atan2(dy, dx);
            measured_global = wrapToPi(meas + APyaw(ap_idx));
            y_diff = wrapToPi(measured_global - global_angle);

            if abs(y_diff) < deg2rad(30)
                dh_dx = [-dy / (dx^2 + dy^2), dx / (dx^2 + dy^2), 0, 0];
                z = [z; y_diff];
                H = [H; dh_dx];
            end
        end

        if isempty(z)
            x_hat_q(:,k) = x_pred;
            P_q(:,:,k) = P_pred;
        else
            R = sigma_aoa^2 * eye(length(z));
            K = P_pred * H' / (H * P_pred * H' + R);
            x_hat_q(:,k) = x_pred + K * z;
            P_q(:,:,k) = (eye(numStates) - K*H) * P_pred;
        end
    end

    estimates_q{i_q} = x_hat_q;
    valid_idx = ~any(isnan(x_hat_q(1:2,:)), 1);
    rmse_vals_Q(i_q) = sqrt(mean((x_hat_q(1,valid_idx) - x_gt(valid_idx)).^2 + ...
                                 (x_hat_q(2,valid_idx) - y_gt(valid_idx)).^2));
end

% Plot RMSE vs Q scale
figure;
plot(q_scales, rmse_vals_Q, '-o', 'LineWidth', 2);
xlabel('Q Scaling Factor'); ylabel('RMSE [m]');
title('Effect of Q on AoA EKF Accuracy');
grid on;

% Trajectory comparisons for Q
figure; hold on;
plot(x_gt, y_gt, 'k-', 'LineWidth', 2);
colors = lines(length(q_scales));
for i = 1:length(q_scales)
    traj = estimates_q{i};
    plot(traj(1,:), traj(2,:), '--', 'Color', colors(i,:), ...
         'DisplayName', sprintf('Q scale = %.2f', q_scales(i)));
end
plot(AP(1,:), AP(2,:), 'ko', 'DisplayName', 'APs');
xlabel('X [m]'); ylabel('Y [m]');
legend; title('AoA EKF Trajectories for Different Q Scaling');
axis equal; grid on;

% Summary RMSEs
fprintf('\n--- RMSE vs AoA noise ---\n');
for i = 1:length(aoa_sigmas_deg)
    fprintf('  σ = %2d° -> RMSE = %.2f m\n', aoa_sigmas_deg(i), rmse_vals_sigma(i));
end
fprintf('\n--- RMSE vs Q scaling ---\n');
for i = 1:length(q_scales)
    fprintf('  Q scale = %.2f -> RMSE = %.2f m\n', q_scales(i), rmse_vals_Q(i));
end
