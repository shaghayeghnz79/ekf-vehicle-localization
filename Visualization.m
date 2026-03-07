clear; close all; clc;
load('LNSM_Project_Data.mat');

i = 1; % Choose 1, 2, or 3

x_y_z = ground_truth{i};  % [3 x T] matrix
x_coords = x_y_z(1, :);
y_coords = x_y_z(2, :);
z_coords = x_y_z(3, :);

%%
% figure;
plot(x_coords, y_coords, '.'); hold on;
plot(AP(1, :), AP(2, :), '^r', 'MarkerSize', 8);
xlabel('X [m]'); ylabel('Y [m]');
title('Vehicle Ground Truth Trajectory');
grid on; axis equal;
legend('Ground Truth', 'APs');


%%
tdoa_data = TDoA{i};  % [9 x T]
T = size(tdoa_data, 2);
time = linspace(0, (T-1)/10, T);  % 10 Hz

figure; hold on;
for k = 1:size(tdoa_data, 1)
    plot(time, tdoa_data(k, :), '.');
end
xlabel('Time [s]');
ylabel('TDoA [m]');
title('TDoA Measurements Over Time');
grid on;
ylim([-50, 100]);



%%
aoa_data = AoA{i};  % [20 x T]
azimuth = aoa_data(1:10, :);
elevation = aoa_data(11:20, :);

figure;
subplot(2,1,1); hold on;
for k = 1:10
    plot(time, rad2deg(azimuth(k, :)), '.');
end
ylabel('Azimuth [deg]');
title('AOA Azimuth (Local)');
grid on;

subplot(2,1,2); hold on;
for k = 1:10
    plot(time, rad2deg(elevation(k, :)), '.');
end
xlabel('Time [s]');
ylabel('Elevation [deg]');
title('AOA Elevation (Local)');
grid on;


%%
figure;

subplot(2,1,1); hold on;
for k = 1:10
    global_azimuth = azimuth(k, :) + APyaw(k);
    plot(time, rad2deg(global_azimuth), '.');
end
ylabel('Azimuth [deg]');
title('AOA Azimuth (Global)');
grid on;

subplot(2,1,2); hold on;
for k = 1:10
    global_elevation = elevation(k, :) + APpitch(k);
    plot(time, rad2deg(global_elevation), '.');
end
xlabel('Time [s]');
ylabel('Elevation [deg]');
title('AOA Elevation (Global)');
grid on;




