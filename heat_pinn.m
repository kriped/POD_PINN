%% ROM with Physics-Informed Neural Network for 1D Heat Equation
% Code is based on the book 
clear all; close all; clc;

%% 1. Generate high-fidelity data for the 1D heat equation
% Physical parameters
L = 1;          % Domain length [0,L]
alpha = 0.01;   % Thermal diffusivity
T = 1;          % Total simulation time
nx = 100;       % Number of spatial discretization points
nt = 200;       % Number of time steps

% Discretization
x = linspace(0, L, nx)';
dx = x(2) - x(1);
t = linspace(0, T, nt);
dt = t(2) - t(1);

% Initial condition (Gaussian pulse)
u0 = exp(-((x-L/2).^2)/(0.1^2));

% Finite difference matrices for second derivative
e = ones(nx,1);
A = spdiags([e -2*e e], [-1 0 1], nx, nx) / dx^2;

% Apply boundary conditions (Dirichlet u(0,t) = u(L,t) = 0)
A(1,:) = 0; A(end,:) = 0;

% Explicit time stepping (for simplicity)
% CFL = alpha*dt/dx^2 should be < 0.5 for stability
CFL = alpha*dt/dx^2;
fprintf('CFL = %.4f\n', CFL);
if CFL >= 0.5
    warning('CFL condition violated! Solution may be unstable.');
end

% Store solution snapshots
U = zeros(nx, nt);
U(:,1) = u0;

% Solve high-fidelity model
u = u0;
for i = 1:nt-1
    u = u + dt * alpha * (A * u);
    u(1) = 0; u(end) = 0;  % enforce boundary conditions
    U(:,i+1) = u;
end

%% 2. Build POD-based ROM
% Compute SVD of snapshot matrix
[Phi, S, ~] = svd(U, 'econ');

% Plot singular values
figure;
semilogy(diag(S)/sum(diag(S)), 'o-');
xlabel('Mode number');
ylabel('Normalized Singular Value');
title('POD Modes Energy Distribution');
grid on;

% Choose number of POD modes to retain
energy_threshold = 0.999;
cumulative_energy = cumsum(diag(S).^2) / sum(diag(S).^2);
r = find(cumulative_energy >= energy_threshold, 1);
fprintf('Retaining %d POD modes to capture %.2f%% of energy\n', r, 100*energy_threshold);

% Truncate modes
Phi_r = Phi(:, 1:r);

% Project high-fidelity snapshots to get ROM coefficients
a = Phi_r' * U;

% Projected matrices for ROM
A_rom = Phi_r' * A * Phi_r;

%% 3. Create Physics-Informed Neural Network to learn ROM dynamics

% Define network architecture
inputSize = r;      % Input: ROM coefficients
hiddenSize = 20;    % Hidden layer size
outputSize = r;     % Output: Time derivative of ROM coefficients

% Create feedforward network
net = feedforwardnet(hiddenSize);
net.divideParam.trainRatio = 0.7;
net.divideParam.valRatio = 0.15;
net.divideParam.testRatio = 0.15;

% Prepare training data
% Input: POD coefficients at each time
X_train = a(:, 1:end-1);

% Output: Time derivative of POD coefficients (using finite difference)
Y_train = (a(:, 2:end) - a(:, 1:end-1)) / dt;

% Train the neural network
[net, tr] = train(net, X_train, Y_train);

%% 4. Create the physics-informed loss function

% Function to compute physics residual (heat equation in ROM space)
function residual = physicsResidual(a_pred, a, Phi_r, A_rom, alpha, dt)
    % a_pred: predicted coefficients at next time
    % a: coefficients at current time
    
    % Neural network prediction (data-driven)
    da_dt_nn = (a_pred - a) / dt;
    
    % Physics model prediction
    da_dt_physics = alpha * A_rom * a;
    
    % Residual between data-driven and physics-based predictions
    residual = norm(da_dt_nn - da_dt_physics) / norm(da_dt_physics);
end

%% 5. Test the ROM-PINN model

% Initial POD coefficients
a_test = a(:, 1);

% Storage for ROM-PINN solution
a_pinn = zeros(r, nt);
a_pinn(:, 1) = a_test;

% Time stepping with ROM-PINN
for i = 1:nt-1
    % 1. Neural network prediction
    da_dt_nn = net(a_pinn(:, i));
    
    % 2. Physics-based correction
    da_dt_physics = alpha * A_rom * a_pinn(:, i);
    
    % 3. Calculate physics residual
    a_next_nn = a_pinn(:, i) + dt * da_dt_nn;
    residual = physicsResidual(a_next_nn, a_pinn(:, i), Phi_r, A_rom, alpha, dt);
    
    % 4. Adapt physics weight based on residual
    physics_weight = 0.3 + 0.2 * min(residual, 1);  % Increase physics weight when residual is high
    
    % 5. Combine NN and physics (weighted average)
    da_dt_combined = (1-physics_weight) * da_dt_nn + physics_weight * da_dt_physics;
    
    % 6. Update ROM coefficients
    a_pinn(:, i+1) = a_pinn(:, i) + dt * da_dt_combined;
    
    % Print residual for every 20th step
    if mod(i, 20) == 0
        fprintf('Time step %d, Physics residual: %.4e\n', i, residual);
    end
end

% Reconstruct full solution from ROM coefficients
U_pinn = Phi_r * a_pinn;

%% 6. Compare high-fidelity and ROM-PINN solutions

% Plot solution at different time steps
plot_times = round(linspace(1,nt,5));

figure;
for i = 1:length(plot_times)
    t_idx = plot_times(i);
    
    subplot(length(plot_times), 1, i);
    plot(x, U(:, t_idx), 'b-', 'LineWidth', 2);
    hold on;
    plot(x, U_pinn(:, t_idx), 'r--', 'LineWidth', 1.5);
    
    title(sprintf('t = %.3f', t(t_idx)));
    xlabel('x');
    ylabel('u(x,t)');
    legend('High-fidelity', 'ROM-PINN');
    grid on;
end

% Calculate and display error
error = norm(U - U_pinn, 'fro') / norm(U, 'fro');
fprintf('Relative error between high-fidelity and ROM-PINN: %.4e\n', error);

%% 7. Example with parameter variation
%define new parameters
% Stress test with large change to diffusivity and much longer simulation
% time

alpha_new = 0.1;   % Thermal diffusivity
dx_new = L/100;       % Number of spatial discretization points
dt_new = T/2001;       % Number of time steps
% Check stability for new parameter
CFL_new = alpha_new*dt_new/dx_new^2;
fprintf('New CFL = %.4f\n', CFL_new);
%new discretization
% Discretization
x_new = (dx_new:dx_new:L)';
nx_new = length(x_new);
t_new = 0:dt_new:T;
nt_new = length(t_new);
if CFL_new >= 0.5
    error('CFL condition violated with new parameter! Solution may be unstable.');
    % Consider reducing dt to maintain stability
    % dt_new = 0.4 * dx^2 / alpha_new;
    % fprintf('Suggested dt: %.6f\n', dt_new);
else
    dt = dt_new;
    dx = dx_new;
    t = t_new;
    x = x_new;
    nt = nt_new;
    nx = nx_new; 
end

% Initial condition
u0_new = exp(-((x-L/2).^2)/(0.1^2));

% High-fidelity solution with new parameter
U_new = zeros(nx, nt);
U_new(:,1) = u0_new;

u = u0_new;
for i = 1:nt-1
    u = u + dt * alpha_new * (A * u);
    u(1) = 0; u(end) = 0;  % enforce boundary conditions
    U_new(:,i+1) = u;
end

% Project initial condition onto ROM basis
a_new = Phi_r' * u0_new;

% Storage for ROM-PINN solution with new parameter
a_pinn_new = zeros(r, nt);
a_pinn_new(:, 1) = a_new;

% Fix the paramNet training to accept single inputs correctly
paramNet = feedforwardnet(10);
% Train with individual scalars as separate samples
paramNet = train(paramNet, [alpha, alpha_new], [1, 1.5]);

% Time stepping with ROM-PINN for new parameter
for i = 1:nt-1
    % 1. Neural network prediction (base dynamics)
    da_dt_nn = net(a_pinn_new(:, i));
    
    % 2. Parameter scaling (learned from data) - now fixed to accept scalar
    param_scale = paramNet(alpha_new);
    
    % 3. Physics-based correction with new parameter
    da_dt_physics = alpha_new * A_rom * a_pinn_new(:, i);
    
    % 4. Calculate physics residual
    a_next_nn = a_pinn_new(:, i) + dt * param_scale * da_dt_nn;
    residual = physicsResidual(a_next_nn, a_pinn_new(:, i), Phi_r, A_rom, alpha_new, dt);
    
    % 5. Adapt physics weight based on residual (more physics for new parameter)
    physics_weight = 0.4 + 0.3 * min(residual, 1);  % Higher base weight for parameter variation
    
    % 6. Combine NN and physics (weighted average)
    da_dt_combined = (1-physics_weight) * param_scale * da_dt_nn + physics_weight * da_dt_physics;
    
    % 7. Update ROM coefficients
    a_pinn_new(:, i+1) = a_pinn_new(:, i) + dt * da_dt_combined;
    
    % Print residual for every 20th step
    if mod(i, 20) == 0
        fprintf('New param time step %d, Physics residual: %.4e\n', i, residual);
    end
end

% Reconstruct full solution from ROM coefficients
U_pinn_new = Phi_r * a_pinn_new;
plot_times = round(linspace(1,nt,5));
% Plot comparison for new parameter
figure;
for i = 1:length(plot_times)
    t_idx = plot_times(i);
    
    subplot(length(plot_times), 1, i);
    plot(x, U_new(:, t_idx), 'b-', 'LineWidth', 2);
    hold on;
    plot(x, U_pinn_new(:, t_idx), 'r--', 'LineWidth', 1.5);
    
    title(sprintf('New Parameter (alpha=%.3f), t = %.3f', alpha_new, t(t_idx)));
    xlabel('x');
    ylabel('u(x,t)');
    legend('High-fidelity', 'ROM-PINN');
    grid on;
end

% Calculate and display error for new parameter
error_new = norm(U_new - U_pinn_new, 'fro') / norm(U_new, 'fro');
fprintf('Relative error for new parameter: %.4e\n', error_new);