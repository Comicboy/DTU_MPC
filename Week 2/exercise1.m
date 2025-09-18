%% Four Tank System Simulation and Linearization

clear; clc; close all

%% Parameters (defined once)
a = [1.2272; 1.2272; 1.2272; 1.2272];      % [cm^2] Outlet pipe cross-sectional areas
At = [380.1327; 380.1327; 380.1327; 380.1327]; % [cm^2] Tank cross-sectional areas
gamma = [0.45; 0.40];                      % Valve positions [-]
g = 981;                                  % [cm/s^2] Gravity
rho = 1.0;                                % [g/cm^3] Density of water

p = [a; At; gamma; g; rho];

%% Initial conditions and inputs
t0 = 0;             % Initial time [s]
tfin = 20*60;         % Final time [s]

% Initial masses (guess)
m0 = zeros(4,1);    % [g] Initial mass in each tank

% Steady-state input flow rates
F1_ss = 250;        % [cm^3/s]
F2_ss = 325;        % [cm^3/s]
us = [F1_ss; F2_ss];

%% Find steady state (equilibrium) of nonlinear system
xs0 = 5000*ones(4,1); % Initial guess for steady state masses

opts = optimoptions('fsolve','Display','off');
xs = fsolve(@(x) FourTankSystemWrap(x, us, p), xs0, opts);

% Compute steady state output (levels)
hs = FourTankSystemOutput(xs, p);

fprintf('Steady state liquid levels (cm):\n')
disp(hs')

%% Linearize system around steady state

T = (At ./ a) .* sqrt(2*hs / g);

A = [-1/T(1)      0       1/T(3)      0;
      0       -1/T(2)       0      1/T(4);
      0          0        -1/T(3)     0;
      0          0          0      -1/T(4)];

B = [rho*gamma(1)      0;
         0        rho*gamma(2);
         0     rho*(1 - gamma(2));
    rho*(1 - gamma(1)) 0];

C = diag(1 ./ (rho * At));

D = zeros(size(C,1), size(B,2));

fprintf('Linearized system matrices:\n')
disp('A ='); disp(A)
disp('B ='); disp(B)
disp('C ='); disp(C)

%% Discretize system with sample time Ts = 4 seconds
Ts = 4; % sample time [s]

% Create continuous-time state-space system (no direct feedthrough)
sys_cont = ss(A, B, C, D);

% Discretize using zero-order hold
sys_disc = c2d(sys_cont, Ts, 'zoh');

Ad = sys_disc.A;
Bd = sys_disc.B;
Cd = sys_disc.C;
Dd = sys_disc.D;

fprintf('Discrete-time system matrices with Ts = %g seconds:\n', Ts);
disp('Ad ='); disp(Ad)
disp('Bd ='); disp(Bd)
disp('Cd ='); disp(Cd)
disp('Dd ='); disp(Dd)

%% Transfer functions for the continous and discrete systems
% Compute transfer functions for continuous and discrete systems
sys_cont_tf = tf(sys_cont);
sys_disc_tf = tf(sys_disc);

% Compute gains, poles and zeros for each system
poles_c = pole(sys_cont);
zeros_c = pole(sys_cont);
gain_c = dcgain(sys_cont);

poles_d = pole(sys_disc);
zeros_d = pole(sys_disc);
gain_d = dcgain(sys_disc); 

%% Simulation parameters
x0 = xs; % start from steady state masses
F2 = F2_ss;

% Step increases in F1 input to test responses
stepPercents = [0.05, 0.10, 0.25];
colors = ['b', 'r', 'g'];
tankLabels = {'Tank 1', 'Tank 2', 'Tank 3', 'Tank 4'};

%% Plot tank levels for nonlinear model step responses
figure;
hLinesStep = gobjects(length(stepPercents), 4); % Preallocate for legend handles

for i = 1:length(stepPercents)
    F1_step = F1_ss * (1 + stepPercents(i));
    u = [F1_step; F2];
    
    [T_nl, X_nl] = ode15s(@(t,x) FourTankSystem(t, x, u, p), [t0 tfin], x0);
    
    % Compute levels
    H_nl = X_nl ./ (rho * At');
    
    for tankIdx = 1:4
        subplot(2,2,tankIdx)
        hold on
        grid on
        hLinesStep(i, tankIdx) = plot(T_nl, H_nl(:, tankIdx), colors(i), 'LineWidth', 1.5);
        xlabel('Time [s]')
        ylabel('Level [cm]')
        title(tankLabels{tankIdx})
    end
end

% Legend only in the first subplot
legend(subplot(2,2,1), hLinesStep(:,1), {'5% Step Increase', '10% Step Increase', '25% Step Increase'}, 'Location', 'best')

sgtitle('Nonlinear Model: Step Responses for F1 Step Increases')

%% Compare linear vs nonlinear step responses

figure;
hLinesComp = gobjects(length(stepPercents)*2, 4); % Preallocate handles for legend

for tankIdx = 1:4
    subplot(2,2,tankIdx)
    hold on
    grid on
    xlabel('Time [s]')
    ylabel('Level [cm]')
    title(['Linear vs Nonlinear: ' tankLabels{tankIdx}])
end

for i = 1:length(stepPercents)
    % Nonlinear simulation (redo for synchronization)
    F1_step = F1_ss * (1 + stepPercents(i));
    u = [F1_step; F2];
    [T_nl, X_nl] = ode15s(@(t,x) FourTankSystem(t, x, u, p), [t0 tfin], x0);
    H_nl = X_nl ./ (rho * At');
    
    % Linear system simulation
    delta_u = u - us;       % Input deviation from steady state
    x0_lin = zeros(4,1);   % Initial linear deviation
    
    % Define linear ODE function handle
    lin_ode = @(t,x) A*x + B*delta_u;
    
    [T_lin, X_lin] = ode15s(lin_ode, [t0 tfin], x0_lin);
    
    % Add steady state to linear states and compute levels
    X_lin_actual = X_lin + xs';  % Add steady state mass
    H_lin = X_lin_actual ./ (rho * At');
    
    % Plot nonlinear solid lines and linear dashed lines
    for tankIdx = 1:4
        subplot(2,2,tankIdx)
        hLinesComp((i-1)*2 + 1, tankIdx) = plot(T_nl, H_nl(:, tankIdx), colors(i), 'LineWidth', 1.5);
        hLinesComp((i-1)*2 + 2, tankIdx) = plot(T_lin, H_lin(:, tankIdx), [colors(i) '--'], 'LineWidth', 1.5);
    end
end

% Legend only in the first subplot
legend(subplot(2,2,1), hLinesComp(:,1), ...
    {'Nonlinear 5%', 'Linear 5%', 'Nonlinear 10%', 'Linear 10%', 'Nonlinear 25%', 'Linear 25%'}, ...
    'Location', 'best')

sgtitle('Linear vs Nonlinear Step Responses for Step Increases in F1')

%% Supporting Functions

function xdot = FourTankSystem(t, x, u, p)
    % Unpack parameters
    a = p(1:4);        % Outlet pipe cross sectional areas
    At = p(5:8);       % Tank cross sectional areas
    gamma = p(9:10);   % Valve positions
    g = p(11);         % Gravity acceleration
    rho = p(12);       % Water density
    
    % Flow rates (inputs)
    F = u;
    
    % Compute inflows
    qin = zeros(4,1);
    qin(1) = gamma(1)*F(1);
    qin(2) = gamma(2)*F(2);
    qin(3) = (1 - gamma(2))*F(2);
    qin(4) = (1 - gamma(1))*F(1);
    
    % Calculate levels
    h = x ./ (rho * At);
    
    % Calculate outflows
    qout = a .* sqrt(2 * g * h);
    
    % Mass balance differential equations
    xdot = zeros(4,1);
    xdot(1) = rho*(qin(1) + qout(3) - qout(1));
    xdot(2) = rho*(qin(2) + qout(4) - qout(2));
    xdot(3) = rho*(qin(3) - qout(3));
    xdot(4) = rho*(qin(4) - qout(4));
end

function res = FourTankSystemWrap(x, u, p)
    % Wrapper for fsolve that returns dx/dt = 0 at steady state
    dx = FourTankSystem(0, x, u, p);
    res = dx;
end

function y = FourTankSystemOutput(x, p)
    % Output function: tank liquid levels [cm]
    rho = p(12);
    At = p(5:8);
    y = x ./ (rho * At);
end
