%% ================================================================
%%  Ultradian Glucose-Insulin Model — MATLAB Simulation (ode45)
%%  Based on: Daneker, Zhang, Karniadakis, Lu (arXiv:2202.01723)
%%  Original ODE model: Sturis et al. (1991)
%%
%%  *** UNIT CONVENTION (critical — read before modifying) ***
%%
%%  The ODE system works in TOTAL MASS units (not concentrations):
%%    Ip  [mU]      total plasma insulin mass
%%    Ii  [mU]      total interstitial insulin mass
%%    G   [mg]      total plasma glucose mass
%%    h1,h2,h3 [mU] three-stage filter states
%%
%%  The paper states initial conditions as concentrations:
%%    Ip(0)=12 uU/ml, Ii(0)=4 uU/ml, G(0)=110 mg/dl
%%
%%  These must be converted to total mass for the ODE:
%%    Ip(0) = 12 [uU/ml] * Vp[L] * 1000[ml/L] / 1e6 * 1e3 = Ip_conc/Vp [mU]
%%         => 12 [uU/ml] * Vp [L]  =  12*3 = 36 mU  (since 1 uU/ml*L = 1 mU)
%%    Ii(0) = 4  * Vi = 4*11  = 44 mU
%%    G(0)  = 110 [mg/dl] * Vg[L] * 10[dl/L] = 110*10*10 = 11000 mg
%%
%%  For plotting, convert back: G_plot = G[mg] / (Vg*10) [mg/dl]
%%                               Ip_plot = Ip[mU] / Vp [uU/ml]
%%                               Ii_plot = Ii[mU] / Vi [uU/ml]
%%
%%  Simulates from t=0 to t=1800 min with three meal events.
%%  Produces a 2x3 subplot of all state variables in physical units.
%% ================================================================

clear; clc; close all;

%% ================================================================
%% 1.  MODEL PARAMETERS  (nominal values, Table 1 of paper)
%% ================================================================
p.Vp    =   3;       % [l]            Plasma insulin distribution volume
p.Vi    =  11;       % [l]            Remote insulin compartment volume
p.Vg    =  10;       % [l]            Glucose space volume
p.E     =   0.2;     % [l/min]        Insulin exchange rate constant
p.tp    =   6;       % [min]          Plasma insulin degradation time constant
p.ti    = 100;       % [min]          Remote insulin degradation time constant
p.td    =  12;       % [min]          Three-stage filter delay constant
p.k     =   0.0083;  % [1/min]        Nutritional driver decay constant
p.Rm    = 209;       % [mU/min]       Max insulin secretion rate
p.a1    =   6.6;     % [-]            Sigmoid offset for insulin production
p.C1    = 300;       % [mg/l]         Glucose threshold — insulin production
p.C2    = 144;       % [mg/l]         Glucose threshold — f2
p.C3    = 100;       % [mg/l]         Glucose scaling  — f3
p.C4    =  80;       % [mU/l]         kappa scaling
p.C5    =  26;       % [mU/l]         Glucose production threshold
p.Ub    =  72;       % [mg/min]       Max insulin-independent glucose uptake
p.U0    =   4;       % [mg/min]       Basal insulin-dependent glucose uptake
p.Um    =  90;       % [mg/min]       Max insulin-dependent glucose uptake
p.Rg    = 180;       % [mg/min]       Max hepatic glucose production
p.alpha =   7.5;     % [-]            Steepness of f4 sigmoid
p.beta  =   1.772;   % [-]            Cooperativity exponent in f3

% Derived constant kappa = (1/C4)*(1/Vi + 1/(E*ti))
p.kappa = (1/p.C4) * (1/p.Vi + 1/(p.E * p.ti));

%% ================================================================
%% 2.  NUTRITIONAL EVENTS  (paper: (300,60), (650,40), (1100,50))
%% ================================================================
p.meal_t = [300, 650, 1100];   % [min]  meal times
p.meal_m = [ 60,  40,   50];   % [g]    carbohydrate quantities

%  Nutritional driver formula (Eq. 2c of paper):
%      I_G(t) = sum_j  m_j[g] * 1000[mg/g] * k * exp(-k*(t - t_j))  for t >= t_j
%  Units: [g]*[mg/g]*[1/min] = [mg/min]  ✓

%% ================================================================
%% 3.  INITIAL CONDITIONS in TOTAL MASS units
%%
%%  Paper states concentrations: [12 uU/ml, 4 uU/ml, 110 mg/dl, 0, 0, 0]
%%  Convert:
%%    Ip(0) = 12 [uU/ml] * Vp [L]       = 12*3   = 36   mU
%%    Ii(0) =  4 [uU/ml] * Vi [L]       =  4*11  = 44   mU
%%    G(0)  = 110[mg/dl] * Vg[L]*10     = 110*100= 11000 mg
%%    h1=h2=h3=0
%% ================================================================
Ip0 = 12  * p.Vp;           %  36 mU
Ii0 =  4  * p.Vi;           %  44 mU
G0  = 110 * p.Vg * 10;      % 11000 mg
x0  = [Ip0; Ii0; G0; 0; 0; 0];

fprintf('Initial conditions (mass units):\n');
fprintf('  Ip0 = %.0f mU  (= 12 uU/ml * Vp)\n', Ip0);
fprintf('  Ii0 = %.0f mU  (= 4  uU/ml * Vi)\n', Ii0);
fprintf('  G0  = %.0f mg  (= 110 mg/dl * Vg * 10 dl/L)\n\n', G0);

%% ================================================================
%% 4.  TIME SPAN
%% ================================================================
tspan = [0, 1800];   % [min]  ~one day

%% ================================================================
%% 5.  ODE SOLVER  (ode15s is better for stiff systems like this)
%% ================================================================
opts = odeset('RelTol', 1e-8, 'AbsTol', 1e-10, 'MaxStep', 1);

% Use ode15s (stiff solver) — much faster and more reliable than ode45 here
[t, x] = ode45(@(t,x) ultradian_ode(t, x, p), tspan, x0, opts);

%% ================================================================
%% 6.  CONVERT BACK TO CONCENTRATION / DISPLAY UNITS
%% ================================================================
Ip_conc = x(:,1) / p.Vp;           % [mU] / [L]  = [mU/L] = [uU/ml]
Ii_conc = x(:,2) / p.Vi;           % uU/ml
G_mgdl  = x(:,3) / (p.Vg * 10);   % [mg] / ([L]*10[dl/L]) = mg/dl
h1      = x(:,4);                  % mU  (filter states, display as-is)
h2      = x(:,5);
h3      = x(:,6);

% Nutritional driver for optional plotting
IG_plot = arrayfun(@(tt) compute_IG(tt, p), t);

%% ================================================================
%% 7.  SUMMARY STATISTICS
%% ================================================================
fprintf('--- Simulation Summary ---\n');
fprintf('Time steps solved: %d\n\n', length(t));
fprintf('State variable ranges (display units):\n');
fprintf('  Ip : %6.1f – %6.1f  uU/ml   (paper Fig 13: ~0–700 peak)\n', min(Ip_conc), max(Ip_conc));
fprintf('  Ii : %6.1f – %6.1f  uU/ml\n', min(Ii_conc), max(Ii_conc));
fprintf('  G  : %6.1f – %6.1f  mg/dl   (paper Fig  2:  80–200)\n', min(G_mgdl), max(G_mgdl));
fprintf('  h1 : %6.1f – %6.1f  mU\n', min(h1), max(h1));
fprintf('  h2 : %6.1f – %6.1f  mU\n', min(h2), max(h2));
fprintf('  h3 : %6.1f – %6.1f  mU\n', min(h3), max(h3));
fprintf('\nFinal values at t = 1800 min:\n');
fprintf('  G  = %.2f mg/dl\n', G_mgdl(end));
fprintf('  Ip = %.2f uU/ml\n', Ip_conc(end));

%% ================================================================
%% 8.  PLOT — 2x3 subplot figure
%% ================================================================
fig_width = 800; 
fig_height = 500;

figure('Name','Ultradian Glucose-Insulin Model', ...
       'Units','pixels', ...
       'Position',[50, 50, fig_width, fig_height]);

% Colour palette
C = {[0.08 0.33 0.62],   ... % deep blue   — Ip
     [0.12 0.58 0.30],   ... % green        — Ii
     [0.75 0.10 0.10],   ... % dark red     — G
     [0.52 0.18 0.56],   ... % purple       — h1
     [0.88 0.45 0.00],   ... % orange       — h2
     [0.00 0.58 0.68]};      % teal         — h3

meal_color = [0.85 0.15 0.15];  % red dashes for meal lines

% Data to plot and axis labels
state_data  = {Ip_conc, Ii_conc, G_mgdl, h1, h2, h3};
ylabels     = {'$I_p$ ($\mu$U/ml)', '$I_i$ ($\mu$U/ml)', ...
               '$G$ (mg/dl)', '$h_1$ (mU)', '$h_2$ (mU)', '$h_3$ (mU)'};
panel_titles = {'Plasma Insulin $I_p$',   'Interstitial Insulin $I_i$', ...
                'Glucose Concentration $G$', ...
                'Delay Filter $h_1$ (Stage 1)', ...
                'Delay Filter $h_2$ (Stage 2)', ...
                'Delay Filter $h_3$ (Stage 3)'};

ax = gobjects(6,1);
for k_sub = 1:6
    ax(k_sub) = subplot(2, 3, k_sub);

    % Main trajectory
    plot(t, state_data{k_sub}, 'Color', C{k_sub}, 'LineWidth', 2.0);
    hold on;

    % Vertical meal-event markers
    for j = 1:length(p.meal_t)
        xline(p.meal_t(j), '--', 'Color', meal_color, ...
              'LineWidth', 1.3, 'Alpha', 0.8);
    end

    xlabel('$t$ (min)',         'Interpreter','latex', 'FontSize', 12);
    ylabel(ylabels{k_sub},      'Interpreter','latex', 'FontSize', 12);
    title(panel_titles{k_sub},  'Interpreter','latex', 'FontSize', 13, ...
          'FontWeight','bold');
    xlim([0 1800]);
    grid on;  box on;
    set(ax(k_sub), 'FontSize', 11, 'TickLabelInterpreter','latex');
end

% Shared legend (attach to subplot 1)
legend(ax(1), ...
    {['$I_p(t)$'],  'Meal events ($t$=300, 650, 1100 min)'}, ...
    'Interpreter','latex', 'Location','northeast', 'FontSize', 10);

% Super-title
sgtitle({'Ultradian Glucose--Insulin Model: State Variable Dynamics', ...
         'Daneker, Zhang, Karniadakis, Lu (arXiv:2202.01723)  |  ' ...
         'Sturis et al. (1991)  |  ode45, $t \in [0,1800]$ min'}, ...
        'Interpreter','latex', 'FontSize', 14, 'FontWeight','bold');

% Tighten layout
set(gcf,'Color','white');

%% ================================================================
%% 9.  SAVE FIGURE
%% ================================================================
outfile = fullfile(pwd, 'ultradian_dynamics_corrected.png');
exportgraphics(gcf, outfile, 'Resolution', 200);
fprintf('\nFigure saved to:\n  %s\n', outfile);


%% ================================================================
%%  LOCAL FUNCTIONS
%% ================================================================

function dxdt = ultradian_ode(t, x, p)
%ULTRADIAN_ODE  RHS of the Ultradian glucose-insulin ODE system.
%
%   State vector (ALL IN TOTAL MASS UNITS):
%     x(1) = Ip  [mU]   total plasma insulin
%     x(2) = Ii  [mU]   total interstitial insulin
%     x(3) = G   [mg]   total plasma glucose
%     x(4) = h1  [mU]   delay filter stage 1
%     x(5) = h2  [mU]   delay filter stage 2
%     x(6) = h3  [mU]   delay filter stage 3
%
%   Equations (1a)-(1d) and (2a)-(2c) of Daneker et al. 2022.

    Ip = x(1);
    Ii = x(2);
    G  = x(3);
    h1 = x(4);
    h2 = x(5);
    h3 = x(6);

    % Guard against Ii <= 0 (can occur transiently at t=0)
    Ii_safe = max(Ii, 1e-12);

    % ----- Nonlinear functions f1 – f4  (Eq. 2a–2b) -----

    % f1(G): Insulin production  [mU/min]
    %   Arg of exp is dimensionless: G[mg]/(Vg[L]*C1[mg/L]) = dimensionless ✓
    f1 = p.Rm / (1 + exp(-G / (p.Vg * p.C1) + p.a1));

    % f2(G): Insulin-independent glucose utilisation  [mg/min]
    %   G/(C2*Vg) = [mg]/([mg/L]*[L]) = dimensionless ✓
    f2 = p.Ub * (1 - exp(-G / (p.C2 * p.Vg)));

    % f3(Ii): Insulin-dependent glucose utilisation coefficient  [1/min]
    %   kappa [1/(mU/L)] * Ii [mU] / Vi [L] => need Ii in mU/L for kappa
    %   kappa*Ii_conc where Ii_conc = Ii/Vi [mU/L]
    %   (kappa*Ii_conc)^(-beta) is dimensionless ✓
    Ii_conc = Ii_safe / p.Vi;     % [mU/L]  — concentration for f3
    f3 = (1 / (p.C3 * p.Vg)) * ...
         (p.U0 + p.Um / (1 + (p.kappa * Ii_conc)^(-p.beta)));

    % f4(h3): Delayed hepatic glucose production  [mg/min]
    %   h3/Vp [mU/L] / C5 [mU/L] = dimensionless ✓
    f4 = p.Rg / (1 + exp(p.alpha * (h3 / (p.C5 * p.Vp) - 1)));

    % ----- Nutritional driver I_G(t)  [mg/min]  (Eq. 2c) -----
    IG = compute_IG(t, p);

    % ----- ODE right-hand sides  (Eqs. 1a–1d) -----

    % Eq. (1a): dIp/dt  [mU/min]
    %   f1 [mU/min], E*(Ip/Vp - Ii/Vi) [L/min * mU/L = mU/min], Ip/tp [mU/min] ✓
    dIp = f1 - p.E * (Ip/p.Vp - Ii/p.Vi) - Ip/p.tp;

    % Eq. (1b): dIi/dt  [mU/min]
    dIi = p.E * (Ip/p.Vp - Ii/p.Vi) - Ii/p.ti;

    % Eq. (1d): dG/dt  [mg/min]
    %   f4 [mg/min], IG [mg/min], f2 [mg/min], f3*G [1/min * mg = mg/min] ✓
    dG = f4 + IG - f2 - f3 * G;

    % Eq. (1c): Three-stage delay filter  [mU/min]
    %   (Ip - h1)/td: [mU]/[min] = [mU/min] ✓
    dh1 = (Ip - h1) / p.td;
    dh2 = (h1 - h2) / p.td;
    dh3 = (h2 - h3) / p.td;

    dxdt = [dIp; dIi; dG; dh1; dh2; dh3];
end


function IG = compute_IG(t, p)
%COMPUTE_IG  Nutritional driver I_G(t) in [mg/min].
%
%   I_G(t) = sum_{j: t >= t_j}  m_j[g] * 1000[mg/g] * k[1/min] * exp(-k*(t-t_j))
%
%   Each meal contributes an exponentially decaying pulse of glucose input.
%   The Heaviside causality is enforced by the dt>=0 check.

    IG = 0;
    for j = 1:length(p.meal_t)
        dt = t - p.meal_t(j);
        if dt >= 0
            IG = IG + p.meal_m(j) * 1000 * p.k * exp(-p.k * dt);
        end
    end
end
