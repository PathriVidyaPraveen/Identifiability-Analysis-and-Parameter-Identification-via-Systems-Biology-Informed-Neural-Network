"""
================================================================================
 PHASE 2 — Forward ODE Simulation, Validation & Out-of-Sample Forecasting
 Paper   : Daneker, Zhang, Karniadakis, Lu — arXiv:2202.01723
 Purpose : (a) Simulate using INFERRED parameters from Phase 1
           (b) Compare all 6 state variables against ground-truth (nominal params)
           (c) Replicate the out-of-sample forecast: new meal at t=2000 min, m=100g

 HOW TO RUN
 ----------
   # Run phase1 first (or use the bundled demo params below if skipping phase 1)
   python phase2_simulation_forecasting.py

 INPUT FILES
 -----------
   inferred_params.npy  — output of phase1_sbinn_estimation.py
                          (falls back to nominal values if file not found)

 OUTPUT FILES
 ------------
   fig_validation_6state.png   — 6-panel state comparison (inferred vs nominal)
   fig_glucose_forecast.png    — glucose forecast including t=2000 meal event
   fig_all_states_forecast.png — all states over [0, 3000] min
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import solve_ivp
import os

# ═════════════════════════════════════════════════════════════════════════════
# 0.  PARAMETERS
# ═════════════════════════════════════════════════════════════════════════════
# Nominal parameters (ground truth)
NOM = dict(
    Vp=3.0,  Vi=11.0, Vg=10.0,
    E=0.2,   tp=6.0,  ti=100.0, td=12.0,  k=0.0083,
    Rm=209.0,a1=6.6,  C1=300.0, C2=144.0, C3=100.0,
    C4=80.0, C5=26.0, Ub=72.0,  U0=4.0,   Um=90.0,
    Rg=180.0,alpha=7.5, beta=1.772,
)

# Load inferred params from Phase 1 (or fall back to nominal for demo)
INFERRED_FILE = "inferred_params.npy"
if os.path.exists(INFERRED_FILE):
    raw = np.load(INFERRED_FILE, allow_pickle=True).item()
    INF = {**NOM, **raw}    # overlay inferred values; keep fixed ones from NOM
    print(f"Loaded inferred parameters from {INFERRED_FILE}")
else:
    # ── DEMO MODE: use paper's published inferred values (Table 1) ────────────
    print(f"'{INFERRED_FILE}' not found — using paper's published inferred values.")
    INF = {**NOM,
           'E':0.201, 'tp':5.99,  'ti':101.20, 'td':11.98, 'k':0.00833,
           'Rm':208.62,'a1':6.59, 'C1':301.26, 'C4':78.76, 'C5':25.94,
           'Ub':71.33, 'Rg':179.86,'alpha':7.54,'beta':1.783,
           # C2, U0, Um — paper notes these are C3-scaled; use nominal C3=100
           'C2':37.65, 'U0':0.0406*100, 'Um':0.890*100,
    }

Vp = NOM['Vp']; Vi = NOM['Vi']; Vg = NOM['Vg']

# ═════════════════════════════════════════════════════════════════════════════
# 1.  ODE SYSTEM (reusable function)
# ═════════════════════════════════════════════════════════════════════════════
def nutritional_driver(t, meal_t, meal_m, k):
    """I_G(t) [mg/min]."""
    val = 0.0
    for tj, mj in zip(meal_t, meal_m):
        dt = t - tj
        if dt >= 0.0:
            val += mj * 1000.0 * k * np.exp(-k * dt)
    return val

def ultradian_rhs(t, x, p, meal_t, meal_m):
    """
    ODE RHS in total mass units:
      x = [Ip(mU), Ii(mU), G(mg), h1(mU), h2(mU), h3(mU)]
    """
    Ip, Ii, G, h1, h2, h3 = x
    Ii_safe = max(Ii, 1e-12)
    Ii_c    = Ii_safe / p['Vi']

    kap = (1.0/p['C4']) * (1.0/p['Vi'] + 1.0/(p['E']*p['ti']))

    f1 = p['Rm'] / (1.0 + np.exp(-G/(p['Vg']*p['C1']) + p['a1']))
    f2 = p['Ub'] * (1.0 - np.exp(-G/(p['C2']*p['Vg'])))
    f3 = (1.0/(p['C3']*p['Vg'])) * (p['U0'] + p['Um']/(1.0+(kap*Ii_c)**(-p['beta'])))
    f4 = p['Rg'] / (1.0 + np.exp(p['alpha']*(h3/(p['C5']*p['Vp']) - 1.0)))
    IG = nutritional_driver(t, meal_t, meal_m, p['k'])

    dIp = f1 - p['E']*(Ip/p['Vp'] - Ii/p['Vi']) - Ip/p['tp']
    dIi = p['E']*(Ip/p['Vp'] - Ii/p['Vi']) - Ii/p['ti']
    dG  = f4 + IG - f2 - f3*G
    dh1 = (Ip - h1)/p['td']
    dh2 = (h1 - h2)/p['td']
    dh3 = (h2 - h3)/p['td']
    return [dIp, dIi, dG, dh1, dh2, dh3]

def simulate(params, meal_t, meal_m, t_end, t_eval=None):
    """Solve the Ultradian ODE; returns (t, sol_y in mass units)."""
    x0 = [
        12.0  * params['Vp'],     # Ip: 12 µU/ml × Vp L = 36 mU
        4.0   * params['Vi'],     # Ii: 4  µU/ml × Vi L = 44 mU
        110.0 * params['Vg']*10,  # G:  110 mg/dl × 100 dl = 11000 mg
        0.0, 0.0, 0.0,
    ]
    sol = solve_ivp(
        fun      = lambda t, x: ultradian_rhs(t, x, params, meal_t, meal_m),
        t_span   = (0.0, t_end),
        y0       = x0,
        method   = "Radau",
        max_step = 2.0,
        rtol     = 1e-8,
        atol     = 1e-10,
        t_eval   = t_eval,
        dense_output = (t_eval is None),
    )
    assert sol.success, f"ODE solve failed: {sol.message}"
    return sol

def to_display(sol):
    """Convert solution from mass units → display units (concentration)."""
    return {
        'Ip_uU': sol.y[0] / Vp,          # mU/L = µU/ml
        'Ii_uU': sol.y[1] / Vi,
        'G_dl' : sol.y[2] / (Vg*10),     # mg → mg/dl
        'h1'   : sol.y[3],               # mU (filter, no canonical concentration)
        'h2'   : sol.y[4],
        'h3'   : sol.y[5],
    }

# ═════════════════════════════════════════════════════════════════════════════
# 2.  SIMULATE  (training window: t ∈ [0, 1800] min)
# ═════════════════════════════════════════════════════════════════════════════
MEAL_T  = [300, 650, 1100]
MEAL_M  = [60,  40,  50  ]
T_TRAIN = 1800.0
t_eval  = np.linspace(0, T_TRAIN, 5000)

print("Simulating with NOMINAL parameters …")
sol_nom = simulate(NOM, MEAL_T, MEAL_M, T_TRAIN, t_eval)
disp_nom = to_display(sol_nom)

print("Simulating with INFERRED parameters …")
sol_inf = simulate(INF, MEAL_T, MEAL_M, T_TRAIN, t_eval)
disp_inf = to_display(sol_inf)

# ═════════════════════════════════════════════════════════════════════════════
# 3.  FIGURE A — 6-Panel Validation Plot
# ═════════════════════════════════════════════════════════════════════════════
COLORS = {
    'nom': '#14538F',   # deep blue   — nominal / ground truth
    'inf': '#CC2222',   # dark red    — inferred
    'meal':'#228B22',   # green       — meal markers
}

state_keys  = ['Ip_uU','Ii_uU','G_dl','h1','h2','h3']
state_labels = [r'$I_p$ (µU/ml)', r'$I_i$ (µU/ml)', r'$G$ (mg/dl)',
                r'$h_1$ (mU)',     r'$h_2$ (mU)',     r'$h_3$ (mU)']
state_titles = ['Plasma Insulin $I_p$',   'Interstitial Insulin $I_i$',
                'Glucose $G$',
                'Delay Filter $h_1$',     'Delay Filter $h_2$',     'Delay Filter $h_3$']

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.patch.set_facecolor('white')

for ax, key, lbl, ttl in zip(axes.flat, state_keys, state_labels, state_titles):
    ax.plot(t_eval, disp_nom[key], color=COLORS['nom'], lw=2.0,
            label='Nominal (ground truth)')
    ax.plot(t_eval, disp_inf[key], color=COLORS['inf'], lw=1.8, ls='--',
            label='Inferred (SBINN)')
    for tj in MEAL_T:
        ax.axvline(tj, ls=':', color=COLORS['meal'], lw=1.2, alpha=0.85)
    ax.set_xlabel('t (min)', fontsize=11)
    ax.set_ylabel(lbl, fontsize=11)
    ax.set_title(ttl, fontsize=12, fontweight='bold')
    ax.set_xlim(0, T_TRAIN)
    ax.grid(True, alpha=0.3, ls=':')
    ax.tick_params(labelsize=10)

# Compute and print per-state RMSE
print("\n── Validation RMSE (inferred vs nominal) ──")
for key, lbl_str in zip(state_keys, ['Ip','Ii','G','h1','h2','h3']):
    rmse = np.sqrt(np.mean((disp_inf[key] - disp_nom[key])**2))
    print(f"  {lbl_str:<4} RMSE = {rmse:.4f}")

handles = [
    plt.Line2D([0],[0], color=COLORS['nom'], lw=2,   label='Nominal (ground truth)'),
    plt.Line2D([0],[0], color=COLORS['inf'], lw=1.8, ls='--', label='Inferred (SBINN)'),
    plt.Line2D([0],[0], color=COLORS['meal'],lw=1.2, ls=':', label='Meal events'),
]
fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=11,
           bbox_to_anchor=(0.5, -0.01))
fig.suptitle('SBINN Parameter Estimation: Inferred vs Nominal State Variables\n'
             r'$t \in [0,\,1800]$ min  |  3 meal events at $t=300,650,1100$ min',
             fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0.04, 1, 0.95])
plt.savefig('fig_validation_6state.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: fig_validation_6state.png")

# ═════════════════════════════════════════════════════════════════════════════
# 4.  OUT-OF-SAMPLE FORECASTING  (t ∈ [0, 3000], new meal at t=2000, m=100g)
# ═════════════════════════════════════════════════════════════════════════════
MEAL_T_FORE  = [300, 650, 1100, 2000]
MEAL_M_FORE  = [60,  40,  50,  100 ]
T_FORE       = 3000.0
t_eval_fore  = np.linspace(0, T_FORE, 8000)

print("\nSimulating forecast with NOMINAL parameters …")
sol_nom_fore = simulate(NOM, MEAL_T_FORE, MEAL_M_FORE, T_FORE, t_eval_fore)
disp_nom_fore = to_display(sol_nom_fore)

print("Simulating forecast with INFERRED parameters …")
sol_inf_fore = simulate(INF, MEAL_T_FORE, MEAL_M_FORE, T_FORE, t_eval_fore)
disp_inf_fore = to_display(sol_inf_fore)

# ── Figure B: Glucose forecast ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
mask_tr = t_eval_fore <= T_TRAIN
mask_fc = t_eval_fore >= T_TRAIN

# Shaded regions
ax.axvspan(0, T_TRAIN, alpha=0.06, color='#14538F', label='Training window')
ax.axvspan(T_TRAIN, T_FORE, alpha=0.06, color='#CC2222', label='Forecast window')
ax.axvline(T_TRAIN, ls='-', color='gray', lw=1.5, alpha=0.8)
ax.text(T_TRAIN+30, ax.get_ylim()[0] if ax.get_ylim()[0] else 80,
        r'$t=1800$', fontsize=9, color='gray')

# Trajectories
ax.plot(t_eval_fore, disp_nom_fore['G_dl'], color=COLORS['nom'], lw=2,
        label='Nominal G(t)')
ax.plot(t_eval_fore[mask_tr], disp_inf_fore['G_dl'][mask_tr],
        color=COLORS['inf'], lw=1.8, ls='--', label='SBINN (trained)')
ax.plot(t_eval_fore[mask_fc], disp_inf_fore['G_dl'][mask_fc],
        color='#FF6600', lw=2, ls='--', label='SBINN (forecast)')

for tj in MEAL_T_FORE:
    ax.axvline(tj, ls=':', color=COLORS['meal'], lw=1.2, alpha=0.8)

ax.set_xlabel('t (min)', fontsize=13)
ax.set_ylabel('G (mg/dl)', fontsize=13)
ax.set_title('Out-of-Sample Glucose Forecast\n'
             r'New meal: $t_j=2000$ min, $m_j=100$ g',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11, ncol=2)
ax.grid(True, alpha=0.3, ls=':')
ax.set_xlim(0, T_FORE)
plt.tight_layout()
plt.savefig('fig_glucose_forecast.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig_glucose_forecast.png")

# ── Figure C: All 4 states over full forecast window (matches paper Fig 13) ───
fig = plt.figure(figsize=(14, 10))
gs  = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.32)

panels = [
    (gs[0,0], 'Ip_uU', r'$I_p$ (µU/ml)', 'Plasma Insulin $I_p$'),
    (gs[0,1], 'Ii_uU', r'$I_i$ (µU/ml)', 'Interstitial Insulin $I_i$'),
    (gs[1,0], 'G_dl',  r'$G$ (mg/dl)',    'Glucose $G$'),
]

# Compute I_G for plotting
IG_nom  = np.array([sum(mj*1000*NOM['k']*np.exp(-NOM['k']*(t-tj))
                        if t>=tj else 0
                        for tj,mj in zip(MEAL_T_FORE,MEAL_M_FORE))
                    for t in t_eval_fore])
IG_inf  = np.array([sum(mj*1000*INF['k']*np.exp(-INF['k']*(t-tj))
                        if t>=tj else 0
                        for tj,mj in zip(MEAL_T_FORE,MEAL_M_FORE))
                    for t in t_eval_fore])

panels.append((gs[1,1], None, r'$I_G$ (mg/min)', 'Nutritional Driver $I_G$'))

for idx, (gsc, key, lbl, ttl) in enumerate(panels):
    ax = fig.add_subplot(gsc)
    if key is not None:
        ax.plot(t_eval_fore, disp_nom_fore[key], color=COLORS['nom'], lw=2, label='Nominal')
        ax.plot(t_eval_fore, disp_inf_fore[key], color=COLORS['inf'], lw=1.8, ls='--',
                label='SBINN')
    else:   # I_G panel
        ax.plot(t_eval_fore, IG_nom, color=COLORS['nom'], lw=2, label='Nominal')
        ax.plot(t_eval_fore, IG_inf, color=COLORS['inf'], lw=1.8, ls='--', label='SBINN')

    ax.axvline(T_TRAIN, ls='-', color='gray', lw=1.2, alpha=0.7)
    for tj in MEAL_T_FORE:
        ax.axvline(tj, ls=':', color=COLORS['meal'], lw=1.1, alpha=0.7)
    ax.set_xlabel('t (min)', fontsize=11)
    ax.set_ylabel(lbl, fontsize=11)
    ax.set_title(ttl, fontsize=12, fontweight='bold')
    ax.set_xlim(0, T_FORE)
    ax.grid(True, alpha=0.3, ls=':')
    if idx == 0:
        ax.legend(fontsize=10)

fig.suptitle('Inferred Dynamics and Forecasting via SBINN\n'
             r'Trained on $t\in[0,1800]$; forecast meal at $t_j=2000$ min',
             fontsize=13, fontweight='bold')
plt.savefig('fig_all_states_forecast.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig_all_states_forecast.png")

# ═════════════════════════════════════════════════════════════════════════════
# 5.  QUANTITATIVE FORECAST ACCURACY
# ═════════════════════════════════════════════════════════════════════════════
mask_new = (t_eval_fore >= 1800) & (t_eval_fore <= 2400)
G_nom_new = disp_nom_fore['G_dl'][mask_new]
G_inf_new = disp_inf_fore['G_dl'][mask_new]
rmse_fore = np.sqrt(np.mean((G_inf_new - G_nom_new)**2))
peak_err  = abs(G_inf_new.max() - G_nom_new.max())

print("\n── Forecast accuracy (t ∈ [1800, 2400] min) ──")
print(f"  RMSE glucose forecast : {rmse_fore:.3f} mg/dl")
print(f"  Peak glucose error    : {peak_err:.3f} mg/dl")
print(f"  Nominal peak G        : {G_nom_new.max():.1f} mg/dl")
print(f"  SBINN peak G          : {G_inf_new.max():.1f} mg/dl")

print("\n✓  Phase 2 complete.")
