"""
================================================================================
 PHASE 3 — Parametric Sensitivity & Monte Carlo Solution-Space Analysis
 Paper   : Daneker, Zhang, Karniadakis, Lu — arXiv:2202.01723
 Purpose : (a) 1-D sensitivity of G_max and G_eq to E and C1  (±50% range)
           (b) 2-D sensitivity heatmap (E × C1)
           (c) Monte Carlo simulation: N=1000 random parameter samples from
               Table-1 search ranges → shaded glucose solution space

 HOW TO RUN
 ----------
   python phase3_sensitivity_montecarlo.py

 OUTPUT FILES
 ------------
   fig_sensitivity_1d.png      — 1-D sensitivity curves for E and C1
   fig_sensitivity_heatmap.png — 2-D heatmap G_max(E, C1)
   fig_monte_carlo.png         — solution space (mean ± std) from N=1000 runs
   monte_carlo_summary.txt     — summary statistics
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.integrate import solve_ivp
from tqdm import tqdm

# ═════════════════════════════════════════════════════════════════════════════
# 0.  NOMINAL PARAMETERS  (Table 1 of paper)
# ═════════════════════════════════════════════════════════════════════════════
NOM = dict(
    Vp=3.0,  Vi=11.0, Vg=10.0,
    E=0.2,   tp=6.0,  ti=100.0, td=12.0,  k=0.0083,
    Rm=209.0,a1=6.6,  C1=300.0, C2=144.0, C3=100.0,
    C4=80.0, C5=26.0, Ub=72.0,  U0=4.0,   Um=90.0,
    Rg=180.0,alpha=7.5, beta=1.772,
)

Vg = NOM['Vg']

# Search ranges from Table 1 (used for Monte Carlo)
SEARCH = dict(
    E    = (0.100, 0.300),
    tp   = (4.00,  8.00 ),
    ti   = (60.0,  140.0),
    td   = (25/3,  50/3 ),
    k    = (0.00166, 0.0150),
    Rm   = (41.8,  376.0),
    a1   = (1.32,  11.9 ),
    C1   = (60.0,  540.0),
    C2   = (28.8,  259.0),
    C4   = (16.0,  144.0),
    C5   = (5.20,  46.8 ),
    Ub   = (14.4,  130.0),
    U0   = (0.800, 7.200),
    Um   = (18.0,  162.0),
    Rg   = (36.0,  324.0),
    alpha= (1.50,  13.5 ),
    beta = (0.354, 3.190),
)

MEAL_T = [300, 650, 1100]
MEAL_M = [60,  40,  50  ]
T_END  = 1800.0

# ═════════════════════════════════════════════════════════════════════════════
# 1.  ODE SOLVER  (shared utility)
# ═════════════════════════════════════════════════════════════════════════════
def nutritional_driver(t, meal_t, meal_m, k):
    val = 0.0
    for tj, mj in zip(meal_t, meal_m):
        dt = t - tj
        if dt >= 0.0:
            val += mj * 1000.0 * k * np.exp(-k * dt)
    return val

def ultradian_rhs(t, x, p):
    Ip, Ii, G, h1, h2, h3 = x
    Ii_safe = max(Ii, 1e-12)
    Ii_c    = Ii_safe / p['Vi']
    kap     = (1.0/p['C4']) * (1.0/p['Vi'] + 1.0/(p['E']*p['ti']))

    f1 = p['Rm'] / (1.0 + np.exp(-G/(p['Vg']*p['C1']) + p['a1']))
    f2 = p['Ub'] * (1.0 - np.exp(-G/(p['C2']*p['Vg'])))
    f3 = (1.0/(p['C3']*p['Vg'])) * (p['U0'] + p['Um']/(1.0+(kap*Ii_c)**(-p['beta'])))
    f4 = p['Rg'] / (1.0 + np.exp(p['alpha']*(h3/(p['C5']*p['Vp']) - 1.0)))
    IG = nutritional_driver(t, MEAL_T, MEAL_M, p['k'])

    dIp = f1 - p['E']*(Ip/p['Vp'] - Ii/p['Vi']) - Ip/p['tp']
    dIi = p['E']*(Ip/p['Vp'] - Ii/p['Vi']) - Ii/p['ti']
    dG  = f4 + IG - f2 - f3*G
    dh1 = (Ip - h1)/p['td']
    dh2 = (h1 - h2)/p['td']
    dh3 = (h2 - h3)/p['td']
    return [dIp, dIi, dG, dh1, dh2, dh3]

def make_x0(p):
    return [
        12.0  * p['Vp'],
        4.0   * p['Vi'],
        110.0 * p['Vg'] * 10,
        0.0, 0.0, 0.0,
    ]

def solve_ode(p, t_eval=None, t_end=T_END):
    """Return (success, t, G_mg/dl) or (False, None, None) on failure."""
    try:
        sol = solve_ivp(
            fun      = lambda t, x: ultradian_rhs(t, x, p),
            t_span   = (0.0, t_end),
            y0       = make_x0(p),
            method   = "Radau",
            max_step = 5.0,
            rtol     = 1e-6,
            atol     = 1e-8,
            t_eval   = t_eval,
        )
        if not sol.success:
            return False, None, None
        G_dl = sol.y[2] / (p['Vg'] * 10)
        return True, sol.t, G_dl
    except Exception:
        return False, None, None

def get_metrics(p, t_eval):
    """Return (G_max [mg/dl], G_eq [mg/dl]) = (maximum, mean of last 200 min)."""
    ok, t, G_dl = solve_ode(p, t_eval)
    if not ok:
        return np.nan, np.nan
    G_max = G_dl.max()
    # Steady-state approx: mean glucose over last 200 min of simulation
    mask  = t >= (t[-1] - 200.0)
    G_eq  = G_dl[mask].mean()
    return G_max, G_eq

t_eval_coarse = np.linspace(0, T_END, 1800)   # 1-min resolution

# ═════════════════════════════════════════════════════════════════════════════
# 2.  1-D SENSITIVITY ANALYSIS  (E and C1, each ±50% from nominal)
# ═════════════════════════════════════════════════════════════════════════════
N_SWEEP = 30   # number of points per parameter sweep

def sweep_1d(param_name, frac_range=0.50, n=N_SWEEP):
    """
    Sweep `param_name` from  (1-frac)*nominal to (1+frac)*nominal.
    Returns (values_array, Gmax_array, Geq_array).
    """
    nom_val = NOM[param_name]
    values  = np.linspace(nom_val*(1-frac_range), nom_val*(1+frac_range), n)
    Gmax_arr = np.empty(n)
    Geq_arr  = np.empty(n)
    for i, v in enumerate(values):
        p = {**NOM, param_name: v}
        Gmax_arr[i], Geq_arr[i] = get_metrics(p, t_eval_coarse)
    return values, Gmax_arr, Geq_arr

print("Running 1-D sensitivity sweep for E …")
E_vals,  E_Gmax,  E_Geq  = sweep_1d('E',  frac_range=0.50)
print("Running 1-D sensitivity sweep for C1 …")
C1_vals, C1_Gmax, C1_Geq = sweep_1d('C1', frac_range=0.50)

# ── Figure: 1-D sensitivity ───────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.patch.set_facecolor('white')

BLUE = '#14538F'; RED = '#CC2222'; ORANGE = '#E07800'

def norm_pct(arr, nom_idx):
    """Convert values to % change from nominal."""
    return (arr / arr[nom_idx] - 1.0) * 100.0

E_nom_idx  = np.argmin(np.abs(E_vals  - NOM['E']))
C1_nom_idx = np.argmin(np.abs(C1_vals - NOM['C1']))

# Panel (0,0): E → G_max
ax = axes[0,0]
ax.plot(E_vals, E_Gmax, color=BLUE, lw=2.5, marker='o', ms=4)
ax.axvline(NOM['E'], ls='--', color='gray', lw=1.3, alpha=0.8, label=f"Nominal E={NOM['E']}")
ax.axhline(E_Gmax[E_nom_idx], ls=':', color='gray', lw=1)
ax.set_xlabel('Exchange rate $E$ (L/min)', fontsize=12)
ax.set_ylabel(r'$G_{\max}$ (mg/dl)', fontsize=12)
ax.set_title(r'Effect of $E$ on Maximum Glucose $G_{\max}$', fontsize=12, fontweight='bold')
ax.legend(fontsize=10); ax.grid(True, alpha=0.3, ls=':')

# Panel (0,1): E → G_eq
ax = axes[0,1]
ax.plot(E_vals, E_Geq, color=RED, lw=2.5, marker='s', ms=4)
ax.axvline(NOM['E'], ls='--', color='gray', lw=1.3, alpha=0.8)
ax.axhline(E_Geq[E_nom_idx], ls=':', color='gray', lw=1)
ax.set_xlabel('Exchange rate $E$ (L/min)', fontsize=12)
ax.set_ylabel(r'$G_{eq}$ (mg/dl)', fontsize=12)
ax.set_title(r'Effect of $E$ on Steady-State Glucose $G_{eq}$', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, ls=':')

# Panel (1,0): C1 → G_max
ax = axes[1,0]
ax.plot(C1_vals, C1_Gmax, color=BLUE, lw=2.5, marker='o', ms=4)
ax.axvline(NOM['C1'], ls='--', color='gray', lw=1.3, alpha=0.8, label=f"Nominal C1={NOM['C1']}")
ax.axhline(C1_Gmax[C1_nom_idx], ls=':', color='gray', lw=1)
ax.set_xlabel('Glucose threshold $C_1$ (mg/L)', fontsize=12)
ax.set_ylabel(r'$G_{\max}$ (mg/dl)', fontsize=12)
ax.set_title(r'Effect of $C_1$ on Maximum Glucose $G_{\max}$', fontsize=12, fontweight='bold')
ax.legend(fontsize=10); ax.grid(True, alpha=0.3, ls=':')

# Panel (1,1): C1 → G_eq
ax = axes[1,1]
ax.plot(C1_vals, C1_Geq, color=RED, lw=2.5, marker='s', ms=4)
ax.axvline(NOM['C1'], ls='--', color='gray', lw=1.3, alpha=0.8)
ax.axhline(C1_Geq[C1_nom_idx], ls=':', color='gray', lw=1)
ax.set_xlabel('Glucose threshold $C_1$ (mg/L)', fontsize=12)
ax.set_ylabel(r'$G_{eq}$ (mg/dl)', fontsize=12)
ax.set_title(r'Effect of $C_1$ on Steady-State Glucose $G_{eq}$', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, ls=':')

fig.suptitle('1-D Parametric Sensitivity Analysis\n'
             r'$E$ and $C_1$ varied $\pm 50\%$ from nominal values',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_sensitivity_1d.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig_sensitivity_1d.png")

# Print sensitivity table
print("\n── 1-D Sensitivity Table ──")
print(f"{'E (L/min)':>14}  {'G_max':>8}  {'G_eq':>8}")
for v, gx, ge in zip(E_vals[::5], E_Gmax[::5], E_Geq[::5]):
    print(f"  {v:>10.4f}    {gx:>8.2f}  {ge:>8.2f}")

print(f"\n{'C1 (mg/L)':>14}  {'G_max':>8}  {'G_eq':>8}")
for v, gx, ge in zip(C1_vals[::5], C1_Gmax[::5], C1_Geq[::5]):
    print(f"  {v:>10.1f}    {gx:>8.2f}  {ge:>8.2f}")

# ═════════════════════════════════════════════════════════════════════════════
# 3.  2-D SENSITIVITY HEATMAP  (G_max as function of E × C1)
# ═════════════════════════════════════════════════════════════════════════════
N_2D   = 15    # grid resolution per axis (N_2D² ODE solves)
E_grid = np.linspace(NOM['E']*0.5,  NOM['E']*1.5,  N_2D)
C1_grid= np.linspace(NOM['C1']*0.5, NOM['C1']*1.5, N_2D)
Gmax_map = np.full((N_2D, N_2D), np.nan)
Geq_map  = np.full((N_2D, N_2D), np.nan)

print(f"\nRunning 2-D sensitivity grid ({N_2D}×{N_2D} = {N_2D**2} ODE solves) …")
for i, Ev in enumerate(E_grid):
    for j, C1v in enumerate(C1_grid):
        p = {**NOM, 'E': Ev, 'C1': C1v}
        gx, ge = get_metrics(p, t_eval_coarse)
        Gmax_map[i, j] = gx
        Geq_map[i, j]  = ge

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor('white')

E_pct  = (E_grid  / NOM['E']  - 1)*100
C1_pct = (C1_grid / NOM['C1'] - 1)*100

for ax, data, title, cmap in zip(
    axes,
    [Gmax_map, Geq_map],
    [r'$G_{\max}$ (mg/dl)', r'$G_{eq}$ (mg/dl)'],
    ['YlOrRd', 'Blues'],
):
    im = ax.pcolormesh(C1_pct, E_pct, data, cmap=cmap, shading='auto')
    ax.axvline(0, ls='--', color='white', lw=1.5, alpha=0.7)
    ax.axhline(0, ls='--', color='white', lw=1.5, alpha=0.7)
    ax.set_xlabel(r'$\Delta C_1$ (%)', fontsize=12)
    ax.set_ylabel(r'$\Delta E$ (%)', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    cbar = fig.colorbar(im, ax=ax, shrink=0.92)
    cbar.set_label(title, fontsize=11)
    ax.tick_params(labelsize=10)

fig.suptitle(r'2-D Sensitivity Heatmap: Effect of $E$ and $C_1$ on Glucose Metrics',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_sensitivity_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig_sensitivity_heatmap.png")

# ═════════════════════════════════════════════════════════════════════════════
# 4.  MONTE CARLO SIMULATION  (N=1000)
# ═════════════════════════════════════════════════════════════════════════════
N_MC   = 1000
t_eval_mc = np.linspace(0, T_END, 900)   # 2-min resolution for speed

rng_mc = np.random.default_rng(123)

# For each of the 17 identifiable parameters, sample uniformly from search range
PARAM_NAMES = list(SEARCH.keys())

G_matrix = []   # will be (n_successful, len(t_eval_mc))
failed = 0

print(f"\nRunning Monte Carlo simulation ({N_MC} iterations) …")
for n in tqdm(range(N_MC), desc="MC"):
    p = {**NOM}   # start from nominal (fixed params kept)
    for name in PARAM_NAMES:
        lb, ub = SEARCH[name]
        p[name] = rng_mc.uniform(lb, ub)

    ok, t_mc, G_mc = solve_ode(p, t_eval=t_eval_mc)
    if ok and not np.any(np.isnan(G_mc)) and G_mc.max() < 1000.0:
        G_matrix.append(G_mc)
    else:
        failed += 1

G_matrix = np.array(G_matrix)   # shape (n_ok, len(t_eval_mc))
n_ok = len(G_matrix)
print(f"  Successful: {n_ok}/{N_MC}  |  Failed/diverged: {failed}")

# ── Nominal reference ─────────────────────────────────────────────────────────
ok_nom, _, G_nom_mc = solve_ode(NOM, t_eval=t_eval_mc)

# ── Compute statistics ────────────────────────────────────────────────────────
G_mean = G_matrix.mean(axis=0)
G_std  = G_matrix.std(axis=0)
G_p5   = np.percentile(G_matrix, 5,  axis=0)
G_p95  = np.percentile(G_matrix, 95, axis=0)
G_p25  = np.percentile(G_matrix, 25, axis=0)
G_p75  = np.percentile(G_matrix, 75, axis=0)

# ── Figure: Monte Carlo solution space ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
fig.patch.set_facecolor('white')

# 5–95th percentile band
ax.fill_between(t_eval_mc, G_p5, G_p95,
                color='#14538F', alpha=0.15, label='5–95th percentile')
# 25–75th percentile band
ax.fill_between(t_eval_mc, G_p25, G_p75,
                color='#14538F', alpha=0.30, label='25–75th percentile (IQR)')
# Mean ± std
ax.fill_between(t_eval_mc, G_mean-G_std, G_mean+G_std,
                color='#14538F', alpha=0.50, label=r'Mean $\pm$ Std')
# Mean
ax.plot(t_eval_mc, G_mean, color='#14538F', lw=2.0, label='MC Mean')
# Nominal
if ok_nom:
    ax.plot(t_eval_mc, G_nom_mc, color='#CC2222', lw=2.0, ls='--',
            zorder=5, label='Nominal parameters')

for tj in MEAL_T:
    ax.axvline(tj, ls=':', color='#228B22', lw=1.2, alpha=0.8)

ax.set_xlabel('t (min)', fontsize=13)
ax.set_ylabel('G (mg/dl)', fontsize=13)
ax.set_title(f'Monte Carlo Solution Space — Glucose $G(t)$\n'
             f'N={n_ok} successful runs  |  '
             r'Parameters uniformly sampled from Table 1 search ranges',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10, ncol=2, loc='upper right')
ax.grid(True, alpha=0.3, ls=':')
ax.set_xlim(0, T_END)
plt.tight_layout()
plt.savefig('fig_monte_carlo.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig_monte_carlo.png")

# ── Figure: Individual trajectories (first 50) overlaid ──────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
for i in range(min(50, n_ok)):
    ax.plot(t_eval_mc, G_matrix[i], color='#14538F', lw=0.6, alpha=0.25)
ax.plot(t_eval_mc, G_mean, color='#14538F', lw=2.5, label='MC Mean', zorder=5)
if ok_nom:
    ax.plot(t_eval_mc, G_nom_mc, color='#CC2222', lw=2.0, ls='--', label='Nominal')
for tj in MEAL_T:
    ax.axvline(tj, ls=':', color='#228B22', lw=1.2, alpha=0.7)
ax.set_xlabel('t (min)', fontsize=13)
ax.set_ylabel('G (mg/dl)', fontsize=13)
ax.set_title('Monte Carlo Individual Trajectories (first 50 shown)', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, ls=':')
ax.set_xlim(0, T_END)
plt.tight_layout()
plt.savefig('fig_monte_carlo_traces.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig_monte_carlo_traces.png")

# ═════════════════════════════════════════════════════════════════════════════
# 5.  SUMMARY STATISTICS
# ═════════════════════════════════════════════════════════════════════════════
summary = f"""
=======================================================================
 Monte Carlo Summary Statistics  (N={n_ok} successful / {N_MC} total)
=======================================================================
 Glucose G(t) statistics across parameter ensemble:

  Metric                   Mean ± Std    P5      P25     P75     P95
  ─────────────────────────────────────────────────────────────────────
  G_max  [mg/dl]     {G_matrix.max(axis=1).mean():8.2f} ± {G_matrix.max(axis=1).std():6.2f}  \
{np.percentile(G_matrix.max(axis=1),5):7.2f} {np.percentile(G_matrix.max(axis=1),25):7.2f} \
{np.percentile(G_matrix.max(axis=1),75):7.2f} {np.percentile(G_matrix.max(axis=1),95):7.2f}

  G_min  [mg/dl]     {G_matrix.min(axis=1).mean():8.2f} ± {G_matrix.min(axis=1).std():6.2f}  \
{np.percentile(G_matrix.min(axis=1),5):7.2f} {np.percentile(G_matrix.min(axis=1),25):7.2f} \
{np.percentile(G_matrix.min(axis=1),75):7.2f} {np.percentile(G_matrix.min(axis=1),95):7.2f}

  G_mean [mg/dl]     {G_matrix.mean(axis=1).mean():8.2f} ± {G_matrix.mean(axis=1).std():6.2f}  \
{np.percentile(G_matrix.mean(axis=1),5):7.2f} {np.percentile(G_matrix.mean(axis=1),25):7.2f} \
{np.percentile(G_matrix.mean(axis=1),75):7.2f} {np.percentile(G_matrix.mean(axis=1),95):7.2f}

 1-D Sensitivity (at nominal values):
  E   :  Gmax range = [{E_Gmax.min():.1f}, {E_Gmax.max():.1f}]  mg/dl  (Δ = {E_Gmax.max()-E_Gmax.min():.1f})
  C1  :  Gmax range = [{C1_Gmax.min():.1f}, {C1_Gmax.max():.1f}]  mg/dl  (Δ = {C1_Gmax.max()-C1_Gmax.min():.1f})
  E   :  Geq  range = [{E_Geq.min():.1f}, {E_Geq.max():.1f}]  mg/dl
  C1  :  Geq  range = [{C1_Geq.min():.1f}, {C1_Geq.max():.1f}]  mg/dl
=======================================================================
"""
print(summary)
with open('monte_carlo_summary.txt', 'w') as f:
    f.write(summary)
print("Saved: monte_carlo_summary.txt")

print("\n✓  Phase 3 complete.")
