"""
================================================================================
 PHASE 1 — SBINN Parameter Estimation (DeepXDE / TensorFlow)
 Paper   : Daneker, Zhang, Karniadakis, Lu — arXiv:2202.01723
 Purpose : Infer 15 ODE parameters of the Ultradian Glucose-Insulin model
           from synthetic glucose-only observations G(t).

 HOW TO RUN
 ----------
   pip install deepxde tensorflow
   python phase1_sbinn_estimation.py

 OUTPUT FILES
 ------------
   inferred_params.npy   — dict with inferred parameter values
   loss_history.png      — training loss curve
   glucose_fit.png       — data fit on G(t)
================================================================================
"""

# ── Standard library ─────────────────────────────────────────────────────────
import os, time
import numpy as np
import matplotlib.pyplot as plt

# ── DeepXDE + TensorFlow backend ─────────────────────────────────────────────
os.environ["DDE_BACKEND"] = "tensorflow"          # set before importing dde
import deepxde as dde
from deepxde.backend import tf

# ═════════════════════════════════════════════════════════════════════════════
# 0.  FIXED PARAMETERS  (Vp, Vi, Vg fixed for structural identifiability)
# ═════════════════════════════════════════════════════════════════════════════
Vp = 3.0    # [L]   plasma insulin distribution volume  — FIXED
Vi = 11.0   # [L]   remote insulin compartment volume   — FIXED
Vg = 10.0   # [L]   glucose space volume                — FIXED

# Nominal values (used for synthetic data generation AND as ground-truth reference)
NOM = dict(
    E    = 0.2,    Vp   = 3.0,    Vi   = 11.0,   Vg   = 10.0,
    tp   = 6.0,    ti   = 100.0,  td   = 12.0,   k    = 0.0083,
    Rm   = 209.0,  a1   = 6.6,    C1   = 300.0,  C2   = 144.0,
    C3   = 100.0,  C4   = 80.0,   C5   = 26.0,   Ub   = 72.0,
    U0   = 4.0,    Um   = 90.0,   Rg   = 180.0,  alpha= 7.5,
    beta = 1.772,
)

# ═════════════════════════════════════════════════════════════════════════════
# 1.  SYNTHETIC DATA GENERATION  (forward ODE with nominal parameters)
# ═════════════════════════════════════════════════════════════════════════════
from scipy.integrate import solve_ivp

def make_kappa(C4, Vi, E, ti):
    return (1.0 / C4) * (1.0 / Vi + 1.0 / (E * ti))

def nutritional_driver(t, meal_t, meal_m, k):
    """I_G(t) [mg/min]  — causal exponential pulses from each meal."""
    val = 0.0
    for tj, mj in zip(meal_t, meal_m):
        dt = t - tj
        if dt >= 0.0:
            val += mj * 1000.0 * k * np.exp(-k * dt)   # g → mg
    return val

def ultradian_rhs(t, x, p, meal_t, meal_m):
    """
    ODE right-hand side.  State vector in TOTAL MASS units:
      x = [Ip[mU], Ii[mU], G[mg], h1[mU], h2[mU], h3[mU]]
    """
    Ip, Ii, G, h1, h2, h3 = x
    Ii_s = max(Ii, 1e-12)

    kap = make_kappa(p['C4'], p['Vi'], p['E'], p['ti'])
    Ii_c = Ii_s / p['Vi']          # concentration for kappa term  [mU/L]

    f1 = p['Rm'] / (1.0 + np.exp(-G / (p['Vg']*p['C1']) + p['a1']))
    f2 = p['Ub'] * (1.0 - np.exp(-G / (p['C2']*p['Vg'])))
    f3 = (1.0/(p['C3']*p['Vg'])) * (p['U0'] + p['Um']/(1.0+(kap*Ii_c)**(-p['beta'])))
    f4 = p['Rg'] / (1.0 + np.exp(p['alpha']*(h3/(p['C5']*p['Vp']) - 1.0)))
    IG = nutritional_driver(t, meal_t, meal_m, p['k'])

    dIp = f1 - p['E']*(Ip/p['Vp'] - Ii/p['Vi']) - Ip/p['tp']
    dIi = p['E']*(Ip/p['Vp'] - Ii/p['Vi']) - Ii/p['ti']
    dG  = f4 + IG - f2 - f3*G
    dh1 = (Ip - h1) / p['td']
    dh2 = (h1 - h2) / p['td']
    dh3 = (h2 - h3) / p['td']
    return [dIp, dIi, dG, dh1, dh2, dh3]

# Initial conditions in mass units (see unit-conversion notes in ode_system_simulation.m)
x0_mass = [
    12.0 * Vp,      # Ip = 36 mU
    4.0  * Vi,      # Ii = 44 mU
    110.0* Vg*10,   # G  = 11000 mg
    0.0, 0.0, 0.0,
]

meal_t = [300, 650, 1100]   # [min]
meal_m = [60,  40,  50  ]   # [g]

T_END   = 1800.0
N_OBS   = 360       # number of random glucose observations

print("Generating synthetic data …")
sol = solve_ivp(
    fun    = lambda t, x: ultradian_rhs(t, x, NOM, meal_t, meal_m),
    t_span = (0.0, T_END),
    y0     = x0_mass,
    method = "Radau",
    max_step = 1.0,
    rtol   = 1e-8,
    atol   = 1e-10,
    dense_output = True,
)
assert sol.success, "ODE solver failed during data generation!"

# Dense evaluation on a fine grid for reference
t_fine  = np.linspace(0, T_END, 5000)
sol_fine = sol.sol(t_fine)             # shape (6, 5000)
G_fine   = sol_fine[2] / (Vg * 10)    # convert mg → mg/dl

# Random 360-point observation sample (glucose only, G in mg/dl)
rng = np.random.default_rng(42)
obs_idx  = np.sort(rng.choice(len(t_fine), size=N_OBS, replace=False))
t_obs    = t_fine[obs_idx].reshape(-1, 1)       # [N_OBS, 1]  seconds
G_obs    = G_fine[obs_idx].reshape(-1, 1)       # [N_OBS, 1]  mg/dl
G_obs   += rng.normal(0, 0.5, G_obs.shape)     # add ~0.5 mg/dl noise

# Collect initial & final state for L_aux  (in mg/dl and uU/ml as observable scales)
x_T0 = sol.sol(0.0)
x_T1 = sol.sol(T_END)

print(f"  G range: {G_fine.min():.1f} – {G_fine.max():.1f} mg/dl")
print(f"  Observations: {N_OBS}  ·  Noise σ ≈ 0.5 mg/dl")

# ═════════════════════════════════════════════════════════════════════════════
# 2.  DEEPXDE SETUP — trainable parameters with bounded search ranges
# ═════════════════════════════════════════════════════════════════════════════
# Search ranges from Table 1 of the paper  (first 7 from Sturis 1991; rest ±80%)
# Format: (lb, ub)
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

def bounded(raw, lb, ub):
    """Map unconstrained raw ∈ ℝ → (lb, ub) via tanh."""
    return ((tf.tanh(raw) + 1.0) / 2.0) * (ub - lb) + lb

# Trainable variables — all initialised to 0.0 (raw space)
_E     = dde.Variable(0.0)
_tp    = dde.Variable(0.0)
_ti    = dde.Variable(0.0)
_td    = dde.Variable(0.0)
_k     = dde.Variable(0.0)
_Rm    = dde.Variable(0.0)
_a1    = dde.Variable(0.0)
_C1    = dde.Variable(0.0)
_C2    = dde.Variable(0.0)
_C4    = dde.Variable(0.0)
_C5    = dde.Variable(0.0)
_Ub    = dde.Variable(0.0)
_U0    = dde.Variable(0.0)
_Um    = dde.Variable(0.0)
_Rg    = dde.Variable(0.0)
_alpha = dde.Variable(0.0)
_beta  = dde.Variable(0.0)

var_list = [_E, _tp, _ti, _td, _k, _Rm, _a1, _C1, _C2,
            _C4, _C5, _Ub, _U0, _Um, _Rg, _alpha, _beta]

def get_params():
    """Return bounded physical parameters from raw trainable variables."""
    E     = bounded(_E,     *SEARCH['E'])
    tp    = bounded(_tp,    *SEARCH['tp'])
    ti    = bounded(_ti,    *SEARCH['ti'])
    td    = bounded(_td,    *SEARCH['td'])
    k     = bounded(_k,     *SEARCH['k'])
    Rm    = bounded(_Rm,    *SEARCH['Rm'])
    a1    = bounded(_a1,    *SEARCH['a1'])
    C1    = bounded(_C1,    *SEARCH['C1'])
    C2    = bounded(_C2,    *SEARCH['C2'])
    C4    = bounded(_C4,    *SEARCH['C4'])
    C5    = bounded(_C5,    *SEARCH['C5'])
    Ub    = bounded(_Ub,    *SEARCH['Ub'])
    U0    = bounded(_U0,    *SEARCH['U0'])
    Um    = bounded(_Um,    *SEARCH['Um'])
    Rg    = bounded(_Rg,    *SEARCH['Rg'])
    alpha = bounded(_alpha, *SEARCH['alpha'])
    beta  = bounded(_beta,  *SEARCH['beta'])
    return E, tp, ti, td, k, Rm, a1, C1, C2, C4, C5, Ub, U0, Um, Rg, alpha, beta

# ═════════════════════════════════════════════════════════════════════════════
# 3.  ODE RESIDUAL FUNCTION  (L_ode)
#     Network input : t  (scaled to [0,1])
#     Network output: [Ip, Ii, G, h1, h2, h3]  in normalised mass units
#                     (output-scaled back in apply_output_transform)
# ═════════════════════════════════════════════════════════════════════════════
# Scaling factors — match order-of-magnitude of each state at t=0
OUT_SCALE = np.array([x0_mass[0], x0_mass[1], x0_mass[2],
                       1.0, 1.0, 1.0], dtype=np.float32)
# h1,h2,h3 start at 0 so we use 1.0; the NN will learn the scale.

C3_FIXED  = NOM['C3']  # fixed by identifiability constraint

def ode_residuals(t, y):
    """
    Returns list of 6 ODE residuals evaluated at collocation points.
    y = [Ip_hat, Ii_hat, G_hat, h1_hat, h2_hat, h3_hat]  (raw network output)
    After output transform: y_phys = OUT_SCALE * y
    """
    E, tp, ti, td, k, Rm, a1, C1, C2, C4, C5, Ub, U0, Um, Rg, alpha, beta = get_params()

    Ip = y[:, 0:1]   # mU
    Ii = y[:, 1:2]   # mU
    G  = y[:, 2:3]   # mg
    h1 = y[:, 3:4]   # mU
    h2 = y[:, 4:5]   # mU
    h3 = y[:, 5:6]   # mU

    Ii_c = tf.maximum(Ii, 1e-12) / Vi   # concentration [mU/L]

    kap  = (1.0/C4) * (1.0/Vi + 1.0/(E*ti))

    f1 = Rm / (1.0 + tf.exp(-G/(Vg*C1) + a1))
    f2 = Ub * (1.0 - tf.exp(-G/(C2*Vg)))
    f3 = (1.0/(C3_FIXED*Vg)) * (U0 + Um/(1.0+(kap*Ii_c)**(-beta)))
    f4 = Rg / (1.0 + tf.exp(alpha*(h3/(C5*Vp) - 1.0)))

    # Nutritional driver  I_G(t) — using soft Heaviside for differentiability
    t_scaled = t * T_END   # t is in [0,1] after input scaling; convert back to minutes
    def meal_pulse(tj, mj):
        dt = t_scaled - tj
        heaviside = tf.sigmoid(dt * 0.5)   # soft step, ~0 for dt<0, ~1 for dt>0
        return mj * 1000.0 * k * tf.exp(-k * tf.maximum(dt, 0.0)) * heaviside

    IG = meal_pulse(300, 60) + meal_pulse(650, 40) + meal_pulse(1100, 50)

    # Jacobians (auto-diff)
    dIp = dde.grad.jacobian(y, t, i=0, j=0)
    dIi = dde.grad.jacobian(y, t, i=1, j=0)
    dG  = dde.grad.jacobian(y, t, i=2, j=0)
    dh1 = dde.grad.jacobian(y, t, i=3, j=0)
    dh2 = dde.grad.jacobian(y, t, i=4, j=0)
    dh3 = dde.grad.jacobian(y, t, i=5, j=0)

    # dIp/dt  (multiply by T_END because t is normalised to [0,1])
    r_Ip = dIp/T_END - (f1 - E*(Ip/Vp - Ii/Vi) - Ip/tp)
    r_Ii = dIi/T_END - (E*(Ip/Vp - Ii/Vi) - Ii/ti)
    r_G  = dG /T_END - (f4 + IG - f2 - f3*G)
    r_h1 = dh1/T_END - (Ip - h1)/td
    r_h2 = dh2/T_END - (h1 - h2)/td
    r_h3 = dh3/T_END - (h2 - h3)/td

    return [r_Ip, r_Ii, r_G, r_h1, r_h2, r_h3]

# ═════════════════════════════════════════════════════════════════════════════
# 4.  DEEPXDE PROBLEM DEFINITION
# ═════════════════════════════════════════════════════════════════════════════
# Time domain normalised to [0, 1]
geom = dde.geometry.TimeDomain(0.0, 1.0)

# ── ODE constraint (L_ode) ────────────────────────────────────────────────────
ode_data = dde.data.PDE(
    geometry         = geom,
    pde              = ode_residuals,
    bcs              = [],
    num_domain       = 2000,    # collocation points
    num_boundary     = 2,
    num_test         = 500,
)

# ── Data observation constraint (L_data) — glucose G only ─────────────────────
# G is state index 2; network output in physical units after scaling
t_obs_norm  = t_obs / T_END    # normalise to [0,1]
G_obs_mass  = G_obs * (Vg*10)  # convert mg/dl → mg (network works in mass)

observe_G = dde.PointSetBC(
    points    = t_obs_norm,
    values    = G_obs_mass,
    component = 2,       # index of G in output vector
)

# ── Auxiliary constraint (L_aux) — initial and final state ────────────────────
T0_norm = np.array([[0.0]])
T1_norm = np.array([[1.0]])

# States at t=0  (exact, from x0_mass)
IC_Ip = dde.PointSetBC(T0_norm, np.array([[x_T0[0]]]), component=0)
IC_Ii = dde.PointSetBC(T0_norm, np.array([[x_T0[1]]]), component=1)
IC_G  = dde.PointSetBC(T0_norm, np.array([[x_T0[2]]]), component=2)
IC_h1 = dde.PointSetBC(T0_norm, np.array([[x_T0[3]]]), component=3)
IC_h2 = dde.PointSetBC(T0_norm, np.array([[x_T0[4]]]), component=4)
IC_h3 = dde.PointSetBC(T0_norm, np.array([[x_T0[5]]]), component=5)

# States at t=T1 (from the forward ODE solution)
FC_Ip = dde.PointSetBC(T1_norm, np.array([[x_T1[0]]]), component=0)
FC_Ii = dde.PointSetBC(T1_norm, np.array([[x_T1[1]]]), component=1)
FC_G  = dde.PointSetBC(T1_norm, np.array([[x_T1[2]]]), component=2)

# Combine all data sources
data = dde.data.PDE(
    geometry = geom,
    pde      = ode_residuals,
    bcs      = [observe_G,
                IC_Ip, IC_Ii, IC_G, IC_h1, IC_h2, IC_h3,
                FC_Ip, FC_Ii, FC_G],
    num_domain   = 2000,
    num_boundary = 2,
    num_test     = 500,
    anchors      = t_obs_norm,    # always include observation times
)

# ═════════════════════════════════════════════════════════════════════════════
# 5.  NETWORK ARCHITECTURE
# ═════════════════════════════════════════════════════════════════════════════
layer_size   = [1] + [64]*4 + [6]   # 1 input → 4 hidden (64 neurons) → 6 outputs
activation   = "tanh"
initializer  = "Glorot uniform"

net = dde.nn.FNN(layer_size, activation, initializer)

# Input scaling: t̃ = t (already in [0,1])
# Feature layer: t̃ concatenated with sin(k*t̃) for k=1..5  (captures periodicity)
def feature_transform(t):
    t_sc = t               # already [0,1]
    feats = [t_sc,
             tf.sin(  np.pi * t_sc),
             tf.sin(2*np.pi * t_sc),
             tf.sin(3*np.pi * t_sc),
             tf.sin(4*np.pi * t_sc),
             tf.sin(5*np.pi * t_sc)]
    return tf.concat(feats, axis=1)

net.apply_feature_transform(feature_transform)

# Output scaling: rescale each component to physical mass magnitude
out_scale_tf = tf.constant(
    [x0_mass[0], x0_mass[1], x0_mass[2], x0_mass[0], x0_mass[0], x0_mass[0]],
    dtype=tf.float32,
)

def output_transform(t, y):
    return y * out_scale_tf

net.apply_output_transform(output_transform)

# ═════════════════════════════════════════════════════════════════════════════
# 6.  LOSS WEIGHTS
#     Structure: [ODE×6, data_G, IC×6, FC×3]
# ═════════════════════════════════════════════════════════════════════════════
# Warm-up: only data + IC losses active (no ODE yet)
w_warmup = [0,0,0,0,0,0,   # ODE terms
            1e-2,           # L_data (G obs)
            1,1,1,1,1,1,    # IC all 6 states
            1,1,1]          # FC 3 states

# Full loss
w_full   = [1,1,1e-2,1,1,1, # ODE terms (dG weight reduced for balance)
            1e-2,            # L_data
            1,1,1,1,1,1,     # IC
            1,1,1]           # FC

# ═════════════════════════════════════════════════════════════════════════════
# 7.  TRAINING
# ═════════════════════════════════════════════════════════════════════════════
model = dde.Model(data, net)

print("\n── Phase A: Warm-up training (10 000 epochs, data+IC only) ──")
model.compile("adam", lr=1e-3, loss_weights=w_warmup)
t0 = time.time()
lh_warmup, _ = model.train(epochs=10_000, display_every=2000)
print(f"   Warm-up done in {time.time()-t0:.1f}s")

print("\n── Phase B: Full training (600 000 epochs) ──")
model.compile("adam", lr=1e-3,
              loss_weights=w_full,
              external_trainable_variables=var_list)

var_callback = dde.callbacks.VariableValue(
    var_list,
    period   = 10_000,
    filename = "variable_history.csv",
)

t1 = time.time()
lh_full, ts_full = model.train(
    epochs        = 600_000,
    display_every = 20_000,
    callbacks     = [var_callback],
)
print(f"   Full training done in {(time.time()-t1)/60:.1f} min")

# ═════════════════════════════════════════════════════════════════════════════
# 8.  EXTRACT INFERRED PARAMETERS
# ═════════════════════════════════════════════════════════════════════════════
def _val(raw_var, lb, ub):
    """Evaluate bounded parameter from raw TF variable."""
    raw = raw_var.numpy()
    return float(((np.tanh(raw) + 1.0) / 2.0) * (ub - lb) + lb)

inferred = {
    'E'    : _val(_E,     *SEARCH['E']),
    'tp'   : _val(_tp,    *SEARCH['tp']),
    'ti'   : _val(_ti,    *SEARCH['ti']),
    'td'   : _val(_td,    *SEARCH['td']),
    'k'    : _val(_k,     *SEARCH['k']),
    'Rm'   : _val(_Rm,    *SEARCH['Rm']),
    'a1'   : _val(_a1,    *SEARCH['a1']),
    'C1'   : _val(_C1,    *SEARCH['C1']),
    'C2'   : _val(_C2,    *SEARCH['C2']),
    'C4'   : _val(_C4,    *SEARCH['C4']),
    'C5'   : _val(_C5,    *SEARCH['C5']),
    'Ub'   : _val(_Ub,    *SEARCH['Ub']),
    'U0'   : _val(_U0,    *SEARCH['U0']),
    'Um'   : _val(_Um,    *SEARCH['Um']),
    'Rg'   : _val(_Rg,    *SEARCH['Rg']),
    'alpha': _val(_alpha, *SEARCH['alpha']),
    'beta' : _val(_beta,  *SEARCH['beta']),
    # Fixed parameters
    'Vp'   : Vp, 'Vi': Vi, 'Vg': Vg,
    'C3'   : NOM['C3'],
}

print("\n" + "="*60)
print("  INFERRED vs NOMINAL PARAMETERS")
print("="*60)
print(f"{'Param':>8}  {'Nominal':>10}  {'Inferred':>10}  {'Err %':>8}")
print("-"*60)
for name in ['E','tp','ti','td','k','Rm','a1','C1','C2','C4','C5','Ub','U0','Um','Rg','alpha','beta']:
    nom = NOM[name]
    inf = inferred[name]
    err = abs(inf - nom) / abs(nom) * 100
    print(f"{name:>8}  {nom:>10.4f}  {inf:>10.4f}  {err:>7.2f}%")
print("="*60)

np.save("inferred_params.npy", inferred)
print("\nSaved: inferred_params.npy")

# ═════════════════════════════════════════════════════════════════════════════
# 9.  PLOTS
# ═════════════════════════════════════════════════════════════════════════════
# ── 9a. Loss curve ────────────────────────────────────────────────────────────
all_loss = np.array(lh_warmup.loss_train + lh_full.loss_train)
fig, ax = plt.subplots(figsize=(9, 4))
ax.semilogy(np.sum(all_loss, axis=1), color='#14538F', lw=1.5, label='Total loss')
ax.axvline(10_000, ls='--', color='#CC2222', lw=1.2, label='End of warm-up')
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Loss (log scale)', fontsize=12)
ax.set_title('SBINN Training Loss', fontsize=14, fontweight='bold')
ax.legend(); ax.grid(True, alpha=0.3, ls=':')
plt.tight_layout()
plt.savefig('loss_history.png', dpi=150)
plt.close()
print("Saved: loss_history.png")

# ── 9b. Glucose fit ──────────────────────────────────────────────────────────
t_pred_norm = (t_fine / T_END).reshape(-1, 1)
y_pred      = model.predict(t_pred_norm)          # shape (5000, 6)
G_pred_dl   = y_pred[:, 2] / (Vg * 10)            # mg → mg/dl

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(t_fine, G_fine,   color='#14538F', lw=2,    label='Ground truth G(t)')
ax.plot(t_fine, G_pred_dl,color='#CC2222', lw=1.5, ls='--', label='SBINN prediction')
ax.scatter(t_obs.flatten(), G_obs.flatten(),
           s=8, color='#E07800', alpha=0.6, zorder=5, label=f'Observations (N={N_OBS})')
for tj in meal_t:
    ax.axvline(tj, ls=':', color='green', lw=1.2, alpha=0.8)
ax.set_xlabel('t (min)', fontsize=12)
ax.set_ylabel('G (mg/dl)', fontsize=12)
ax.set_title('SBINN Glucose Fit vs Ground Truth', fontsize=14, fontweight='bold')
ax.legend(fontsize=10); ax.grid(True, alpha=0.3, ls=':')
plt.tight_layout()
plt.savefig('glucose_fit.png', dpi=150)
plt.close()
print("Saved: glucose_fit.png")

print("\n✓  Phase 1 complete.")
