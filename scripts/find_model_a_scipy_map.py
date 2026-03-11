"""Find the scipy-formula MAP for Model A via optimization from the NUTS MAP.

The NUTS MAP parameters give lnL=811.093 in the scipy frame. A quick Nelder-Mead
minimization from this point finds the true scipy-formula MAP for Model A.
"""
import sys
sys.path.insert(0, 'src')
import json
import numpy as np
from scipy.optimize import minimize

with open('output/production_results.json') as f:
    r = json.load(f)

from ultra_slow_de.ingest import load_pantheon_plus
from ultra_slow_de.builtin_data import load_all_bao
from ultra_slow_de.params import CosmoParams, ModelAParams
from ultra_slow_de.inference import joint_logposterior
from ultra_slow_de.likelihood import SNLikelihoodCached

sn = load_pantheon_plus()
bao = load_all_bao()
datasets = [(sn, "mb")] + bao

# Pre-cache SN for speed
sn_cache = SNLikelihoodCached(sn)

# NUTS MAP parameters for Model A
a_r = r['production']['a']
bf = a_r['posterior']
h0_0 = bf['h0']['best_fit']
om_0 = bf['omega_m']['best_fit']
w0_0 = bf['w0']['best_fit']
B_0 = bf['B']['best_fit']
C_0 = bf['C']['best_fit']
omega_0 = bf['omega']['best_fit']

print(f"NUTS MAP: h0={h0_0:.4f}, om={om_0:.5f}, w0={w0_0:.4f}, B={B_0:.5f}, C={C_0:.5f}, omega={omega_0:.4f}")

# Bounds for Model A (from sampler_nuts.py _BOUNDS_NP - same as production_run.py)
BOUNDS_A = np.array([
    [60.0, 80.0],   # h0
    [0.1,  0.5],    # omega_m
    [-1.5, -0.5],   # w0
    [-0.3,  0.3],   # B
    [-0.3,  0.3],   # C
    [0.1,   5.0],   # omega
])

def neg_lnl(theta):
    h0, om, w0, B, C, omg = theta
    # Bounds check
    for i, (lo, hi) in enumerate(BOUNDS_A):
        if theta[i] < lo or theta[i] > hi:
            return 1e10
    cosmo = CosmoParams(h0=h0, omega_m=om)
    ma = ModelAParams(w0=w0, B=B, C=C, omega=omg)
    try:
        ll = joint_logposterior(datasets, cosmo, model_a=ma, rd=147.09,
                                include_planck=True, sn_cache=sn_cache)
        return -ll
    except Exception:
        return 1e10

x0 = np.array([h0_0, om_0, w0_0, B_0, C_0, omega_0])

# First eval at NUTS MAP
ll_start = -neg_lnl(x0)
print(f"lnL at NUTS MAP (scipy formula): {ll_start:.4f}")

# Optimize from NUTS MAP using L-BFGS-B with explicit bounds
print("Optimizing from NUTS MAP (L-BFGS-B with bounds)...")
scipy_bounds = [(lo, hi) for lo, hi in BOUNDS_A]
result = minimize(neg_lnl, x0, method='L-BFGS-B',
                  bounds=scipy_bounds,
                  options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-6})

ll_opt = -result.fun
theta_opt = result.x
print(f"Optimization success: {result.success}, iterations: {result.nit}")
print(f"lnL at scipy MAP: {ll_opt:.4f}  (improvement: {ll_opt - ll_start:+.4f})")
print(f"scipy MAP: h0={theta_opt[0]:.4f}, om={theta_opt[1]:.5f}, w0={theta_opt[2]:.4f}, B={theta_opt[3]:.5f}, C={theta_opt[4]:.5f}, omega={theta_opt[5]:.4f}")

# If omega hit boundary, also try with wider omega range
if abs(theta_opt[5] - BOUNDS_A[5,1]) < 0.01:
    print("\nomega hit upper boundary. Checking if likelihood improves beyond...")
    # Extend boundary temporarily
    BOUNDS_WIDE = BOUNDS_A.copy()
    BOUNDS_WIDE[5,1] = 10.0
    scipy_bounds_wide = [(lo, hi) for lo, hi in BOUNDS_WIDE]
    result2 = minimize(neg_lnl, theta_opt, method='L-BFGS-B',
                       bounds=scipy_bounds_wide,
                       options={'maxiter': 2000, 'ftol': 1e-10})
    ll_wide = -result2.fun
    print(f"With omega up to 10: lnL={ll_wide:.4f} at omega={result2.x[5]:.4f}")
    if ll_wide > ll_opt:
        ll_opt = ll_wide
        theta_opt = result2.x
        print(f"Using wider omega bound: omega={theta_opt[5]:.4f}")

# Compare with LCDM
lcdm_r = r['production']['lcdm']
cosmo_lcdm = CosmoParams(h0=lcdm_r['posterior']['h0']['best_fit'],
                          omega_m=lcdm_r['posterior']['omega_m']['best_fit'])
ll_lcdm = joint_logposterior(datasets, cosmo_lcdm, include_planck=True)
print(f"\nLCDM scipy lnL: {ll_lcdm:.4f}")
print(f"Delta lnL (Model A - LCDM): {ll_opt - ll_lcdm:+.4f}")

N_DATA = 1644
k_a, k_lcdm = 6, 2
delta_aic = -2*(ll_opt - ll_lcdm) + 2*(k_a - k_lcdm)
delta_bic = -2*(ll_opt - ll_lcdm) + (k_a - k_lcdm)*np.log(N_DATA)
print(f"Corrected ΔAIC(A vs LCDM): {delta_aic:+.2f}")
print(f"Corrected ΔBIC(A vs LCDM): {delta_bic:+.2f}")

# Save optimized MAP
output = {
    'model_a_scipy_map': {
        'h0': float(theta_opt[0]),
        'omega_m': float(theta_opt[1]),
        'w0': float(theta_opt[2]),
        'B': float(theta_opt[3]),
        'C': float(theta_opt[4]),
        'omega': float(theta_opt[5]),
        'best_loglike_scipy': float(ll_opt),
        'lcdm_loglike_scipy': float(ll_lcdm),
        'delta_lnL': float(ll_opt - ll_lcdm),
        'delta_aic': float(delta_aic),
        'delta_bic': float(delta_bic),
    }
}
with open('output/model_a_scipy_map.json', 'w') as f:
    json.dump(output, f, indent=2)
print("\nSaved to output/model_a_scipy_map.json")
