"""Evaluate all model best-fits using a single consistent (scipy/MCMC) loglike formula.

This resolves the NUTS/MCMC numerical implementation discrepancy by evaluating
all models at their respective MAP parameters using the same loglike code path.
"""
import sys
sys.path.insert(0, 'src')
import json
import numpy as np

with open('output/production_results.json') as f:
    r = json.load(f)

from ultra_slow_de.ingest import load_pantheon_plus
from ultra_slow_de.builtin_data import load_all_bao
from ultra_slow_de.params import CosmoParams, ModelAParams
from ultra_slow_de.model_b import ModelBParams
from ultra_slow_de.inference import joint_logposterior

sn = load_pantheon_plus()
bao = load_all_bao()
datasets = [(sn, "mb")] + bao

# ---- LCDM ----
lcdm_r = r['production']['lcdm']
h0_lcdm = lcdm_r['posterior']['h0']['best_fit']
om_lcdm = lcdm_r['posterior']['omega_m']['best_fit']
cosmo_lcdm = CosmoParams(h0=h0_lcdm, omega_m=om_lcdm)
ll_lcdm = joint_logposterior(datasets, cosmo_lcdm, include_planck=True)
print(f"LCDM MAP lnL (scipy-formula): {ll_lcdm:.4f}   [stored: {lcdm_r['best_loglike_found']:.4f}]")

# ---- Model A ----
a_r = r['production']['a']
bf = a_r['posterior']
h0_a = bf['h0']['best_fit']
om_a = bf['omega_m']['best_fit']
w0_a = bf['w0']['best_fit']
B_a = bf['B']['best_fit']
C_a = bf['C']['best_fit']
omega_a = bf['omega']['best_fit']
cosmo_a = CosmoParams(h0=h0_a, omega_m=om_a)
ma = ModelAParams(w0=w0_a, B=B_a, C=C_a, omega=omega_a)
ll_a = joint_logposterior(datasets, cosmo_a, model_a=ma, rd=147.09, include_planck=True)
print(f"Model A MAP lnL (scipy-formula): {ll_a:.4f}   [stored NUTS: {a_r['best_loglike_found']:.4f}]")

# ---- Model B ----
b_r = r['production']['b']
bf = b_r['posterior']
h0_b = bf['h0']['best_fit']
om_b = bf['omega_m']['best_fit']
mu_b = bf['mu']['best_fit']
cosmo_b = CosmoParams(h0=h0_b, omega_m=om_b)
mb = ModelBParams(mu=mu_b)
ll_b = joint_logposterior(datasets, cosmo_b, model_b=mb, rd=147.09, include_planck=True)
print(f"Model B MAP lnL (scipy-formula): {ll_b:.4f}   [stored: {b_r['best_loglike_found']:.4f}]")

# ---- Consistent comparison ----
print()
print("=== Consistent scipy-formula comparison ===")
print(f"LCDM:    lnL = {ll_lcdm:.4f}")
print(f"Model A: lnL = {ll_a:.4f}  (d_lnL = {ll_a - ll_lcdm:+.4f})")
print(f"Model B: lnL = {ll_b:.4f}  (d_lnL = {ll_b - ll_lcdm:+.4f})")

# AIC / BIC
k_lcdm, k_a, k_b = 2, 6, 3
n_data = len(sn.y_obs) + sum(len(ds.y_obs) for ds, obs in bao if obs != 'mb') + 2  # +2 Planck
print(f"\nN_data = {n_data}")

def aic(ll, k): return -2*ll + 2*k
def bic(ll, k, n): return -2*ll + k*np.log(n)

print()
print("=== Information Criteria (scipy-frame) ===")
for label, ll, k in [("LCDM", ll_lcdm, k_lcdm), ("Model A", ll_a, k_a), ("Model B", ll_b, k_b)]:
    a_val = aic(ll, k)
    b_val = bic(ll, k, n_data)
    da = a_val - aic(ll_lcdm, k_lcdm)
    db = b_val - bic(ll_lcdm, k_lcdm, n_data)
    print(f"  {label}: AIC={a_val:.2f},  BIC={b_val:.2f},  dAIC={da:+.2f},  dBIC={db:+.2f}")
