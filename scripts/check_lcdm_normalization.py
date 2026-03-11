"""Check LCDM loglike normalization difference between MCMC path and NUTS path."""
import sys
sys.path.insert(0, 'src')
import json
import numpy as np

with open('output/production_results.json') as f:
    results = json.load(f)

lcdm_r = results['production']['lcdm']
h0_bf = lcdm_r['posterior']['h0']['best_fit']
om_bf = lcdm_r['posterior']['omega_m']['best_fit']
print(f"LCDM MCMC best-fit: h0={h0_bf:.4f}, omega_m={om_bf:.6f}")
print(f"MCMC saved best_loglike: {lcdm_r['best_loglike_found']:.4f}")

from ultra_slow_de.ingest import load_pantheon_plus
from ultra_slow_de.builtin_data import load_all_bao
from ultra_slow_de.inference import joint_logposterior
from ultra_slow_de.params import CosmoParams

sn = load_pantheon_plus()
bao = load_all_bao()
datasets = [(sn, "mb")] + bao

# MCMC path: omega_r=0 for SN/BAO likelihoods
cosmo_no_r = CosmoParams(h0=h0_bf, omega_m=om_bf)  # omega_r=0
ll_no_r = joint_logposterior(datasets, cosmo_no_r, include_planck=True)
print(f"No-radiation loglike at MCMC best-fit: {ll_no_r:.4f}")

# NUTS path: omega_r included in low-z computations  
_OMEGA_R_H2 = 4.18e-5
omega_r = _OMEGA_R_H2 / (h0_bf / 100.0) ** 2
cosmo_rad = CosmoParams(h0=h0_bf, omega_m=om_bf, omega_r=omega_r)
ll_rad = joint_logposterior(datasets, cosmo_rad, include_planck=True)
print(f"With-radiation loglike at MCMC best-fit: {ll_rad:.4f}")
print(f"Difference (no-rad - with-rad): {ll_no_r - ll_rad:.4f}")

# Also check contributions individually
print()
print("--- Per-dataset contributions ---")
from ultra_slow_de.inference import loglike_for_dataset, loglike_planck_compressed, _h_on_grid

# Get shared H(z) grid for each
from ultra_slow_de.inference import _h_on_grid
z1, h1 = _h_on_grid(cosmo_no_r, None, None)
z2, h2 = _h_on_grid(cosmo_rad, None, None)

for ds, obs in datasets:
    ll1 = loglike_for_dataset(ds, obs, cosmo_no_r, _precomputed_grid=(z1, h1))
    ll2 = loglike_for_dataset(ds, obs, cosmo_rad, _precomputed_grid=(z2, h2))
    print(f"  {obs}: no-rad={ll1:.4f}, rad={ll2:.4f}, diff={ll1-ll2:.4f}")

pl1 = loglike_planck_compressed(cosmo_no_r)
pl2 = loglike_planck_compressed(cosmo_rad)
print(f"  planck: no-rad={pl1:.4f}, rad={pl2:.4f}, diff={pl1-pl2:.4f}")
print()
print(f"Total no-rad: {ll_no_r:.4f}")
print(f"Total with-rad: {ll_rad:.4f}")
