"""Isolate which term causes the 2.08 discrepancy between MCMC and JAX-NUTS lnL."""
import sys
sys.path.insert(0, 'src')
import json
import numpy as np
import jax.numpy as jnp
import jax

with open('output/production_results.json') as f:
    results = json.load(f)

lcdm_r = results['production']['lcdm']
h0 = lcdm_r['posterior']['h0']['best_fit']
om = lcdm_r['posterior']['omega_m']['best_fit']

from ultra_slow_de.ingest import load_pantheon_plus
from ultra_slow_de.builtin_data import load_all_bao
from ultra_slow_de.params import CosmoParams
from ultra_slow_de.inference import _h_on_grid, loglike_for_dataset, loglike_planck_compressed
from ultra_slow_de.sampler_nuts import _prepare, _hz_grid_lcdm_jax, _comoving_dist_jax, C_KM_S

sn = load_pantheon_plus()
bao = load_all_bao()
datasets = [(sn, "mb")] + bao
rd = 147.09

# MCMC: per-dataset contributions
cosmo = CosmoParams(h0=h0, omega_m=om)
z_grid, h_grid = _h_on_grid(cosmo, None, None)

print("=== MCMC formula ===")
total_mcmc = 0.0
for ds, obs in datasets:
    ll = loglike_for_dataset(ds, obs, cosmo, _precomputed_grid=(z_grid, h_grid))
    print(f"  {obs}: {ll:.5f}")
    total_mcmc += ll
pl = loglike_planck_compressed(cosmo)
print(f"  planck: {pl:.5f}")
total_mcmc += pl
print(f"  TOTAL: {total_mcmc:.5f}")

# JAX-NUTS: per-term contributions
print("\n=== JAX-NUTS formula ===")
prep = _prepare(datasets, rd=rd, include_planck=True)

# Reimplement _ll_theta step by step
theta = jnp.array([h0, om])
z_low = prep.z_low_grid
z_cmb = prep.z_cmb_grid

h_low = _hz_grid_lcdm_jax(theta[0], theta[1], z_low)
dm_low = _comoving_dist_jax(z_low, h_low)

# SN Ia
z_sn = prep.z_sn; y_sn = prep.y_sn
c_inv_sn = prep.c_inv_sn; c_inv_ones_sn = prep.c_inv_ones_sn
e_sn = prep.e_sn; sn_const = prep.sn_const

dm_sn = jnp.interp(z_sn, z_low, dm_low)
dl_sn = dm_sn * (1.0 + z_sn)
mu_model = 5.0 * jnp.log10(jnp.clip(dl_sn * 1e6 / 10.0, 1e-30, None))
delta = y_sn - mu_model
c_inv_d = c_inv_sn @ delta
a_val = delta @ c_inv_d
b_val = c_inv_ones_sn @ delta
chi2_min = a_val - b_val * b_val / e_sn
sn_ll = sn_const - 0.5 * chi2_min
print(f"  SN: {float(sn_ll):.5f}  (sn_const={float(sn_const):.5f}, chi2_min/2={float(0.5*chi2_min):.5f})")

total_nuts = float(sn_ll)

# BAO
for z_b, y_b, c_inv_b, obs_code, ll_const_b in prep.bao:
    dm_b = jnp.interp(z_b, z_low, dm_low)
    h_b = jnp.interp(z_b, z_low, h_low)
    dm_rd_pred = dm_b / rd
    dh_rd_pred = (C_KM_S / h_b) / rd
    dv_rd_pred = (z_b * dm_b ** 2 * (C_KM_S / h_b)) ** (1.0 / 3.0) / rd
    if obs_code == 1:
        y_pred = dh_rd_pred
    elif obs_code == 2:
        y_pred = dv_rd_pred
    else:
        y_pred = dm_rd_pred
    r_b = y_b - y_pred
    bao_ll = ll_const_b + (-0.5 * (r_b @ (c_inv_b @ r_b)))
    print(f"  BAO obs_code={obs_code}: {float(bao_ll):.5f}  (ll_const={float(ll_const_b):.5f})")
    total_nuts += float(bao_ll)

# Planck
if prep.include_planck:
    z_star = 1089.92
    h_cmb = _hz_grid_lcdm_jax(theta[0], theta[1], z_cmb)
    dm_cmb = _comoving_dist_jax(z_cmb, h_cmb)
    dm_star = float(jnp.interp(jnp.array([z_star]), z_cmb, dm_cmb)[0])
    r_pred = float(jnp.sqrt(om) * dm_star * h0 / C_KM_S)
    la_pred = float(jnp.pi * dm_star / prep.rs_star)
    planck_chi2 = (r_pred - float(prep.r_obs))**2 / float(prep.sigma_r)**2 + \
                  (la_pred - float(prep.la_obs))**2 / float(prep.sigma_la)**2
    planck_ll_nuts = float(prep.planck_const) - 0.5 * planck_chi2
    print(f"  Planck: {planck_ll_nuts:.5f}  (planck_const={float(prep.planck_const):.5f})")
    total_nuts += planck_ll_nuts

# fσ8
if prep.fsig8_z is not None:
    print(f"  fσ8 const: {float(prep.fsig8_const):.5f} (fsig8 is present)")

print(f"  TOTAL: {total_nuts:.5f}")

print()
print(f"Discrepancy (MCMC - NUTS): {total_mcmc - total_nuts:.5f}")

# Check SN const vs MCMC SN const
print()
print("=== SN constant check ===")
from ultra_slow_de.likelihood import SNLikelihoodCached
sn_cache = SNLikelihoodCached(sn)
print(f"SNLikelihoodCached._const: {sn_cache._const:.5f}")
print(f"NUTS sn_const: {float(sn_const):.5f}")
print(f"SN const diff: {sn_cache._const - float(sn_const):.5f}")
