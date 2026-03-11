"""Diagnose LCDM loglike discrepancy between NUTS best-fit and MCMC formula."""
import sys
sys.path.insert(0, 'src')
import json
import numpy as np

with open('output/production_results.json') as f:
    results = json.load(f)

lcdm_r = results['production']['lcdm']
h0_mcmc = lcdm_r['posterior']['h0']['best_fit']
om_mcmc = lcdm_r['posterior']['omega_m']['best_fit']
print(f"LCDM MCMC best-fit: h0={h0_mcmc:.5f}, omega_m={om_mcmc:.6f}")
print(f"MCMC saved best_loglike: {lcdm_r['best_loglike_found']:.4f}")

from ultra_slow_de.ingest import load_pantheon_plus
from ultra_slow_de.builtin_data import load_all_bao
from ultra_slow_de.inference import joint_logposterior
from ultra_slow_de.params import CosmoParams

sn = load_pantheon_plus()
bao = load_all_bao()
datasets = [(sn, "mb")] + bao

# Compute MCMC formula loglike at MCMC best-fit
cosmo_mcmc = CosmoParams(h0=h0_mcmc, omega_m=om_mcmc)
ll_at_mcmc = joint_logposterior(datasets, cosmo_mcmc, include_planck=True)
print(f"MCMC-formula loglike at MCMC bf:   {ll_at_mcmc:.4f}")

# Load the NUTS LCDM results
nuts_results_file = 'output/nuts_lcdm_results.json'
try:
    with open(nuts_results_file) as f:
        nuts_r = json.load(f)
    h0_nuts = nuts_r['posterior']['h0']['best_fit']
    om_nuts = nuts_r['posterior']['omega_m']['best_fit']
    best_ll_nuts = nuts_r['best_loglike_found']
    print(f"\nLCDM NUTS best-fit: h0={h0_nuts:.5f}, omega_m={om_nuts:.6f}")
    print(f"NUTS saved best_loglike: {best_ll_nuts:.4f}")
    
    # Compute MCMC formula loglike at NUTS best-fit
    cosmo_nuts = CosmoParams(h0=h0_nuts, omega_m=om_nuts)
    ll_at_nuts = joint_logposterior(datasets, cosmo_nuts, include_planck=True)
    print(f"MCMC-formula loglike at NUTS bf:   {ll_at_nuts:.4f}")
    print(f"Difference (MCMC-at-nuts vs NUTS-saved): {ll_at_nuts - best_ll_nuts:.4f}")
except FileNotFoundError:
    print(f"\nNUTS results file {nuts_results_file} not found.")
    print("Checking for NUTS results in production_results...")

# Also check what NUTS lnL value we can reproduce independently
# The NUTS best_ll=806.635 was from the run. Let's check via JAX NUTS path.
from ultra_slow_de.sampler_nuts import _prepare, _make_logpost_jax_generic
import jax.numpy as jnp
import jax

# Prepare data
class DummyCfg:
    pass

prep = _prepare(datasets, rd=147.09, include_planck=True)

# LCDM hz_fn
from ultra_slow_de.sampler_nuts import _hz_grid_lcdm_jax

bounds_np = np.array([[60.0, 80.0], [0.1, 0.5]], dtype=np.float64)
def hz_fn(theta, z_grid):
    return _hz_grid_lcdm_jax(theta[0], theta[1], z_grid)

logpost, psi_to_theta, theta_to_psi = _make_logpost_jax_generic(
    prep, hz_fn, bounds_np, use_lcdm_for_planck=False
)

# Compute loglike at MCMC best-fit using NUTS JAX path
theta_mcmc = jnp.array([h0_mcmc, om_mcmc])
from ultra_slow_de.sampler_nuts import _make_logpost_jax_generic

# We need the _ll_theta function, not _logpost_psi
# Hack: create a zero Jacobian psi=0 -> theta at center, then eval
# Actually, just extract _ll_theta value by calling logpost(theta_to_psi(theta))
psi_mcmc = theta_to_psi(theta_mcmc)
# ll = logpost(psi) - log_jacobian(psi)
# We need log_jacobian
lo = jnp.array(bounds_np[:, 0])
hi = jnp.array(bounds_np[:, 1])
width = hi - lo
sig = jax.nn.sigmoid(psi_mcmc)
log_jac_mcmc = float(jnp.sum(jnp.log(sig) + jnp.log1p(-sig) + jnp.log(width)))
ll_jax_at_mcmc = float(logpost(psi_mcmc)) - log_jac_mcmc
print(f"\nJAX-NUTS loglike at MCMC bf:       {ll_jax_at_mcmc:.4f}")
print(f"Discrepancy (MCMC-formula vs JAX-NUTS): {ll_at_mcmc - ll_jax_at_mcmc:.4f}")
