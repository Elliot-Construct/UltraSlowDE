"""Run NUTS for ΛCDM to match the NUTS convergence standard used for Models A and B.

Usage:
    python scripts/run_nuts_lcdm.py

Outputs:
  - Prints convergence diagnostics and best lnL.
  - Updates output/production_results.json in-place:
      production.lcdm.{max_split_rhat, min_ess, accept_rate_mean, accept_rates,
                        best_loglike_found, max_loglike_all_samples,
                        chain_best_loglike, posterior, best_fit_theta,
                        steps_used, n_chains, backend, information_criteria, waic}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ultra_slow_de.builtin_data import load_all_bao
from ultra_slow_de.ingest import load_pantheon_plus
from ultra_slow_de.sampler_nuts import run_nuts_lcdm
from ultra_slow_de.sampler import _rhat_split, _aggregate_ess

# ── Reproduce the production dataset setup ────────────────────────────────
sn_ds = load_pantheon_plus()  # uses DATA_ROOT = <project>/data automatically
bao_datasets = load_all_bao()

datasets = [(sn_ds, "mb")] + [(ds, obs) for ds, obs in bao_datasets]

# Fiducial sound horizon used throughout the production run
RD = 147.09

# ── Run NUTS ──────────────────────────────────────────────────────────────
LCDM_THETA0 = np.array([67.74, 0.3089])   # Planck 2018 central values

print("Running NUTS for ΛCDM: 4 chains × 4000 post-warmup samples …")
result = run_nuts_lcdm(
    datasets=datasets,
    theta0=LCDM_THETA0,
    rd=RD,
    include_planck=True,
    n_chains=4,
    n_warmup=800,
    n_samples=4000,
    seed=99999,
    init_scatter=0.3,
    float_sigma8=False,   # ΛCDM has no fσ8 data in this run
)

# ── Diagnostics ───────────────────────────────────────────────────────────
chains = result.chains            # (n_chains, n_samples, n_params)
lls    = result.loglike           # (n_chains, n_samples)

accept_rates = list(result.accept_rates.tolist())
accept_mean  = float(np.mean(result.accept_rates))

flat_chains = chains.reshape(-1, chains.shape[-1])
flat_lls    = lls.flatten()

idx_max   = int(np.argmax(flat_lls))
best_ll   = float(flat_lls[idx_max])
best_theta = flat_chains[idx_max].tolist()
chain_best = [float(np.max(lls[i])) for i in range(lls.shape[0])]

rhat = _rhat_split(chains)
ess  = _aggregate_ess(chains)
max_rhat = float(np.max(rhat))
min_ess  = float(np.min(ess))

print(f"\n=== ΛCDM NUTS results ===")
print(f"  accept_rates : {[round(a, 3) for a in accept_rates]}")
print(f"  best lnL     : {best_ll:.4f}")
print(f"  max split-R̂  : {max_rhat:.4f}")
print(f"  min ESS      : {min_ess:.1f}")
print(f"  chain-best   : {[round(c, 3) for c in chain_best]}")
print(f"  best_theta   : h0={best_theta[0]:.4f}, omega_m={best_theta[1]:.4f}")

# ── Compute scipy-consistent lnL at NUTS MAP ─────────────────────────────
# This is needed for consistent model comparison (matches what we report
# for Models A and B in the paper).
from ultra_slow_de.inference import joint_logposterior
from ultra_slow_de.params import CosmoParams

cosmo_map = CosmoParams(h0=best_theta[0], omega_m=best_theta[1])
lnl_scipy = joint_logposterior(datasets, cosmo_map, rd=RD, include_planck=True)
print(f"  scipy lnL at NUTS MAP: {lnl_scipy:.4f}")

# ── Build per-parameter posterior summary ─────────────────────────────────
param_names = result.param_names
posterior = {}
for j, p in enumerate(param_names):
    samp = flat_chains[:, j]
    q16, q50, q84 = np.percentile(samp, [16, 50, 84])
    posterior[p] = {
        "mean": float(np.mean(samp)),
        "median": float(q50),
        "ci68_low": float(q16),
        "ci68_high": float(q84),
        "std": float(np.std(samp)),
        "best_fit": float(best_theta[j]),
    }

# ── Update production_results.json ───────────────────────────────────────
OUT = ROOT / "output" / "production_results.json"
with open(OUT) as fh:
    prod = json.load(fh)

lcdm = prod["production"]["lcdm"]
lcdm.update(
    backend="blackjax-nuts",
    n_chains=4,
    steps_used=4000,
    accept_rates=accept_rates,
    accept_rate_mean=accept_mean,
    chain_best_loglike=chain_best,
    best_loglike_found=best_ll,
    max_loglike_all_samples=best_ll,
    best_fit_theta=best_theta,
    posterior=posterior,
    max_split_rhat=max_rhat,
    min_ess=min_ess,
    split_rhat={p: float(rhat[j]) for j, p in enumerate(param_names)},
    ess_aggregate={p: float(ess[j]) for j, p in enumerate(param_names)},
)

# Update the model comparison consistent_lnl for ΛCDM as well
if "model_comparison" in prod:
    mc = prod["model_comparison"]
    for key in ("consistent_lnl", "reference_lnl"):
        if key in mc:
            mc[key] = lnl_scipy
    # Also update any ΛCDM-specific row that holds the lnL
    for entry in mc.get("rows", []):
        if entry.get("model", "").lower() in ("lcdm", "lambda_cdm"):
            entry["lnl_scipy"] = lnl_scipy
            entry["best_ll"] = best_ll

prod["production"]["lcdm"] = lcdm

with open(OUT, "w") as fh:
    json.dump(prod, fh, indent=2)

print(f"\n  production_results.json updated (ΛCDM NUTS: R̂={max_rhat:.4f}, ESS={min_ess:.0f})")
print("Done.")
