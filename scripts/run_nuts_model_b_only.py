"""Run Model B with NUTS sampler and update production_results.json in-place."""
import json
import time
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ultra_slow_de.sampler_nuts import run_nuts_model_b
from ultra_slow_de.ingest import load_pantheon_plus
from ultra_slow_de.builtin_data import load_all_bao

RESULTS_PATH = "output/production_results.json"

print("Loading datasets...")
sn = load_pantheon_plus()
bao = load_all_bao()
datasets = [(sn, "mb")] + bao

rd = 147.09  # Mpc, standard value used in production

# Use current MAP as theta0
with open(RESULTS_PATH) as f:
    d = json.load(f)

b_current = d["production"]["b"]
bf = b_current["best_fit_theta"]
theta0 = np.array(bf[:3], dtype=np.float64)
print(f"theta0 for Model B: {theta0}")

N_WARMUP = 1500
N_SAMPLES = 3000
N_CHAINS = 4
print(f"Running NUTS: n_warmup={N_WARMUP}, n_samples={N_SAMPLES}, n_chains={N_CHAINS}...")
t0 = time.perf_counter()

result = run_nuts_model_b(
    datasets=datasets,
    theta0=theta0,
    rd=rd,
    include_planck=True,
    n_chains=N_CHAINS,
    n_warmup=N_WARMUP,
    n_samples=N_SAMPLES,
    seed=42,
)

elapsed = time.perf_counter() - t0
print(f"Done in {elapsed:.1f}s")
print(f"R_hat per param: {dict(zip(result.param_names, result.rhat_per_param))}")
print(f"ESS per param:   {dict(zip(result.param_names, result.ess_per_param))}")
print(f"Accept rates:    {result.accept_rates}")

# Flatten chains
chains = result.chains  # (n_chains, n_samples, n_params)
flat = chains.reshape(-1, chains.shape[-1])

lls = result.loglike.reshape(-1)
idx_max = int(np.argmax(lls))
max_ll = float(lls[idx_max])
best_theta = flat[idx_max]

print(f"Best lnL: {max_ll:.4f}")
print(f"Best theta: {dict(zip(result.param_names, best_theta.tolist()))}")

# Build posterior summaries
posterior = {}
for j, p in enumerate(result.param_names):
    vals = flat[:, j]
    q16, q50, q84 = np.percentile(vals, [16, 50, 84])
    posterior[p] = {
        "mean": float(np.mean(vals)),
        "median": float(q50),
        "ci68_low": float(q16),
        "ci68_high": float(q84),
        "std": float(np.std(vals)),
        "best_fit": float(best_theta[j]),
    }

rhat_map = {p: float(v) for p, v in zip(result.param_names, result.rhat_per_param)}
ess_map = {p: float(v) for p, v in zip(result.param_names, result.ess_per_param)}

print("\nPosterior means:")
for p, v in posterior.items():
    lo = v["ci68_low"]
    hi = v["ci68_high"]
    print(f"  {p}: {v['mean']:.4f}  [{lo:.4f}, {hi:.4f}]")

# Update production_results.json
b_new = dict(b_current)
b_new.update({
    "steps_used": int(N_SAMPLES),
    "burn_in_steps": 0,
    "n_chains": N_CHAINS,
    "accept_rates": [float(x) for x in result.accept_rates],
    "accept_rate_mean": float(np.mean(result.accept_rates)),
    "sampler_backend": result.backend,
    "param_names": list(result.param_names),
    "posterior": posterior,
    "split_rhat": rhat_map,
    "ess_aggregate": ess_map,
    "max_split_rhat": float(np.nanmax(result.rhat_per_param)),
    "min_ess": float(np.min(result.ess_per_param)),
    "chain_best_loglike": [float(np.max(result.loglike[i])) for i in range(result.loglike.shape[0])],
    "best_loglike_found": max_ll,
    "max_loglike_all_samples": max_ll,
    "best_fit_theta": [float(x) for x in best_theta],
    "chains": result.chains.tolist(),
    "loglike": result.loglike.tolist(),
    "runtime_seconds": float(elapsed),
})

d["production"]["b"] = b_new
with open(RESULTS_PATH, "w") as f:
    json.dump(d, f, indent=2)

print(f"\nSaved updated results to {RESULTS_PATH}")
print(f"max_split_rhat={b_new['max_split_rhat']:.4f}, min_ess={b_new['min_ess']:.1f}")
