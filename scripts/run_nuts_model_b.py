"""Run ΛCDM with NUTS for convergence fix (R̂=1.106 in standard MCMC)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ultra_slow_de.production_run import SamplerConfig, run_single_model
from src.ultra_slow_de.production_run import load_pantheon_plus, load_all_bao

sn = load_pantheon_plus()
bao = load_all_bao()
datasets = [(sn, "mb")] + bao
cfg = SamplerConfig(use_nuts=True)

print("Running ΛCDM with NUTS...")
result = run_single_model("lcdm", datasets, cfg, rd=147.09, run_tag="nuts_lcdm_fix")

print("DONE")
print(f"max_rhat: {result['max_split_rhat']:.4f}")
print(f"min_ess:  {result['min_ess']:.1f}")
print(f"accept:   {result['accept_rate_mean']:.3f}")
print(f"best_lnL: {result['best_loglike_found']:.4f}")
for p, v in result["posterior"].items():
    hi = v["ci68_high"] - v["mean"]
    lo = v["mean"] - v["ci68_low"]
    print(f"  {p}: {v['mean']:.4f} +{hi:.4f}/-{lo:.4f}  (best={v['best_fit']:.4f})")
