"""Print updated table values from production_results.json."""
import json
import numpy as np
from pathlib import Path

root = Path(__file__).resolve().parents[1]
data = json.loads((root / "output" / "production_results.json").read_text())
prod = data["production"]
mc = data["model_comparison"]

print("=== MODEL COMPARISON ===")
for m in ("lcdm", "a", "b"):
    p = prod[m]
    ic = p["information_criteria"]
    w_val = p.get("waic", {}).get("waic", float("nan"))
    dlcdm = mc["delta_vs_lcdm"][m]
    dw = dlcdm.get("waic", float("nan"))
    ll = p["max_loglike_all_samples"]
    dll = ll - prod["lcdm"]["max_loglike_all_samples"]
    aic_d = dlcdm["aic"]
    aicc_d = dlcdm["aicc"]
    bic_d = dlcdm["bic"]
    print(
        f"  {m}: lnL={ll:.3f}  d_lnL={dll:+.3f}"
        f"  dAIC={aic_d:+.2f}  dAICc={aicc_d:+.2f}"
        f"  dBIC={bic_d:+.1f}  dWAIC={dw:+.1f}"
    )

print()
print("=== PARAMETER CONSTRAINTS ===")
for m in ("lcdm", "a", "b"):
    post = prod[m]["posterior"]
    print(f"  --- {m} ---")
    for pname, v in post.items():
        lo = v["median"] - v["ci68_low"]
        hi = v["ci68_high"] - v["median"]
        print(
            f"    {pname}: mean={v['mean']:.4f}"
            f"  +{hi:.4f}/-{lo:.4f}"
            f"  best={v['best_fit']:.4f}"
        )

print()
print("=== CONVERGENCE ===")
for m in ("lcdm", "a", "b"):
    p = prod[m]
    nc = p["n_chains"]
    ns = p["steps_used"]
    acc = p["accept_rate_mean"]
    rhat = p["max_split_rhat"]
    ess = p["min_ess"]
    print(f"  {m}: {nc}x{ns}  acc={acc:.3f}  R-hat={rhat:.4f}  ESS={ess:.1f}")
