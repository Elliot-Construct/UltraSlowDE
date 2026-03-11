"""Final validation: cross-check production_results.json against manuscript tables."""
import json

with open('output/production_results.json') as f:
    data = json.load(f)

prod = data.get('production', {})
mc = data.get('model_comparison', {})

print("=== BEST-FIT PARAMETERS & CONVERGENCE ===")
for k in ['lcdm', 'a', 'b']:
    m = prod.get(k, {})
    lnl = m.get('best_loglike_found', 'N/A')
    theta = m.get('best_fit_theta', [])
    pnames = m.get('param_names', [])
    ic = m.get('information_criteria', {})
    rhat = m.get('max_split_rhat', 'N/A')
    ess = m.get('min_ess', 'N/A')
    backend = m.get('backend', m.get('sampler_backend', 'N/A'))
    
    lnl_str = f"{lnl:.2f}" if isinstance(lnl, float) else str(lnl)
    aic = ic.get('aic', 'N/A')
    bic = ic.get('bic', 'N/A')
    aic_str = f"{aic:.2f}" if isinstance(aic, float) else str(aic)
    bic_str = f"{bic:.2f}" if isinstance(bic, float) else str(bic)
    
    print(f"Model {k}: lnL_best={lnl_str}, max_Rhat={rhat}, min_ESS={ess}, backend={backend}")
    print(f"  AIC={aic_str}, BIC={bic_str}")
    if theta and pnames:
        pairs = zip(pnames, theta)
        param_str = ", ".join(f"{n}={v:.4f}" for n, v in pairs)
        print(f"  params: {param_str}")
    print()

print("=== MODEL COMPARISON (delta vs LCDM) ===")
dvl = mc.get('delta_vs_lcdm', {})
for k in ['lcdm', 'a', 'b']:
    d = dvl.get(k, {})
    dbic = d.get('bic', 'N/A')
    daic = d.get('aic', 'N/A')
    dwaic = d.get('waic', 'N/A')
    dbic_str = f"{dbic:.2f}" if isinstance(dbic, float) else str(dbic)
    daic_str = f"{daic:.2f}" if isinstance(daic, float) else str(daic)
    dwaic_str = f"{dwaic:.2f}" if isinstance(dwaic, float) else str(dwaic)
    print(f"  delta_{k}_vs_lcdm: DBIC={dbic_str}, DAIC={daic_str}, DWAIC={dwaic_str}")

print()
print("=== SENSITIVITY ===")
sens = data.get('sensitivity', {})
print(json.dumps({k: v for k, v in sens.items() if not isinstance(v, dict) or len(str(v)) < 200}, indent=2))
