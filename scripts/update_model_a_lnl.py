"""Update production_results.json with corrected Model A lnL and IC values.

The correction: Model A was run with NUTS (JAX frame) while LCDM/ModelB used MCMC
(scipy frame). The NUTS JAX loglike underestimates lnL by ~0.78 at Model A's MAP
relative to the scipy formula. After scipy optimization from the NUTS MAP, the
correct scipy-frame lnL for Model A is 811.474.
"""
import sys
sys.path.insert(0, 'src')
import json
import numpy as np

# Load saved optimization result
with open('output/model_a_scipy_map.json') as f:
    opt = json.load(f)['model_a_scipy_map']

ll_a_new = opt['best_loglike_scipy']
print(f"Corrected Model A scipy lnL: {ll_a_new:.4f}")

# Constants
N_DATA = 1644
k_a = 6
k_lcdm = 2

# Recompute AIC/BIC
aic_a_new = -2*ll_a_new + 2*k_a
bic_a_new = -2*ll_a_new + k_a * np.log(N_DATA)
aicc_a_new = aic_a_new + 2*k_a*(k_a+1)/(N_DATA - k_a - 1)

# Load production results
with open('output/production_results.json') as f:
    results = json.load(f)

# Get LCDM AIC/BIC
lcdm = results['production']['lcdm']
ll_lcdm = lcdm['best_loglike_found']
aic_lcdm = lcdm['information_criteria']['aic']
bic_lcdm = lcdm['information_criteria']['bic']
aicc_lcdm = lcdm['information_criteria']['aicc']
print(f"LCDM lnL={ll_lcdm:.4f}, AIC={aic_lcdm:.4f}, BIC={bic_lcdm:.4f}")

delta_aic = aic_a_new - aic_lcdm
delta_bic = bic_a_new - bic_lcdm
delta_aicc = aicc_a_new - aicc_lcdm

print(f"\nCorrected Model A:")
print(f"  AIC = {aic_a_new:.4f}  (was {results['production']['a']['information_criteria']['aic']:.4f})")
print(f"  BIC = {bic_a_new:.4f}  (was {results['production']['a']['information_criteria']['bic']:.4f})")
print(f"  AICc = {aicc_a_new:.4f}  (was {results['production']['a']['information_criteria']['aicc']:.4f})")
print(f"  ΔAIC = {delta_aic:.4f}  (was {results['model_comparison']['delta_vs_lcdm']['a']['aic']:.4f})")
print(f"  ΔBIC = {delta_bic:.4f}  (was {results['model_comparison']['delta_vs_lcdm']['a']['bic']:.4f})")

# Update production_results.json
results['production']['a']['best_loglike_found'] = float(ll_a_new)
results['production']['a']['max_loglike_all_samples'] = float(ll_a_new)
results['production']['a']['information_criteria']['aic'] = float(aic_a_new)
results['production']['a']['information_criteria']['bic'] = float(bic_a_new)
results['production']['a']['information_criteria']['aicc'] = float(aicc_a_new)

# Update best_fit parameters to scipy MAP
opt_keys = {'h0': opt['h0'], 'omega_m': opt['omega_m'],
            'w0': opt['w0'], 'B': opt['B'], 'C': opt['C'], 'omega': opt['omega']}
for param, val in opt_keys.items():
    if param in results['production']['a']['posterior']:
        results['production']['a']['posterior'][param]['best_fit'] = float(val)

# Update best_fit_theta list too
results['production']['a']['best_fit_theta'] = [float(opt[p]) for p in ('h0','omega_m','w0','B','C','omega')]

# Update model_comparison deltas
results['model_comparison']['delta_vs_lcdm']['a']['aic'] = float(delta_aic)
results['model_comparison']['delta_vs_lcdm']['a']['bic'] = float(delta_bic)
results['model_comparison']['delta_vs_lcdm']['a']['aicc'] = float(delta_aicc)

# Update model_comparison raw lnL and AIC/BIC
results['model_comparison']['raw']['a']['aic'] = float(aic_a_new)
results['model_comparison']['raw']['a']['bic'] = float(bic_a_new)
results['model_comparison']['raw']['a']['aicc'] = float(aicc_a_new)
if 'max_lnL' in results['model_comparison']['raw']['a']:
    results['model_comparison']['raw']['a']['max_lnL'] = float(ll_a_new)

# Add correction note
results['production']['a']['lnl_correction_note'] = (
    "best_loglike_found corrected from NUTS-JAX (810.310) to scipy-formula (811.474) "
    "via L-BFGS-B optimization from the NUTS MAP, ensuring consistent normalization "
    "with LCDM and ModelB MCMC loglike values."
)

with open('output/production_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nproduction_results.json updated.")
print(f"Summary: ΔAIC(A vs LCDM) = {delta_aic:+.2f}, ΔBIC = {delta_bic:+.2f}")
