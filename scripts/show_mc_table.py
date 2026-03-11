import json
with open('output/production_results.json') as f:
    r = json.load(f)
mc = r['model_comparison']
print("delta_vs_lcdm:")
for m, v in mc['delta_vs_lcdm'].items():
    print(f"  {m}: {v}")
print()
print("raw:")
for m in ('lcdm','a','b'):
    v = mc['raw'][m]
    aic = v.get('aic', v.get('AIC', 'na'))
    bic = v.get('bic', v.get('BIC', 'na'))
    lnl = v.get('max_lnL', v.get('lnL', v.get('best_loglike', 'na')))
    print(f"  {m}: {list(v.items())[:8]}")
