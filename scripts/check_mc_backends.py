import json
with open('output/production_results.json') as f:
    r = json.load(f)
prod = r['production']
for m in ('lcdm','a','b'):
    v = prod[m]
    be = v.get('sampler_backend', 'na')
    ll = v['best_loglike_found']
    st = v.get('steps_used', 'na')
    print(f"{m}: backend={be}, lnL={ll:.4f}, steps={st}")
mc = r['model_comparison']
print()
print("model_comparison keys:", list(mc.keys())[:15])
for k, val in mc.items():
    if isinstance(val, dict):
        print(f"  {k}: {list(val.keys())[:5]}")
    else:
        print(f"  {k}: {val}")
