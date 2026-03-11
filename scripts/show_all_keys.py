import json
with open('output/production_results.json') as f:
    r = json.load(f)
prod = r['production']
# Show all keys for model a
print("All A keys:")
for k in prod['a']:
    v = prod['a'][k]
    if isinstance(v, (int, float, str, bool)):
        print(f"  {k}: {v}")
    elif isinstance(v, dict):
        print(f"  {k}: dict with keys {list(v.keys())[:6]}")
    elif isinstance(v, list):
        print(f"  {k}: list len={len(v)}")
