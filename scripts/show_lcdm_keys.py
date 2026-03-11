import json
with open('output/production_results.json') as f:
    r = json.load(f)
prod = r['production']
print("LCDM production keys:", list(prod['lcdm'].keys())[:15])
print("A production keys:", list(prod['a'].keys())[:15])
