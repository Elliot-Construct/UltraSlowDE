import json
with open('output/production_results.json') as f:
    r = json.load(f)
print("Model A posterior keys:", list(r['production']['a']['posterior'].keys()))
print("Model B posterior keys:", list(r['production']['b']['posterior'].keys()))
print("LCDM posterior keys:", list(r['production']['lcdm']['posterior'].keys()))
