import requests
import json

r = requests.get('https://netrunnerdb.com/api/2.0/public/cards')

with open('nrdb.json', 'w') as f:
	json.dump(r.json(), f)