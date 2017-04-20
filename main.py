import requests

r = requests.get('https://netrunnerdb.com/api/2.0/public/cards')
data = r.json()

