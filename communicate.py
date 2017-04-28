from twilio.rest import Client
import json

with open('secret.json', 'r') as f:
	secret = json.loads(f.read())

def jobs_done(message):
	client = Client(secret['twilio']['account_sid'],
		 secret['twilio']['auth_token'])

	client.messages.create(
		to="+12022560198",
		from_="+12028319873",
		body=message)

	return None