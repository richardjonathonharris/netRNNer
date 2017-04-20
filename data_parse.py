import json
import pprint
import random

with open('nrdb.json', 'r') as f:
	data = json.load(f)

exist_keys = []
for card in data['data']:
	for key in card.keys():
		if key not in exist_keys:
			exist_keys.append(key)
		else:
			continue

def fix_faction(faction):
	faction = faction.title()
	if faction[0:7] == 'Neutral':
		return 'neutral'
	elif faction[0:3] != 'haas':
		return faction.replace('-', ' ')
	else:
		return faction.upper()

def exist_wrapper(input_dict, input_key):
	value = input_dict.get(input_key, '')
	if value == '':
		return ''
	else:
		return input_key + ': ' + str(value)

def text_wrapper(full_text):
	full_text = full_text.replace('[', ' ')
	full_text = full_text.replace(']', ' ')
	for tag in ['<strong>', '</strong>', '<trace>', '</trace>']:
		full_text = full_text.replace(tag, '')
	return full_text

def text_blobber(input_dict):
	title = input_dict.get('title', '')
	faction_code = 'faction: ' + fix_faction(input_dict.get('faction_code', ''))
	type_code = 'type: ' + input_dict.get('type_code', '').upper()
	cost = exist_wrapper(input_dict, 'cost')
	strength = exist_wrapper(input_dict, 'strength')
	memory_cost = exist_wrapper(input_dict, 'memory_cost')
	advancement_cost = exist_wrapper(input_dict, 'advancement_cost')
	agenda_points = exist_wrapper(input_dict, 'agenda_points')
	trash_cost = exist_wrapper(input_dict, 'trash_cost')
	full_text = text_wrapper(input_dict.get('text', ''))
	flavor_text = exist_wrapper(input_dict, 'flavor_text')
	all_items = [title, faction_code, type_code,
	cost, strength, memory_cost, advancement_cost, agenda_points,
	trash_cost, full_text, flavor_text]
	new_string = ' '.join(all_items)
	return ' '.join(new_string.split()).replace('_', ' ').lower()

with open('card_text.txt', 'w') as f:
	for card in data['data']:
		f.write(text_blobber(card) + '\n')