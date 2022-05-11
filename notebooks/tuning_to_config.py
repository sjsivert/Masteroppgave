# %%
from typing import OrderedDict

import yaml

# %%
# Read txt file
filename = "../models/dataset_2-lstm-multivariate-tune-400-trails/tuning.txt"
output_filename= "../parsed_tuning.yaml"
is_for_autoencoder_config = True
number_of_features = 4
stateful_lstm = True

def read_file(filename):
	with open(filename, 'r') as f:
			return f.readlines()
lines = read_file(filename)
# %%
def raw_tuning_cleaning(lines):
	"""
	Raw clean the file so that only the dictonary is left
	"""
	lines_cleaned = []
	for line in lines:
		remove_line = False
		if 'Parameter Tuning' in line: remove_line = True
		if '\n' == line: remove_line = True
		if 'Model tuning:' in line: remove_line = True
		if 'Parameters:' in line: remove_line = True
		if 'Error:' in line:
			line = line[51:]
		if not remove_line:
			lines_cleaned.append(line)
	return lines_cleaned

lines_cleaned = raw_tuning_cleaning(lines)
(lines_cleaned)

# %%

def parse_layers(list_of_layers):
	"""
	Parse the layers dict and return layers on the correct format
	"""
	list_of_layers_parsed = []
	for i in range(len(list_of_layers)):
		if "OrderedDict" in list_of_layers[i]:
			layer = {
				"hidden_size": int(list_of_layers[i+1]),
				"dropout": float(list_of_layers[i+3]),
				"recurrent_dropout": float(list_of_layers[i+5])
			}
			list_of_layers_parsed.append(layer)
	return list_of_layers_parsed


# %%

def extract_config(lines):
	"""
	Parse dictionaries and return them in correct config format
	"""
	model_structure = []
	for model in lines:
		line = model
		line = line.split()
		lines_cleaned = []
		for l in line:
			l = l.rstrip(":")
			l = l.rstrip("'")
			l = l.rstrip(",")
			l = l.rstrip(")]}")
			l = l.lstrip("({[")
			l = l.lstrip("'")
			lines_cleaned.append(l)
		dict = {}
		for l in range(0, len( lines_cleaned)-1, 2 ):
			key = lines_cleaned[l]
			value = lines_cleaned[l+1]
			if not key == 'layers':
				dict[key] = value
			else:
				list_of_layers = lines_cleaned[l:]
				layers_list = parse_layers(list_of_layers)
				dict["layers"] = layers_list
				break
		# Delete unnececary keys
		dict.pop("number_of_layers")
		for key, value in dict.items():
			# Make alle values integer
			not_a_list_or_optimizer = ("optimizer" not in key and not (type(value) is list))
			dict[key] = float(value) if not_a_list_or_optimizer else value
		
		# Add necessary keys
		dict["number_of_features"] = number_of_features
		dict["stateful_lstm"] = stateful_lstm
		dict["optimizer_name"] = dict["optimizer_name"].rstrip("'")
		dict["time_series_id"] = int(dict["time_series_id"])
		if dict.get("batch_size"):
			dict["batch_size"] = int(dict["batch_size"])
		dict["epochs"] = int(dict.pop("number_of_epochs"))
		
		# Add lstm key if config is for autoencoder
		if is_for_autoencoder_config:
			autoencoder_dict = {}
			autoencoder_dict["time_series_id"] = dict.pop("time_series_id")
			autoencoder_dict["lstm"] = dict
			dict = autoencoder_dict
		
		model_structure.append(dict)
			
	print(yaml.dump(model_structure))
	file = open(output_filename, "w")
	yaml.dump(model_structure, file)
	file.close()

extract_config(lines_cleaned)

"""
	model_structure:
		- time_series_id: 11852
			lstm:
				optimizer_name: 'Adam'
				stateful_lstm: True
				loss: "mae"
				learning_rate: 0.00629378
				number_of_features: 1
				epochs: 17
				layers:
					- hidden_size: 19
						dropout: 0.16012
						recurrent_dropout: 0.0942
"""




	