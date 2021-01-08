# read config file
with open("config.json", "r") as f_json:
    config = json.load(f_json)

PREVIOUS_SIZE = config['previous_size']
CURRENT_SIZE = config['current_size']
FUTURE_SIZE = config['future_size']
RECEPTIVE_SIZE = PREVIOUS_SIZE + CURRENT_SIZE + FUTURE_SIZE

mod = (CURRENT_SIZE - (size_of_source % CURRENT_SIZE)) % CURRENT_SIZE