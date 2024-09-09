import json

import numpy as np

# Carica i dati JSON da un file
with open('./data/all_data/data.json', 'r') as f:
    data = json.load(f)

# Inizializza una lista per contenere tutti i valori di "x"
all_x_values = []

# Itera attraverso le chiavi numeriche in "user_data"
for key in data['user_data']:
    x_data = data['user_data'][key]['x']

    # Itera attraverso le sotto-chiavi di "x" e raccogli i valori
    for sub_data in x_data:
        all_x_values.extend(sub_data)

# Converti la lista di valori in un array numpy
all_x_values = np.array(all_x_values)

# Calcola la media e la deviazione standard
mean_x = np.mean(all_x_values)
std_x = np.std(all_x_values)

# Stampa i risultati
print(f"Mean: {mean_x}")
print(f"Standard Deviation: {std_x}")
