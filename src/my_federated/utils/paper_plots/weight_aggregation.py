import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def softmax(matrix, temp: float = 1):
    exp_matrix = np.exp(matrix / temp)
    softmax_matrix = exp_matrix / np.sum(exp_matrix, axis=0, keepdims=True)
    return softmax_matrix


# Impostiamo i parametri
epochs = 100  # Numero di epoche
clients = 10  # Numero di clients
fontsize_title = 22
fontsize_axis_name = 20
fontsize_axis_values = 18
ticks_pos = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]  # Posizioni dei tick
tick_labels = [str(tick_pos) for tick_pos in ticks_pos]  # Nomi dei tick

# Generiamo i dati per le heatmap
np.random.seed(42)

# Heatmap uniforme (sinistra) con colore piatto
data_left = np.full((clients, epochs), 1.0 / clients)  # Valore uniforme

# Heatmap in alto a destra (clients con valori alti cambiano nel tempo)
data_top_right = np.full((clients, epochs), 1.0 / clients)
start_mid = epochs // 2
end_mid = (3 * epochs) // 4
data_top_right[:, start_mid:end_mid] += np.random.normal(loc=0.0, scale=0.1, size=(clients, end_mid - start_mid))
for epoch in range(epochs // 2):
    high_clients = np.random.choice(clients, size=3, replace=False)  # Cambiano i clients con valori alti
    data_top_right[high_clients, epoch] += (np.random.rand() * 3)

data_top_right = softmax(data_top_right)

# Heatmap in basso a destra (solo pochi clients sempre alti, ma cambiano nel tempo)
data_bottom_right = np.random.normal(loc=0.5, scale=0.1, size=(clients, epochs))
for epoch in range(epochs):
    high_clients = np.random.choice(clients, size=3, replace=False)  # Cambiano i clients con valori alti
    data_bottom_right[high_clients, epoch] += np.random.rand()

data_bottom_right = softmax(data_bottom_right, temp=0.1)

# Determiniamo il range globale della scala colore
vmin = min(data_left.min(), data_top_right.min(), data_bottom_right.min())
vmax = max(data_left.max(), data_top_right.max(), data_bottom_right.max())

# Creiamo la figura con 2 righe e 2 colonne
fig, axs = plt.subplots(2, 1, figsize=(20, 12.5))

# Heatmap a sinistra (centrata verticalmente)
"""sns.heatmap(data_left, ax=axs[0, 0], cmap="viridis", cbar=False, vmin=vmin, vmax=vmax)
axs[0, 0].set_title("(a)", fontsize=fontsize_title)
axs[0, 0].set_xlabel("Epoch", fontsize=fontsize_axis_name)
axs[0, 0].set_ylabel("Client", fontsize=fontsize_axis_name)
axs[0, 0].set_xticks(ticks_pos)  # Imposta le posizioni dei tick
axs[0, 0].set_xticklabels(tick_labels)  # Imposta i testi per ogni tick
# Ruota i numeri sull'asse y di 90° (verticale)
axs[0, 0].tick_params(axis='y', rotation=0)
# Ruota i numeri sull'asse x di 30°
axs[0, 0].tick_params(axis='x', rotation=0)
axs[0, 0].tick_params(axis='both', labelsize=fontsize_axis_values)
axs[0, 0].grid(False)  # Disabilita la griglia

axs[1, 0].set_visible(False)  # Nasconde il subplot in basso a sinistra
axs[1, 0].grid(False)"""

# Heatmap in alto a destra
ax1 = axs[0]
sns.heatmap(data_top_right, ax=ax1, cmap="viridis", cbar=False, vmin=vmin, vmax=vmax)
ax1.set_title("(a)", fontsize=fontsize_title)
ax1.set_ylabel("Client ID", fontsize=fontsize_axis_name)
ax1.set_xticks(ticks_pos)  # Imposta le posizioni dei tick
ax1.set_xticklabels(tick_labels)  # Imposta i testi per ogni tick
# Ruota i numeri sull'asse y di 90° (verticale)
ax1.tick_params(axis='y', rotation=0)
# Ruota i numeri sull'asse x di 30°
ax1.tick_params(axis='x', rotation=0)
ax1.tick_params(axis='both', labelsize=fontsize_axis_values)
ax1.grid(False)  # Disabilita la griglia

# Heatmap in basso a destra
ax2 = axs[1]
sns.heatmap(data_bottom_right, ax=ax2, cmap="viridis", cbar=False, vmin=vmin, vmax=vmax)
ax2.set_title("(b)", fontsize=fontsize_title)
ax2.set_xlabel("Epoch", fontsize=fontsize_axis_name)
ax2.set_ylabel("Client ID", fontsize=fontsize_axis_name)
ax2.set_xticks(ticks_pos)  # Imposta le posizioni dei tick
ax2.set_xticklabels(tick_labels)  # Imposta i testi per ogni tick
# Ruota i numeri sull'asse y di 90° (verticale)
ax2.tick_params(axis='y', rotation=0)
# Ruota i numeri sull'asse x di 30°
ax2.tick_params(axis='x', rotation=0)
ax2.tick_params(axis='both', labelsize=fontsize_axis_values)
ax2.grid(False)  # Disabilita la griglia

# Aggiungiamo una legenda globale
cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])  # Posizioniamo la legenda a destra dell'intera figura
# cbar_ax = axs[2]
sns.heatmap(np.array([[vmin, vmax]]), cbar=True, cbar_ax=cbar_ax, cmap="viridis")
cbar_ax.set_ylabel("Aggregation Weight", fontsize=fontsize_axis_name)
cbar_ax.set_yticks([vmin, 1 / clients, vmax])
cbar_ax.set_xticks([])
cbar_ax.set_yticklabels([f"{vmin:.2f}", f"{1 / clients:.2f}", f"{vmax:.2f}"])
cbar_ax.tick_params(axis='y', rotation=0)
cbar_ax.tick_params(axis='y', labelsize=fontsize_axis_values)
cbar_ax.invert_yaxis()

# Disabilita la griglia prima di salvare il PDF
for ax in axs.flatten():
    ax.grid(False)

# Mostriamo il grafico
plt.tight_layout(rect=[0, 0, 0.9, 1])  # Lasciamo spazio per la legenda globale
plt.savefig('weight_aggregation.pdf', format='pdf', bbox_inches='tight')

plt.show()
