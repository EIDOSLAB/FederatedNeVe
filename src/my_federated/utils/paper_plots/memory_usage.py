import matplotlib.pyplot as plt
import numpy as np

# Dati di esempio
datasets = ['CIFAR10 (10)', 'CIFAR100 (100)', 'BloodMNIST (8)']
sizes = ['Aux-10', 'Aux-100']
techniques = ['All', 'Only-LL']

# Memory footprints in MB per ciascun caso: [aux10, aux100]
memory_all = {
    'CIFAR10 (10)': [94.75, 947.46],
    'CIFAR100 (100)': [94.75, 947.49908],
    'BloodMNIST (8)': [94.75, 947.46]
}

memory_only_ll = {
    'CIFAR10 (10)': [0.00039, 0.00381],
    'CIFAR100 (100)': [0.00381, 0.03815],
    'BloodMNIST (8)': [0.0003, 0.003]
}

# Definiamo i 4 colori per ogni combinazione
color_all_aux10 = '#1f77b4'  # blu per all, aux10
color_all_aux100 = '#aec7e8'  # azzurro chiaro per all, aux100
color_only_ll_aux10 = '#ff7f0e'  # arancione per only ll, aux10
color_only_ll_aux100 = '#ffbb78'  # arancione chiaro per only ll, aux100

# Parametri per il grafico
n_datasets = len(datasets)
n_sizes = len(sizes)  # 2: aux10 e aux100
barWidth = 0.2

# Posizioni base per ciascun dataset
positions = np.arange(n_datasets)
# Definiamo due offset per posizionare i gruppi (aux10 e aux100) all'interno di ogni dataset
group_offsets = np.array([-barWidth, barWidth])

plt.figure(figsize=(20, 12.5))

# Ciclo su ciascun dataset
for i, dataset in enumerate(datasets):
    # Aux10
    pos_aux10 = positions[i] + group_offsets[0]
    # Barre per aux10: tecnica "all" e "only ll" con barre affiancate
    bar1 = plt.bar(pos_aux10 - barWidth / 2, memory_all[dataset][0],
                   color=color_all_aux10, width=barWidth, edgecolor='grey',
                   label='All, Aux-10' if i == 0 else "")
    bar2 = plt.bar(pos_aux10 + barWidth / 2, memory_only_ll[dataset][0],
                   color=color_only_ll_aux10, width=barWidth, edgecolor='grey',
                   label='Only-LL, Aux-10' if i == 0 else "")

    # Aux100
    pos_aux100 = positions[i] + group_offsets[1]
    # Barre per aux100: tecnica "all" e "only ll" con barre affiancate
    bar3 = plt.bar(pos_aux100 - barWidth / 2, memory_all[dataset][1],
                   color=color_all_aux100, width=barWidth, edgecolor='grey',
                   label='All, Aux-100' if i == 0 else "")
    bar4 = plt.bar(pos_aux100 + barWidth / 2, memory_only_ll[dataset][1],
                   color=color_only_ll_aux100, width=barWidth, edgecolor='grey',
                   label='Only-LL, Aux-100' if i == 0 else "")


    # Funzione per aggiungere etichette alle barre
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            # Aggiungiamo il testo centrato orizzontalmente sopra la barra.
            plt.text(
                bar.get_x() + bar.get_width() / 2, height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=18, rotation=0
            )


    add_value_labels(bar1)
    add_value_labels(bar2)
    add_value_labels(bar3)
    add_value_labels(bar4)
# Etichette e titolo
plt.xlabel('Dataset', fontweight='bold', fontsize=20)
plt.ylabel('Memory Footprint (MB)', fontweight='bold', fontsize=20)
# plt.title('Memory Footprint Comparison by Dataset, Aux Size and Technique')
plt.xticks(positions, datasets)
plt.legend()
# Scala logaritmica sull'asse y per evidenziare anche i valori piccoli
plt.yscale('log')
plt.tight_layout()
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=18)
plt.savefig('memory_footprint.pdf', format='pdf', bbox_inches='tight')
plt.show()
