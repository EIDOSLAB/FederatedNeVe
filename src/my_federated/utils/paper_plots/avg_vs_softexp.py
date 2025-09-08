import matplotlib.pyplot as plt
import numpy as np

# Dati di esempio
datasets = ['CIFAR10', 'CIFAR100', 'BloodMNIST']
data_type = ['IID', 'NON-IID']
techniques = ['Avg', 'SoftMin']

# Memory footprints in MB per ciascun caso: [aux10, aux100]
avg_perf = {
    'CIFAR10': [88.72, 57.53],
    'CIFAR100': [66.59, 49.93],
    'BloodMNIST': [57.80, 58.09]
}

softmin_perf = {
    'CIFAR10': [79.56, 71.05],
    'CIFAR100': [54.74, 49.09],
    'BloodMNIST': [90.40, 60.20]
}

avg_var = {
    'CIFAR10': [1.14, 1.35],
    'CIFAR100': [2.44, 0.59],
    'BloodMNIST': [0.74, 5.08]
}

softmin_var = {
    'CIFAR10': [0.18, 1.04],
    'CIFAR100': [0.16, 0.57],
    'BloodMNIST': [0.76, 3.09]
}

# Definiamo i 4 colori per ogni combinazione
color_avg_iid = '#1f77b4'  # blu per all, aux10
color_softmin_iid = '#aec7e8'  # azzurro chiaro per all, aux100
color_avg_noniid = '#ff7f0e'  # arancione per only ll, aux10
color_softmin_noniid = '#ffbb78'  # arancione chiaro per only ll, aux100

# Parametri per il grafico
n_datasets = len(datasets)
n_sizes = len(data_type)  # 2: aux10 e aux100
barWidth = 0.23

# Posizioni base per ciascun dataset
positions = np.arange(n_datasets)
# Definiamo due offset per posizionare i gruppi (aux10 e aux100) all'interno di ogni dataset
group_offsets = np.array([-barWidth, barWidth])

plt.figure(figsize=(20, 12.5))

# Ciclo su ciascun dataset
for i, dataset in enumerate(datasets):
    # Aux10
    pos_aux10 = positions[i] + group_offsets[0]
    bar1 = plt.bar(pos_aux10 - barWidth / 2, avg_perf[dataset][0],
                   yerr=avg_var[dataset][0], capsize=5, color=color_avg_iid,
                   width=barWidth, edgecolor='grey',
                   label='WAvg, IID' if i == 0 else "")

    bar2 = plt.bar(pos_aux10 + barWidth / 2, softmin_perf[dataset][0],
                   yerr=softmin_var[dataset][0], capsize=5, color=color_avg_noniid,
                   width=barWidth, edgecolor='grey',
                   label='SoftMin, IID' if i == 0 else "")

    pos_aux100 = positions[i] + group_offsets[1]
    bar3 = plt.bar(pos_aux100 - barWidth / 2, avg_perf[dataset][1],
                   yerr=avg_var[dataset][1], capsize=5, color=color_softmin_iid,
                   width=barWidth, edgecolor='grey',
                   label='WAvg, NON-IID' if i == 0 else "")

    bar4 = plt.bar(pos_aux100 + barWidth / 2, softmin_perf[dataset][1],
                   yerr=softmin_var[dataset][1], capsize=5, color=color_softmin_noniid,
                   width=barWidth, edgecolor='grey',
                   label='SoftMin, NON-IID' if i == 0 else "")


    # Funzione per aggiungere etichette alle barre
    def add_value_labels(bars, variances):
        for bar, var in zip(bars, variances):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height + var + .5,  # Sposta sopra la deviazione standard
                     f'{height:.2f}±{var:.2f}',
                     ha='center', va='bottom', fontsize=16, rotation=0)


    add_value_labels(bar1, [avg_var[dataset][0]])
    add_value_labels(bar2, [softmin_var[dataset][0]])
    add_value_labels(bar3, [avg_var[dataset][1]])
    add_value_labels(bar4, [softmin_var[dataset][1]])

# Etichette e titolo
plt.xlabel('Dataset', fontweight='bold', fontsize=20)
plt.ylabel('Performance [%]', fontweight='bold', fontsize=20)
# plt.title('Memory Footprint Comparison by Dataset, Aux Size and Technique')
plt.xticks(positions, datasets)
plt.legend()
# Scala logaritmica sull'asse y per evidenziare anche i valori piccoli
# plt.yscale('log')
plt.tight_layout()
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=18)
plt.savefig('avg_vs_softmin.pdf', format='pdf', bbox_inches='tight')
plt.show()
