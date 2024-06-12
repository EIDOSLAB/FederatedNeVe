from io import BytesIO

import numpy as np
import seaborn as sns
import wandb
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from wandb.sdk.data_types.image import Image


def mplfig_2_wandbfig(figure: Figure) -> Image:
    # Salva la figura in formato PNG
    image_stream = BytesIO()
    figure.savefig(image_stream, format='png')
    image_stream.seek(0)
    image_array = plt.imread(image_stream)
    image_stream.close()
    wandb_fig = wandb.Image(image_array)
    return wandb_fig


def create_confusion_matrix_figure(confusion_matrix: np.ndarray, title: str = "Average Confusion Matrix"):
    # Crea la figura di matplotlib
    fig, ax = plt.subplots(figsize=(8, 6))

    # Crea il plot della matrice di confusione utilizzando Seaborn
    sns.heatmap(confusion_matrix, annot=True, fmt=".2f", cmap="Blues", ax=ax,
                xticklabels=range(confusion_matrix.shape[0]), yticklabels=range(confusion_matrix.shape[0]))

    # Imposta i titoli e le etichette
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Real')
    fig.tight_layout()
    return fig


def create_distribution_figure(client_distr: list[int], client_labels: list, title: str = "Client Distribution"):
    # Crea la figura di matplotlib
    fig, ax = plt.subplots(figsize=(8, 6))

    # Crea il plot della distribuzione sul subplot
    bars = ax.bar(client_labels, client_distr, color='skyblue')

    # Aggiungi l'annotazione sopra ogni barra con i valori normalizzati
    for i, bar in enumerate(bars):
        ax.text(bar.get_x() + bar.get_width() / 2, min(bar.get_height() + 1, 100),
                f'{client_distr[i]:.2f}', ha='center', va='bottom')

    # Imposta i titoli e le etichette
    ax.set_xlabel('Classes')
    ax.set_ylabel('Distribution [%]')
    ax.set_title(title)

    # Imposta le etichette delle classi
    ax.set_xticks(client_labels)
    ax.set_xticklabels(client_labels)

    # Imposta il limite superiore dell'asse y a 100
    ax.set_ylim(0, 100)
    fig.tight_layout()
    return fig
